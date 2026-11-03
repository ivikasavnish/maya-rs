//! Maya2 Native (upstream id "maya3") in candle: Qwen3-1.7B backbone + cb0 head, the DepthV2 stack for codebooks
//! 1..31, the backbone-side audio input embedding and the speaker anchor. Inference only, one seat.
//!
//! Numerics follow the Python serving path (maya3.engine + serve.fast_depth): backbone in bf16, depth in fp16,
//! speaker cross-attention folded into per-layer constants (one unmasked memory token => softmax is exactly 1 =>
//! the block adds o(v(mem)) whatever the query is).

use anyhow::{Context, Result};
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{linear_no_bias, ops, Linear, VarBuilder};
use std::path::Path;

pub const N_Q: usize = 32;
pub const N_CODES: usize = 2048;
pub const AEOS: usize = N_CODES;

// ---------------------------------------------------------------------------------------------- backbone
pub const HIDDEN: usize = 2048;
const LAYERS: usize = 28;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;
const INTER: usize = 6144;
const RMS_EPS: f64 = 1e-6;
const ROPE_THETA: f64 = 1_000_000.0;

struct QwenLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Tensor,
    k_norm: Tensor,
    in_norm: Tensor,
    post_norm: Tensor,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// One seat's backbone KV cache: [1, KV_HEADS, capacity, HEAD_DIM] per layer, filled up to `len`.
pub struct Kv {
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    pub len: usize,
}

pub struct Backbone {
    layers: Vec<QwenLayer>,
    norm: Tensor,
    cos: Tensor,
    sin: Tensor,
    cb0_head: Linear,
    dtype: DType,
    dev: Device,
    pub capacity: usize,
}

fn repeat_kv(x: &Tensor, n: usize) -> Result<Tensor> {
    if n == 1 {
        return Ok(x.clone());
    }
    let (b, h, t, d) = x.dims4()?;
    Ok(x.unsqueeze(2)?.expand((b, h, n, t, d))?.reshape((b, h * n, t, d))?)
}

/// softmax(q k^T * scale [+ causal]) v with the score/softmax in fp32. q [1,H,T,D], k/v [1,H,L,D].
fn attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64, causal: bool) -> Result<Tensor> {
    let dt = q.dtype();
    let t = q.dim(2)?;
    let l = k.dim(2)?;
    let mut att = (q.matmul(&k.t()?)?.to_dtype(DType::F32)? * scale)?;
    if causal && t > 1 {
        // query i (absolute l - t + i) sees keys j <= l - t + i
        let off = l - t;
        let mask: Vec<f32> = (0..t)
            .flat_map(|i| (0..l).map(move |j| if j > off + i { f32::NEG_INFINITY } else { 0.0 }))
            .collect();
        let mask = Tensor::from_vec(mask, (t, l), q.device())?;
        att = att.broadcast_add(&mask)?;
    }
    let att = ops::softmax_last_dim(&att)?.to_dtype(dt)?;
    Ok(att.matmul(v)?)
}

impl Backbone {
    fn load(vb: VarBuilder, capacity: usize, dev: &Device) -> Result<Self> {
        let lm = vb.pp("backbone.lm.model");
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let l = lm.pp(format!("layers.{i}"));
            let a = l.pp("self_attn");
            let m = l.pp("mlp");
            layers.push(QwenLayer {
                q: linear_no_bias(HIDDEN, HEADS * HEAD_DIM, a.pp("q_proj"))?,
                k: linear_no_bias(HIDDEN, KV_HEADS * HEAD_DIM, a.pp("k_proj"))?,
                v: linear_no_bias(HIDDEN, KV_HEADS * HEAD_DIM, a.pp("v_proj"))?,
                o: linear_no_bias(HEADS * HEAD_DIM, HIDDEN, a.pp("o_proj"))?,
                q_norm: a.get(HEAD_DIM, "q_norm.weight")?,
                k_norm: a.get(HEAD_DIM, "k_norm.weight")?,
                in_norm: l.get(HIDDEN, "input_layernorm.weight")?,
                post_norm: l.get(HIDDEN, "post_attention_layernorm.weight")?,
                gate: linear_no_bias(HIDDEN, INTER, m.pp("gate_proj"))?,
                up: linear_no_bias(HIDDEN, INTER, m.pp("up_proj"))?,
                down: linear_no_bias(INTER, HIDDEN, m.pp("down_proj"))?,
            });
        }
        let inv: Vec<f32> = (0..HEAD_DIM / 2)
            .map(|i| 1.0 / ROPE_THETA.powf(2.0 * i as f64 / HEAD_DIM as f64) as f32)
            .collect();
        let inv = Tensor::from_vec(inv, (1, HEAD_DIM / 2), dev)?;
        let pos = Tensor::arange(0u32, capacity as u32, dev)?.to_dtype(DType::F32)?.reshape((capacity, 1))?;
        let freqs = pos.matmul(&inv)?;
        Ok(Self {
            layers,
            norm: lm.get(HIDDEN, "norm.weight")?,
            cos: freqs.cos()?.to_dtype(vb.dtype())?,
            sin: freqs.sin()?.to_dtype(vb.dtype())?,
            cb0_head: linear_no_bias(HIDDEN, N_CODES + 1, vb.pp("backbone.cb0_head"))?,
            dtype: vb.dtype(),
            dev: dev.clone(),
            capacity,
        })
    }

    pub fn new_kv(&self) -> Result<Kv> {
        let z = || Tensor::zeros((1, KV_HEADS, self.capacity, HEAD_DIM), self.dtype, &self.dev);
        Ok(Kv {
            k: (0..LAYERS).map(|_| z()).collect::<candle_core::Result<_>>()?,
            v: (0..LAYERS).map(|_| z()).collect::<candle_core::Result<_>>()?,
            len: 0,
        })
    }

    /// x [B, T, HIDDEN] (bf16), row b appended to seat kvs[b] at its own position (B > 1 requires T == 1).
    /// Returns the final-norm hidden of each row's LAST position [B, HIDDEN]. The projections and MLP run batched;
    /// RoPE and attention run per seat (each seat has its own length).
    pub fn forward(&self, kvs: &mut [&mut Kv], x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        anyhow::ensure!(b == kvs.len() && (b == 1 || t == 1), "backbone batch shape");
        for kv in kvs.iter() {
            anyhow::ensure!(kv.len + t <= self.capacity, "backbone KV capacity exceeded");
        }
        let scale = 1.0 / (HEAD_DIM as f64).sqrt();
        let rope: Vec<(Tensor, Tensor)> = kvs
            .iter()
            .map(|kv| Ok((self.cos.narrow(0, kv.len, t)?, self.sin.narrow(0, kv.len, t)?)))
            .collect::<Result<_>>()?;
        // decode step of several seats: one rope launch for all of them with per-row positions [B, 1, D/2]
        let rope_b = if b > 1 {
            let c: Vec<Tensor> = rope.iter().map(|r| r.0.clone()).collect();
            let s: Vec<Tensor> = rope.iter().map(|r| r.1.clone()).collect();
            Some((Tensor::cat(&c, 0)?.unsqueeze(1)?.contiguous()?, Tensor::cat(&s, 0)?.unsqueeze(1)?.contiguous()?))
        } else {
            None
        };
        let groups = HEADS / KV_HEADS;
        let mut x = x.clone();
        for (li, l) in self.layers.iter().enumerate() {
            let h = ops::rms_norm(&x, &l.in_norm, RMS_EPS as f32)?;
            let q = l.q.forward(&h)?.reshape((b, t, HEADS, HEAD_DIM))?;
            let k = l.k.forward(&h)?.reshape((b, t, KV_HEADS, HEAD_DIM))?;
            let v = l.v.forward(&h)?.reshape((b, t, KV_HEADS, HEAD_DIM))?;
            let q = ops::rms_norm(&q.contiguous()?, &l.q_norm, RMS_EPS as f32)?.transpose(1, 2)?.contiguous()?;
            let k = ops::rms_norm(&k.contiguous()?, &l.k_norm, RMS_EPS as f32)?.transpose(1, 2)?.contiguous()?;
            let v = v.transpose(1, 2)?.contiguous()?;
            let (q, k) = match &rope_b {
                Some((c, s)) => (candle_nn::rotary_emb::rope(&q, c, s)?, candle_nn::rotary_emb::rope(&k, c, s)?),
                None => {
                    let (c, s) = &rope[0];
                    (candle_nn::rotary_emb::rope(&q, c, s)?, candle_nn::rotary_emb::rope(&k, c, s)?)
                }
            };
            let mut outs = Vec::with_capacity(b);
            for (i, kv) in kvs.iter_mut().enumerate() {
                let (qi, ki, vi) = if b == 1 { (q.clone(), k.clone(), v.clone()) } else { (q.narrow(0, i, 1)?, k.narrow(0, i, 1)?, v.narrow(0, i, 1)?) };
                kv.k[li].slice_set(&ki.contiguous()?, 2, kv.len)?;
                kv.v[li].slice_set(&vi.contiguous()?, 2, kv.len)?;
                let kk = kv.k[li].narrow(2, 0, kv.len + t)?;
                let vv = kv.v[li].narrow(2, 0, kv.len + t)?;
                if t == 1 {
                    // GQA without materialising repeated K/V: the `groups` query heads of a KV head become its rows
                    let qg = qi.reshape((1, KV_HEADS, groups, HEAD_DIM))?;
                    outs.push(attention(&qg, &kk, &vv, scale, false)?.reshape((1, HEADS, 1, HEAD_DIM))?);
                } else {
                    let kk = repeat_kv(&kk, groups)?.contiguous()?;
                    let vv = repeat_kv(&vv, groups)?.contiguous()?;
                    outs.push(attention(&qi.contiguous()?, &kk, &vv, scale, true)?);
                }
            }
            let a = if b == 1 { outs.pop().unwrap() } else { Tensor::cat(&outs, 0)? };
            let a = a.transpose(1, 2)?.reshape((b, t, HEADS * HEAD_DIM))?;
            x = (x + l.o.forward(&a)?)?;
            let h = ops::rms_norm(&x, &l.post_norm, RMS_EPS as f32)?;
            let m = (l.gate.forward(&h)?.silu()? * l.up.forward(&h)?)?;
            x = (x + l.down.forward(&m)?)?;
        }
        for kv in kvs.iter_mut() {
            kv.len += t;
        }
        let last = x.i((.., t - 1, ..))?.contiguous()?;
        Ok(ops::rms_norm(&last, &self.norm, RMS_EPS as f32)?)
    }

    /// cb0 + AEOS logits (fp32) [B, 2049]
    pub fn cb0_logits(&self, h: &Tensor) -> Result<Tensor> {
        Ok(self.cb0_head.forward(h)?.to_dtype(DType::F32)?)
    }
}

// ---------------------------------------------------------------------------------------------- depth
const DD: usize = 1024;
const DH: usize = 16;
const DFFN: usize = 3072;
const DLAYERS: usize = 4;
const GROUPS: usize = 11;
const RANK: usize = 128;
const LN_EPS: f32 = 1e-5;

fn group_of_level(level: usize) -> usize {
    match level {
        1..=8 => level - 1,
        9..=16 => 8,
        17..=24 => 9,
        _ => 10,
    }
}

struct Ln {
    w: Tensor,
    b: Tensor,
}

impl Ln {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self { w: vb.get(DD, "weight")?, b: vb.get(DD, "bias")? })
    }
    fn fwd(&self, x: &Tensor) -> Result<Tensor> {
        Ok(ops::layer_norm(&x.contiguous()?, &self.w, &self.b, LN_EPS)?)
    }
}

struct SelfBlock {
    n1: Ln,
    n2: Ln,
    qkv: Linear,
    o: Linear,
    f1: Linear,
    f2: Linear,
}

pub struct Depth {
    in_proj: Linear,
    level0: Tensor,
    embed: Tensor, // [31, 2048, 1024]: code_up(code_lo) + level_emb[1..]
    groups: Vec<Vec<SelfBlock>>,
    cross: Vec<Tensor>, // per layer [1024]
    norm: Ln,
    heads: Vec<Linear>,
    neg_inf: Tensor,
    pub temp: f64,
    pub top_k: usize,
}

impl Depth {
    fn load(vb: VarBuilder, vb32: VarBuilder, dev: &Device) -> Result<Self> {
        let d = vb.pp("depth");
        let d32 = vb32.pp("depth");
        let dt = vb.dtype();
        let code_lo = d32.get(((N_Q - 1) * N_CODES, RANK), "code_lo.weight")?;
        let code_up = d32.get((DD, RANK), "code_up.weight")?;
        let level = d32.get((N_Q, DD), "level_emb.weight")?;
        let table = code_lo.matmul(&code_up.t()?)?.reshape((N_Q - 1, N_CODES, DD))?;
        let embed = table.broadcast_add(&level.narrow(0, 1, N_Q - 1)?.unsqueeze(1)?)?.to_dtype(dt)?.contiguous()?;
        let level0 = level.i(0)?.to_dtype(dt)?;
        let mut groups = Vec::new();
        for g in 0..GROUPS {
            let mut blocks = Vec::new();
            for l in 0..DLAYERS {
                let b = d.pp(format!("groups.{g}.{l}"));
                blocks.push(SelfBlock {
                    n1: Ln::load(b.pp("n1"))?,
                    n2: Ln::load(b.pp("n2"))?,
                    qkv: linear_no_bias(DD, 3 * DD, b.pp("qkv"))?,
                    o: linear_no_bias(DD, DD, b.pp("o"))?,
                    f1: linear_no_bias(DD, DFFN, b.pp("f1"))?,
                    f2: linear_no_bias(DFFN, DD, b.pp("f2"))?,
                });
            }
            groups.push(blocks);
        }
        // named / plain voices: the speaker memory is the learned null token
        let mem = d.get((1, 1, DD), "null_mem")?;
        let mut cross = Vec::new();
        for l in 0..DLAYERS {
            let c = d.pp(format!("cross.{l}"));
            let kv = linear_no_bias(DD, 2 * DD, c.pp("kv"))?;
            let o = linear_no_bias(DD, DD, c.pp("o"))?;
            let v = kv.forward(&mem)?.narrow(D::Minus1, DD, DD)?;
            cross.push(o.forward(&v)?.reshape(DD)?);
        }
        let heads = (0..N_Q - 1)
            .map(|i| linear_no_bias(DD, N_CODES, d.pp(format!("heads.{i}"))))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            in_proj: linear_no_bias(HIDDEN, DD, d.pp("in_proj"))?,
            level0,
            embed,
            groups,
            cross,
            norm: Ln::load(d.pp("norm"))?,
            heads,
            neg_inf: Tensor::full(f32::NEG_INFINITY, (1, N_CODES), dev)?,
            temp: 0.7,
            top_k: 64,
        })
    }

    /// Top-k + temperature sampling on the GPU (Gumbel-max == multinomial over the masked softmax).
    /// logits [B, V] f32, u [B, V] uniform noise in (0, 1) -> [B] u32.
    fn sample(&self, logits: &Tensor, u: &Tensor) -> Result<Tensor> {
        let logits = (logits / self.temp)?;
        if self.top_k <= 1 {
            return Ok(logits.argmax(D::Minus1)?);
        }
        let (sorted, _) = logits.sort_last_dim(false)?;
        let kth = sorted.narrow(1, self.top_k.min(N_CODES) - 1, 1)?;
        let keep = logits.broadcast_ge(&kth)?;
        let g = u.log()?.neg()?.log()?.neg()?;
        let s = keep.where_cond(&(logits + g)?, &self.neg_inf.broadcast_as(keep.shape())?)?;
        Ok(s.argmax(D::Minus1)?)
    }

    /// Fresh sampling noise for one frame of B seats: [B, 31, 2048] f32 in (0, 1).
    pub fn noise(&self, b: usize, dev: &Device) -> Result<Tensor> {
        Ok(Tensor::rand(1e-7f32, 1.0f32 - 1e-7, (b, N_Q - 1, N_CODES), dev)?)
    }

    /// Codebooks 1..31 for one frame of B seats with fresh noise (eager path).
    pub fn frame(&self, h: &Tensor, cb0: &Tensor) -> Result<Tensor> {
        let u = self.noise(h.dim(0)?, h.device())?;
        self.frame_with(h, cb0, &u)
    }

    /// Codebooks 1..31 for one frame of B seats. h [B, 2048] backbone hidden (bf16), cb0 [B] u32, u noise
    /// [B, 31, 2048] -> codes [B, 32] u32. Shapes depend on B only, so this is what the CUDA graphs capture.
    pub fn frame_with(&self, h: &Tensor, cb0: &Tensor, u: &Tensor) -> Result<Tensor> {
        let b = h.dim(0)?;
        let dt = self.level0.dtype();
        let scale = 1.0 / ((DD / DH) as f64).sqrt();
        let mut prefix: Vec<Tensor> = Vec::with_capacity(N_Q);
        prefix.push(self.in_proj.forward(&h.to_dtype(dt)?)?.broadcast_add(&self.level0)?);
        prefix.push(self.embed.i(0)?.index_select(cb0, 0)?);
        let mut codes = vec![cb0.clone()];
        let mut kv: Vec<Option<(Tensor, Tensor)>> = vec![None; DLAYERS];
        let mut prev: Option<usize> = None;
        for level in 1..N_Q {
            let g = group_of_level(level);
            let n = level + 1;
            let incremental = prev == Some(g);
            let reused_later = level + 1 < N_Q && group_of_level(level + 1) == g;
            let mut x = if incremental { prefix[level].unsqueeze(1)? } else { Tensor::stack(&prefix[..n], 1)? };
            for (li, blk) in self.groups[g].iter().enumerate() {
                let t = x.dim(1)?;
                let qkv = blk.qkv.forward(&blk.n1.fwd(&x)?)?.reshape((b, t, 3, DH, DD / DH))?;
                let mut q = qkv.i((.., .., 0))?.transpose(1, 2)?.contiguous()?;
                let k = qkv.i((.., .., 1))?.transpose(1, 2)?.contiguous()?;
                let v = qkv.i((.., .., 2))?.transpose(1, 2)?.contiguous()?;
                let a = if incremental {
                    let (kc, vc) = kv[li].take().context("depth kv")?;
                    let k = Tensor::cat(&[&kc, &k], 2)?;
                    let v = Tensor::cat(&[&vc, &v], 2)?;
                    let a = attention(&q, &k, &v, scale, false)?;
                    kv[li] = Some((k, v));
                    a
                } else {
                    if reused_later {
                        kv[li] = Some((k.clone(), v.clone()));
                    }
                    if li == DLAYERS - 1 {
                        q = q.narrow(2, t - 1, 1)?;
                        x = x.narrow(1, t - 1, 1)?;
                        attention(&q, &k, &v, scale, false)?
                    } else {
                        attention(&q, &k, &v, scale, true)?
                    }
                };
                let t2 = x.dim(1)?;
                let a = a.transpose(1, 2)?.reshape((b, t2, DD))?;
                x = (x + blk.o.forward(&a)?)?;
                let f = blk.f2.forward(&blk.f1.forward(&blk.n2.fwd(&x)?)?.gelu_erf()?)?;
                x = (x + f)?.broadcast_add(&self.cross[li])?;
            }
            let t = x.dim(1)?;
            let last = x.i((.., t - 1, ..))?;
            let logits = self.heads[level - 1].forward(&self.norm.fwd(&last)?)?.to_dtype(DType::F32)?;
            let code = self.sample(&logits, &u.i((.., level - 1, ..))?)?;
            if n < N_Q {
                prefix.push(self.embed.i(level)?.index_select(&code, 0)?);
            }
            codes.push(code);
            prev = Some(g);
        }
        Ok(Tensor::stack(&codes, 1)?)
    }
}

// ---------------------------------------------------------------------------------------------- whole model
pub struct Maya {
    pub dev: Device,
    pub backbone: Backbone,
    pub depth: Depth,
    embed_tokens: Tensor, // CPU, bf16 [vocab, 2048]
    audio_table: Tensor,  // [32*2048, 2048] bf16
    audio_norm: Tensor,
    audio_offsets: Tensor,
    anchor: Tensor, // [2048] bf16: speaker anchor for a null speaker (named / plain voices)
}

impl Maya {
    pub fn load(dir: &Path, capacity: usize, dev: &Device) -> Result<Self> {
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json"))?)?;
        let mut files: Vec<String> = index["weight_map"]
            .as_object()
            .context("weight_map")?
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        files.sort();
        files.dedup();
        let paths: Vec<_> = files.iter().map(|f| dir.join(f)).collect();
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, dev)? };
        let vb16 = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F16, dev)? };
        let vb32 = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::F32, dev)? };
        let cpu = unsafe { VarBuilder::from_mmaped_safetensors(&paths, DType::BF16, &Device::Cpu)? };
        let embed_tokens = cpu.get((194_548, HIDDEN), "backbone.lm.model.embed_tokens.weight")?;
        let backbone = Backbone::load(vb.clone(), capacity, dev)?;
        let depth = Depth::load(vb16, vb32, dev)?;
        let a = vb.pp("speaker_anchor");
        let fc1 = candle_nn::linear(192, 512, a.pp("fc1"))?;
        let fc2 = candle_nn::linear(512, HIDDEN, a.pp("fc2"))?;
        let null = a.get(192, "null_vec")?.unsqueeze(0)?;
        let anchor = fc2.forward(&fc1.forward(&null)?.gelu_erf()?)?.reshape(HIDDEN)?;
        let offsets: Vec<u32> = (0..N_Q as u32).map(|i| i * N_CODES as u32).collect();
        Ok(Self {
            dev: dev.clone(),
            backbone,
            depth,
            embed_tokens,
            audio_table: vb.get((N_Q * N_CODES, HIDDEN), "audio_in.table.weight")?,
            audio_norm: vb.get(HIDDEN, "audio_in.norm.weight")?,
            audio_offsets: Tensor::from_vec(offsets, N_Q, dev)?,
            anchor,
        })
    }

    /// Prompt embeddings [1, T, 2048] bf16: text embeddings, speaker anchor added at the final <|audio|> sentinel.
    pub fn prompt(&self, tokens: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(tokens, &Device::Cpu)?;
        let emb = self.embed_tokens.index_select(&ids, 0)?.to_device(&self.dev)?;
        let t = tokens.len();
        let last = emb.narrow(0, t - 1, 1)?.broadcast_add(&self.anchor)?;
        let emb = if t > 1 { Tensor::cat(&[&emb.narrow(0, 0, t - 1)?, &last], 0)? } else { last };
        Ok(emb.unsqueeze(0)?)
    }

    /// Backbone input for the next position from each seat's 32 codes [B, 32] u32 -> [B, 1, 2048] bf16.
    pub fn feedback(&self, codes: &Tensor) -> Result<Tensor> {
        let b = codes.dim(0)?;
        let idx = codes.broadcast_add(&self.audio_offsets)?.flatten_all()?;
        let e = self.audio_table.index_select(&idx, 0)?.reshape((b, N_Q, HIDDEN))?.to_dtype(DType::F32)?.sum(1)?;
        let e = (e.to_dtype(DType::BF16)? / (N_Q as f64).sqrt())?;
        let x = ops::rms_norm(&e.contiguous()?, &self.audio_norm, f32::EPSILON)?;
        Ok(x.unsqueeze(1)?)
    }
}
