//! Compare the Rust port with tensors dumped by scripts/make_reference.py (Python maya3, greedy decode).

use crate::engine::{Engine, Paths};
use crate::model::N_Q;
use anyhow::{Context, Result};
use candle_core::{DType, Tensor};
use serde_json::Value;
use std::path::Path;

fn vecf(v: &Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    (max, dot / (na * nb + 1e-12))
}

fn argmax(v: &[f32]) -> usize {
    (0..v.len()).fold(0, |b, i| if v[i] > v[b] { i } else { b })
}

pub fn run(reference: &str, model_dir: &Path, mimi_file: &Path) -> Result<()> {
    let r: Value = serde_json::from_str(&std::fs::read_to_string(reference)?)?;
    let mut eng = Engine::load(&Paths { model_dir: model_dir.into(), mimi_file: mimi_file.into() }, 2048, 1)?;
    let dev = eng.maya.dev.clone();
    let voice = r["voice"].as_str().unwrap();
    let text = format!("{{{voice}}}: {}", r["text"].as_str().unwrap());
    let tokens = eng.prompter.tokens(&text, r["lang"].as_str().unwrap())?;
    let want: Vec<u32> = r["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
    println!("tokens match: {} ({} tokens)", tokens == want, tokens.len());
    anyhow::ensure!(tokens == want, "tokens differ: {tokens:?} vs {want:?}");

    let prompt = eng.maya.prompt(&tokens)?;
    let rows: Vec<Vec<f32>> = prompt.to_dtype(DType::F32)?.squeeze(0)?.to_vec2()?;
    println!("prompt first row diff {:?}", diff(&rows[0], &vecf(&r["prompt_first"])));
    println!("prompt last row (anchor) diff {:?}", diff(rows.last().unwrap(), &vecf(&r["prompt_last"])));

    let mut kv = eng.maya.backbone.new_kv()?;
    let mut h = eng.maya.backbone.forward(&mut [&mut kv], &prompt)?;
    if let Some(gs) = eng.graphs.as_ref() {
        // the captured depth graph must reproduce the eager depth stack bit for bit (same h, cb0, noise)
        let f0 = &r["frames"][0];
        let rh = Tensor::new(vecf(&f0["h"]), &dev)?.to_dtype(DType::BF16)?.unsqueeze(0)?;
        let cb0 = Tensor::new(&[f0["codes"][0].as_u64().unwrap() as u32], &dev)?;
        let mut same = 0;
        for _ in 0..3 {
            let u = eng.maya.depth.noise(1, &dev)?;
            let eager: Vec<u32> = eng.maya.depth.frame_with(&rh, &cb0, &u)?.flatten_all()?.to_vec1()?;
            let (gc, gf) = gs[0].run(&rh, &cb0, &u)?;
            let graph: Vec<u32> = gc.flatten_all()?.to_vec1()?;
            let ef: Vec<f32> = eng.maya.feedback(&Tensor::new(eager.as_slice(), &dev)?.unsqueeze(0)?)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let gfv: Vec<f32> = gf.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            same += (eager == graph && ef == gfv) as usize;
        }
        println!("cuda graph vs eager depth+feedback: {same}/3 identical");
        anyhow::ensure!(same == 3, "captured depth graph differs from the eager depth stack");
    }
    if std::env::var("MAYA_BENCH").is_ok() {
        let x = Tensor::randn(0f32, 1.0, (1, 2048), &dev)?;
        let t = std::time::Instant::now();
        for _ in 0..310 {
            let _ = x.sort_last_dim(false)?;
        }
        dev.synchronize()?;
        println!("bench: sort_last_dim [1,2048] x31 = {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 10.0);
        let rh = Tensor::new(vecf(&r["frames"][0]["h"]), &dev)?.to_dtype(DType::BF16)?.unsqueeze(0)?;
        let cb0 = Tensor::new(&[1u32], &dev)?;
        let u = eng.maya.depth.noise(1, &dev)?;
        for k in [64usize, 1] {
            eng.maya.depth.top_k = k;
            let _ = eng.maya.depth.frame_with(&rh, &cb0, &u)?;
            dev.synchronize()?;
            let t = std::time::Instant::now();
            for _ in 0..10 {
                let _ = eng.maya.depth.frame_with(&rh, &cb0, &u)?;
            }
            dev.synchronize()?;
            println!("bench: eager depth frame top_k={k}: {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / 10.0);
        }
        eng.maya.depth.top_k = 64;
    }
    eng.maya.depth.top_k = 1; // greedy, like the reference
    let frames = r["frames"].as_array().context("frames")?;
    let mut code_mismatch = 0usize;
    let mut ref_codes_all = Vec::new();
    for (t, f) in frames.iter().enumerate() {
        let hv: Vec<f32> = h.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let (hmax, hcos) = diff(&hv, &vecf(&f["h"]));
        let lg: Vec<f32> = eng.maya.backbone.cb0_logits(&h)?.flatten_all()?.to_vec1()?;
        let rlg = vecf(&f["cb0_logits"]);
        let (lmax, lcos) = diff(&lg, &rlg);
        let ref_codes: Vec<u32> = f["codes"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        // depth from the REFERENCE hidden + cb0 (isolates the depth stack)
        let rh = Tensor::new(vecf(&f["h"]), &dev)?.to_dtype(DType::BF16)?.unsqueeze(0)?;
        let cb0 = Tensor::new(&[ref_codes[0]], &dev)?;
        let codes: Vec<u32> = eng.maya.depth.frame(&rh, &cb0)?.flatten_all()?.to_vec1()?;
        let mism = codes.iter().zip(&ref_codes).filter(|(a, b)| a != b).count();
        code_mismatch += mism;
        let rc = Tensor::new(ref_codes.as_slice(), &dev)?.unsqueeze(0)?;
        let fb = eng.maya.feedback(&rc)?;
        let fbv: Vec<f32> = fb.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        let (fmax, fcos) = diff(&fbv, &vecf(&f["feedback"]));
        println!(
            "frame {t:2}: h max {hmax:.4} cos {hcos:.6} | cb0 logits max {lmax:.3} cos {lcos:.6} argmax {}/{} | depth codes mismatch {mism}/31 | feedback max {fmax:.4} cos {fcos:.6}",
            argmax(&lg[..2048]),
            argmax(&rlg[..2048])
        );
        ref_codes_all.push(ref_codes);
        // teacher-force the reference feedback so frames stay comparable
        let rfb = Tensor::new(vecf(&f["feedback"]), &dev)?.to_dtype(DType::BF16)?.reshape((1, 1, 2048))?;
        h = eng.maya.backbone.forward(&mut [&mut kv], &rfb)?;
    }
    println!("depth code mismatches total {code_mismatch}/{}", 31 * frames.len());

    // batched decode (2 seats of different lengths in one step) must match stepping each seat alone
    let short = prompt.narrow(1, 0, prompt.dim(1)? - 3)?;
    let (mut a1, mut b1, mut a2, mut b2) =
        (eng.maya.backbone.new_kv()?, eng.maya.backbone.new_kv()?, eng.maya.backbone.new_kv()?, eng.maya.backbone.new_kv()?);
    for (kv, p) in [(&mut a1, &prompt), (&mut b1, &short), (&mut a2, &prompt), (&mut b2, &short)] {
        eng.maya.backbone.forward(&mut [kv], p)?;
    }
    let mut worst: f32 = 1.0;
    for t in 0..4.min(frames.len().saturating_sub(1)) {
        let x0 = Tensor::new(vecf(&frames[t]["feedback"]), &dev)?.to_dtype(DType::BF16)?.reshape((1, 1, 2048))?;
        let x1 = Tensor::new(vecf(&frames[t + 1]["feedback"]), &dev)?.to_dtype(DType::BF16)?.reshape((1, 1, 2048))?;
        let hb = eng.maya.backbone.forward(&mut [&mut a1, &mut b1], &Tensor::cat(&[&x0, &x1], 0)?)?;
        let ha = eng.maya.backbone.forward(&mut [&mut a2], &x0)?;
        let hs = eng.maya.backbone.forward(&mut [&mut b2], &x1)?;
        for (r, single) in [(0, &ha), (1, &hs)] {
            let bv: Vec<f32> = hb.narrow(0, r, 1)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            let sv: Vec<f32> = single.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
            worst = worst.min(diff(&bv, &sv).1);
        }
    }
    println!("batched vs single backbone step: worst cos {worst:.6}");
    anyhow::ensure!(worst > 0.999, "batched backbone step diverges from single-seat steps");

    eng.mimi.reset_state();
    let mut pcm = Vec::new();
    for c in &ref_codes_all {
        let t = Tensor::new(c.as_slice(), &dev)?.reshape((1, N_Q, 1))?;
        if let Some(p) = eng.mimi.decode_step(&candle_core::StreamTensor::from_tensor(t))?.as_option() {
            pcm.extend(p.flatten_all()?.to_vec1::<f32>()?);
        }
    }
    let rp = vecf(&r["pcm"]);
    let n = pcm.len().min(rp.len());
    println!("mimi: rust {} samples, ref {} samples", pcm.len(), rp.len());
    // the last frame of a non-streaming decode differs (transposed-conv tail), compare all but it
    let m = n.saturating_sub(1920);
    println!("mimi pcm diff (all but last frame) {:?}", diff(&pcm[..m], &rp[..m]));
    Ok(())
}
