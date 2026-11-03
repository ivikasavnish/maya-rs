//! One GPU-owner thread, continuous batching over `max_seats` seats (maya3.engine's design without CUDA graphs).
//! Per tick, for every active seat at once: cb0 logits (one batched head + one host copy), controlled stop + cb0
//! sampling on the host, depth codes 1..31 (batched), backbone feedback (batched), one-frame streaming Mimi decode
//! (per seat state, shared weights) -> int16 PCM, then one batched backbone step. A new request (or the next piece of
//! a long one) is prefilled on its own when it takes a seat. Guards as in maya3.engine.

use crate::model::{Kv, Maya, AEOS, N_CODES, N_Q};
use crate::text::{self, Prompter, GUARDS};
use anyhow::{Context, Result};
use candle_core::{Device, StreamTensor, Tensor};
use candle_transformers::models::mimi;
use rand::{rngs::StdRng, Rng, SeedableRng};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;
use tokio::sync::mpsc::UnboundedSender;

pub const SR: usize = 24_000;
pub const FRAME: usize = 1920;
const HARD_CAP: usize = 1500;
const GAP_S: f64 = 0.15;
pub const CHUNK_S: f64 = 20.0;

pub enum Event {
    Audio(Vec<i16>),
    End(Value),
    Error { code: String, error: String },
}

#[derive(Clone, Default)]
pub struct Sampling {
    pub seed: Option<u64>,
    pub temperature: Option<f32>,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub repetition_penalty: Option<f32>,
}

pub struct Job {
    pub id: String,
    pub pieces: Vec<String>,
    pub lang: String,
    /// named voice display name ("Ahana"), or None for plain
    pub voice: Option<String>,
    pub max_frames: Option<usize>,
    pub sampling: Sampling,
    pub events: UnboundedSender<Event>,
    pub cancelled: Arc<AtomicBool>,
    pub t_submit: Instant,
}

impl Job {
    fn gone(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed) || self.events.is_closed()
    }
    fn error(&self, code: &str, error: impl Into<String>) {
        let _ = self.events.send(Event::Error { code: code.into(), error: error.into() });
    }
}

#[derive(Clone)]
pub struct VoiceInfo {
    pub id: String,
    pub meta: Value,
}

#[derive(Default)]
pub struct Shared {
    pub ready: AtomicBool,
    pub failed: Mutex<Option<String>>,
    pub active: AtomicUsize,
    pub pending: AtomicUsize,
    pub completed: AtomicUsize,
    pub tick_ms: Mutex<VecDeque<f64>>,
    pub phase_ms: Mutex<Value>,
}

pub struct EngineHandle {
    pub tx: mpsc::Sender<Job>,
    pub shared: Arc<Shared>,
    pub voices: Vec<VoiceInfo>,
    pub max_seats: usize,
}

pub struct Paths {
    pub model_dir: PathBuf,
    pub mimi_file: PathBuf,
}

pub fn load_voices(model_dir: &Path) -> Result<Vec<VoiceInfo>> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(model_dir.join("voices/voices.json"))?)?;
    let items = v.get("voices").cloned().unwrap_or(v);
    let mut out = Vec::new();
    for it in items.as_array().context("voices.json")? {
        let name = ["name", "display_name", "id"]
            .iter()
            .find_map(|k| it.get(*k).and_then(Value::as_str))
            .context("voice without name")?;
        let mode = it.get("mode").and_then(Value::as_str).unwrap_or("named");
        anyhow::ensure!(mode == "named", "voice {name}: mode {mode} needs reference audio, which this server does not serve");
        let mut meta = it.clone();
        if let Some(m) = meta.as_object_mut() {
            for k in ["ecapa", "centroid", "ref_clips"] {
                m.remove(k);
            }
            m.insert("name".into(), json!(name));
        }
        out.push(VoiceInfo { id: name.to_string(), meta });
    }
    Ok(out)
}

pub fn start(paths: Paths, capacity: usize, max_seats: usize) -> Result<EngineHandle> {
    let voices = load_voices(&paths.model_dir)?;
    let shared = Arc::new(Shared::default());
    let (tx, rx) = mpsc::channel::<Job>();
    let sh = shared.clone();
    std::thread::Builder::new().name("maya-engine".into()).spawn(move || {
        let res = (|| -> Result<()> {
            let mut eng = Engine::load(&paths, capacity, max_seats)?;
            eng.warmup(&sh)?;
            sh.ready.store(true, Ordering::SeqCst);
            eprintln!("maya: ready ({max_seats} seats)");
            eng.serve(&rx, &sh)
        })();
        if let Err(e) = res {
            eprintln!("maya: engine failed: {e:#}");
            *sh.failed.lock().unwrap() = Some(format!("{e:#}"));
        }
        while let Ok(job) = rx.try_recv() {
            job.error("unavailable", "engine stopped");
        }
    })?;
    Ok(EngineHandle { tx, shared, voices, max_seats })
}

struct Cb0State {
    temp: f32,
    top_k: usize,
    top_p: f32,
    rep: f32,
    hist: Vec<i64>, // presence-penalty window (12)
    hazard: f32,
    aeos_bias: f32,
}

/// One request in a seat. Piece state is reset by `Engine::plan` for every text piece.
struct Seat {
    job: Job,
    kv: Kv,
    mimi: mimi::Model,
    rng: StdRng,
    piece: usize,
    h: Option<Tensor>, // [1, 2048] hidden that produces the next frame
    st: Cb0State,
    hist: Vec<i64>, // loop-guard cb0 tail (96)
    frames: usize,
    cap: usize,
    loose: usize,
    minimum: usize,
    suspend: Vec<(usize, usize)>,
    boost_at: Option<usize>,
    n_boost: usize,
    frames_total: usize,
    stops: Vec<Value>,
    t_first: Option<Instant>,
}

pub struct Engine {
    pub maya: Maya,
    pub mimi: mimi::Model,
    pub prompter: Prompter,
    max_seats: usize,
    kv_pool: Vec<Kv>,
    pub graphs: Option<Vec<crate::graph::FrameGraph>>,
    /// MAYA_PROFILE=1: synchronise after each phase so last_tick_ms shows GPU time per phase (slower)
    profile: bool,
}

/// cb0 controlled stop + sampling (maya3.engine.Engine._cb0) for one seat. Returns (stop code, cb0).
fn cb0_step(logits: &[f32], st: &mut Cb0State, rng: &mut StdRng, frames: usize, minimum: usize) -> (u8, u32) {
    let mut l: Vec<f32> = logits.iter().map(|x| x / st.temp).collect();
    l[AEOS] += st.aeos_bias;
    if st.rep > 0.0 {
        let mut seen = vec![false; l.len()];
        for &c in &st.hist {
            if c >= 0 && (c as usize) != AEOS {
                seen[c as usize] = true;
            }
        }
        for (i, s) in seen.iter().enumerate() {
            if *s {
                l[i] -= st.rep;
            }
        }
    }
    let mut srt = l.clone();
    srt.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let kth = srt[(st.top_k.max(1) - 1).min(l.len() - 1)];
    for x in l.iter_mut() {
        if *x < kth {
            *x = f32::NEG_INFINITY;
        }
    }
    let softmax = |l: &[f32]| {
        let m = l.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = l.iter().map(|x| (x - m).exp()).collect();
        let s: f32 = e.iter().sum();
        e.into_iter().map(|x| x / s).collect::<Vec<f32>>()
    };
    if st.top_p < 1.0 {
        let p = softmax(&l);
        let mut idx: Vec<usize> = (0..p.len()).collect();
        idx.sort_by(|a, b| p[*b].partial_cmp(&p[*a]).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0;
        for &i in &idx {
            if cum > st.top_p {
                l[i] = f32::NEG_INFINITY;
            }
            cum += p[i];
        }
    }
    let probs = softmax(&l);
    let pa = probs[AEOS];
    let arg = (1..probs.len()).fold(0, |b, i| if probs[i] > probs[b] { i } else { b });
    let can = frames >= minimum;
    if can {
        st.hazard = 1.0 - (1.0 - st.hazard) * (1.0 - pa);
    }
    let code = if can && arg == AEOS {
        2
    } else if can && pa >= 0.35 {
        3
    } else if can && st.hazard >= 0.6 {
        4
    } else {
        0
    };
    let total: f32 = probs[..N_CODES].iter().sum();
    let mut r = rng.random::<f32>() * total;
    let mut cb0 = N_CODES - 1;
    for (i, p) in probs[..N_CODES].iter().enumerate() {
        if *p > 0.0 {
            cb0 = i; // rounding: falls back to the last code with mass
            if r < *p {
                break;
            }
            r -= p;
        }
    }
    st.hist.remove(0);
    st.hist.push(cb0 as i64);
    (code, cb0 as u32)
}

impl Engine {
    pub fn load(paths: &Paths, capacity: usize, max_seats: usize) -> Result<Self> {
        let dev = if std::env::var("MAYA_DEVICE").as_deref() == Ok("cpu") { Device::Cpu } else { {
            let d = Device::new_cuda_with_stream(0)?;
            // one GPU thread, one stream: no cross-stream event bookkeeping (it also blocks CUDA graph capture)
            unsafe { d.as_cuda_device()?.disable_event_tracking() };
            d
        } };
        let t0 = Instant::now();
        let maya = Maya::load(&paths.model_dir, capacity, &dev)?;
        let mimi = mimi::load(paths.mimi_file.to_str().context("mimi path")?, Some(N_Q), &dev)?;
        let prompter = Prompter::new(&paths.model_dir.join("backbone/tokenizer.json"))?;
        // every seat's KV is allocated up front, so running out of GPU memory fails at start, not mid-call
        let kv_pool = (0..max_seats).map(|_| maya.backbone.new_kv()).collect::<Result<Vec<_>>>()?;
        eprintln!("maya: loaded in {:.1}s", t0.elapsed().as_secs_f64());
        let graphs = match crate::graph::capture_all(&maya, max_seats) {
            Ok(g) => g,
            Err(e) => {
                eprintln!("maya: CUDA graph capture failed ({e:#}); running the depth stack eagerly");
                None
            }
        };
        let profile = std::env::var("MAYA_PROFILE").as_deref() == Ok("1");
        Ok(Self { maya, mimi, prompter, max_seats, kv_pool, graphs, profile })
    }

    fn warmup(&mut self, sh: &Shared) -> Result<()> {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        let mut seats = Vec::new();
        for _ in 0..self.max_seats.min(2) {
            let job = Job {
                id: "warmup".into(),
                pieces: vec!["Hello, this is a warmup.".into()],
                lang: "en".into(),
                voice: None,
                max_frames: Some(3),
                sampling: Sampling::default(),
                events: tx.clone(),
                cancelled: Arc::new(AtomicBool::new(false)),
                t_submit: Instant::now(),
            };
            if let Some(s) = self.admit(job)? {
                seats.push(s);
            }
        }
        while !seats.is_empty() {
            self.tick(&mut seats, sh)?;
        }
        sh.completed.store(0, Ordering::SeqCst);
        sh.tick_ms.lock().unwrap().clear();
        Ok(())
    }

    fn serve(&mut self, rx: &mpsc::Receiver<Job>, sh: &Shared) -> Result<()> {
        let mut pending: VecDeque<Job> = VecDeque::new();
        let mut seats: Vec<Seat> = Vec::new();
        loop {
            if seats.is_empty() && pending.is_empty() {
                match rx.recv() {
                    Ok(j) => pending.push_back(j),
                    Err(_) => return Ok(()),
                }
            }
            while let Ok(j) = rx.try_recv() {
                pending.push_back(j);
            }
            // each prefill stalls the seats already speaking for ~25 ms: admit at most 2 per tick while any play
            let (idle, mut admitted) = (seats.is_empty(), 0);
            while seats.len() < self.max_seats && (idle || admitted < 2) {
                let Some(job) = pending.pop_front() else { break };
                admitted += 1;
                sh.pending.fetch_sub(1, Ordering::SeqCst);
                if job.gone() {
                    continue;
                }
                match self.admit(job) {
                    Ok(Some(s)) => seats.push(s),
                    Ok(None) => {}
                    Err(e) => eprintln!("maya: admission failed: {e:#}"),
                }
            }
            sh.active.store(seats.len(), Ordering::SeqCst);
            if !seats.is_empty() {
                if let Err(e) = self.tick(&mut seats, sh) {
                    // a failed tick fails the seats in it, not the engine
                    eprintln!("maya: tick failed: {e:#}");
                    for s in seats.drain(..) {
                        s.job.error("internal", format!("{e:#}"));
                        self.kv_pool.push(s.kv);
                    }
                }
            }
            sh.active.store(seats.len(), Ordering::SeqCst);
        }
    }

    /// Give `job` a seat and prefill its first piece. None = rejected (error already sent to the job).
    fn admit(&mut self, job: Job) -> Result<Option<Seat>> {
        let Some(kv) = self.kv_pool.pop() else {
            job.error("unavailable", "no free seat");
            return Ok(None);
        };
        if let Some(seed) = job.sampling.seed {
            let _ = self.maya.dev.set_seed(seed.wrapping_mul(69069).wrapping_add(1));
        }
        let rng = match job.sampling.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_os_rng(),
        };
        let mut seat = Seat {
            job,
            kv,
            mimi: self.mimi.clone(),
            rng,
            piece: 0,
            h: None,
            st: Cb0State { temp: 0.9, top_k: 64, top_p: 1.0, rep: 0.8, hist: vec![], hazard: 0.0, aeos_bias: 0.0 },
            hist: vec![],
            frames: 0,
            cap: 0,
            loose: 0,
            minimum: 0,
            suspend: vec![],
            boost_at: None,
            n_boost: 0,
            frames_total: 0,
            stops: vec![],
            t_first: None,
        };
        match self.plan(&mut seat) {
            Ok(true) => Ok(Some(seat)),
            Ok(false) => {
                self.kv_pool.push(seat.kv);
                Ok(None)
            }
            Err(e) => {
                seat.job.error("internal", format!("{e:#}"));
                self.kv_pool.push(seat.kv);
                Ok(None)
            }
        }
    }

    /// Reset the seat for its current piece and prefill the prompt. Ok(false) = request rejected (error sent).
    fn plan(&mut self, s: &mut Seat) -> Result<bool> {
        let job = &s.job;
        let text = &job.pieces[s.piece];
        let prompt_text = match &job.voice {
            Some(name) => format!("{{{name}}}: {}", text.trim()),
            None => text.clone(),
        };
        let tokens = match self.prompter.tokens(&prompt_text, &job.lang) {
            Ok(t) => t,
            Err(e) => {
                job.error("invalid_request", format!("{e:#}"));
                return Ok(false);
            }
        };
        let n = tokens.len();
        let (ef, er) = text::expected_frames(&job.lang, text.chars().count());
        s.loose = HARD_CAP.min((2.0 * (ef + er) + 40.0) as usize);
        s.cap = s.loose.min(((ef * 0.5) as usize + 8).max(text::cap_frames(text, &job.lang, GUARDS.dur_k, GUARDS.dur_margin_s)));
        if let Some(m) = job.max_frames {
            s.cap = s.cap.min(1.max(m.saturating_sub(s.frames_total)));
        }
        let capacity = self.maya.backbone.capacity;
        if n + s.cap + 2 > capacity {
            s.cap = capacity.saturating_sub(n + 2);
            if s.cap < 8 {
                job.error("invalid_request", format!("prompt of {n} tokens leaves no room in KV capacity {capacity}"));
                return Ok(false);
            }
        }
        s.minimum = 4.max((0.5 * ef) as usize);
        s.suspend = text::suspend_windows(text, &job.lang);
        let sp = &job.sampling;
        s.st = Cb0State {
            temp: sp.temperature.unwrap_or(0.9),
            top_k: sp.top_k.unwrap_or(64),
            top_p: sp.top_p.unwrap_or(1.0),
            rep: sp.repetition_penalty.unwrap_or(0.8),
            hist: vec![-1; 12],
            hazard: 0.0,
            aeos_bias: 0.0,
        };
        s.hist = vec![-1; GUARDS.hist];
        s.frames = 0;
        s.boost_at = None;
        s.n_boost = 0;
        s.kv.len = 0;
        s.mimi.reset_state();
        let prompt = self.maya.prompt(&tokens)?;
        s.h = Some(self.maya.backbone.forward(&mut [&mut s.kv], &prompt)?);
        Ok(true)
    }

    /// Close the seat's current piece. Returns true when the seat continues with its next piece.
    fn finish_piece(&mut self, s: &mut Seat, reason: &str, sh: &Shared) -> Result<bool> {
        s.stops.push(json!({"reason": reason, "frames": s.frames, "cap": s.cap, "n_boost": s.n_boost}));
        let job = &s.job;
        let limit_hit = job.max_frames.map_or(false, |m| s.frames_total >= m);
        if s.piece + 1 < job.pieces.len() && !job.gone() && !limit_hit {
            let _ = job.events.send(Event::Audio(vec![0; (GAP_S * SR as f64) as usize]));
            s.piece += 1;
            return self.plan(s);
        }
        sh.completed.fetch_add(1, Ordering::SeqCst);
        let _ = job.events.send(Event::End(json!({
            "reason": reason,
            "frames": s.frames_total,
            "audio_s": (s.frames_total * FRAME) as f64 / SR as f64,
            "ttfa_s": s.t_first.map(|t| t.duration_since(job.t_submit).as_secs_f64()),
            "wall_s": job.t_submit.elapsed().as_secs_f64(),
            "pieces": job.pieces.len(),
            "stops": s.stops,
        })));
        Ok(false)
    }

    /// One 80 ms frame for every seat. Seats that end leave `seats` (their KV goes back to the pool).
    fn tick(&mut self, seats: &mut Vec<Seat>, sh: &Shared) -> Result<()> {
        let t0 = Instant::now();
        // cancelled seats leave first
        let mut i = 0;
        while i < seats.len() {
            if seats[i].job.gone() {
                let s = seats.swap_remove(i);
                self.kv_pool.push(s.kv);
            } else {
                i += 1;
            }
        }
        if seats.is_empty() {
            return Ok(());
        }
        let dev = self.maya.dev.clone();
        let hs: Vec<Tensor> = seats.iter().map(|s| s.h.clone().unwrap()).collect();
        let h = Tensor::cat(&hs, 0)?;
        let logits: Vec<Vec<f32>> = self.maya.backbone.cb0_logits(&h)?.to_vec2()?;
        // host-side stop + cb0 for every seat
        let mut ended: Vec<(usize, String)> = Vec::new();
        let mut live: Vec<usize> = Vec::new();
        let mut cb0s: Vec<u32> = Vec::new();
        for (i, s) in seats.iter_mut().enumerate() {
            let (code, cb0) = cb0_step(&logits[i], &mut s.st, &mut s.rng, s.frames, s.minimum);
            if code != 0 {
                ended.push((i, ["", "aeos", "aeos_argmax", "aeos_tau", "aeos_hazard"][code as usize].into()));
            } else {
                live.push(i);
                cb0s.push(cb0);
            }
        }
        let t1 = Instant::now();
        let mut t2 = t1;
        let mut t3 = t1;
        let mut cont: Vec<usize> = Vec::new(); // live seats that continue to a next frame
        let mut fb_rows: Vec<usize> = Vec::new(); // their row in `fb`
        let mut fb: Option<Tensor> = None;
        if !live.is_empty() {
            let idx = Tensor::new(live.iter().map(|&i| i as u32).collect::<Vec<_>>(), &dev)?;
            let hl = h.index_select(&idx, 0)?;
            let cb = Tensor::new(cb0s.as_slice(), &dev)?;
            let codes = match self.graphs.as_ref().and_then(|g| g.get(live.len() - 1)) {
                Some(g) => {
                    let (codes, f) = g.run(&hl, &cb, &self.maya.depth.noise(live.len(), &dev)?)?;
                    fb = Some(f);
                    codes
                }
                None => {
                    let codes = self.maya.depth.frame(&hl, &cb)?;
                    fb = Some(self.maya.feedback(&codes)?);
                    codes
                }
            };
            if self.profile {
                dev.synchronize()?;
            }
            t2 = Instant::now();
            let mut pcm_t = Vec::with_capacity(live.len());
            for (r, &i) in live.iter().enumerate() {
                let c = codes.narrow(0, r, 1)?.reshape((1, N_Q, 1))?;
                let out = seats[i].mimi.decode_step(&StreamTensor::from_tensor(c))?;
                pcm_t.push(out.as_option().map(|t| t.flatten_all()).transpose()?);
            }
            let lens: Vec<usize> = pcm_t.iter().map(|p| p.as_ref().map_or(0, |t| t.elem_count())).collect();
            let present: Vec<Tensor> = pcm_t.into_iter().flatten().collect();
            let all: Vec<f32> = if present.is_empty() { vec![] } else { Tensor::cat(&present, 0)?.to_vec1()? };
            t3 = Instant::now();
            let mut off = 0;
            let now = Instant::now();
            let g = &GUARDS;
            for (r, &i) in live.iter().enumerate() {
                let s = &mut seats[i];
                let pcm16: Vec<i16> =
                    all[off..off + lens[r]].iter().map(|x| (x.clamp(-1.0, 1.0) * 32767.0).round() as i16).collect();
                off += lens[r];
                s.frames += 1;
                s.frames_total += 1;
                s.t_first.get_or_insert(now);
                let _ = s.job.events.send(Event::Audio(pcm16));
                s.hist.remove(0);
                s.hist.push(cb0s[r] as i64);
                if s.frames >= s.cap {
                    let reason = if s.job.max_frames.map_or(false, |m| s.frames_total >= m) {
                        "max_frames"
                    } else if s.cap < s.loose {
                        "cap_guard"
                    } else {
                        "cap"
                    };
                    ended.push((i, reason.into()));
                    continue;
                }
                if s.frames >= g.start
                    && s.frames % g.every == 0
                    && !s.suspend.iter().any(|(a, b)| *a <= s.frames && s.frames <= *b)
                {
                    match text::loop_detect(&s.hist, s.frames, g) {
                        Some(_) => {
                            if s.boost_at.is_none() {
                                s.boost_at = Some(s.frames);
                                s.n_boost += 1;
                                s.st.aeos_bias = g.boost;
                            }
                            if s.frames - s.boost_at.unwrap() >= g.backstop {
                                ended.push((i, "loop_backstop".into()));
                                continue;
                            }
                        }
                        None => {
                            if s.boost_at.take().is_some() {
                                s.st.aeos_bias = 0.0;
                            }
                        }
                    }
                }
                cont.push(i);
                fb_rows.push(r);
            }
        }
        // one batched backbone step for the seats that go on
        if !cont.is_empty() {
            let rows = Tensor::new(fb_rows.iter().map(|&r| r as u32).collect::<Vec<_>>(), &dev)?;
            let x = fb.as_ref().unwrap().index_select(&rows, 0)?;
            let mut kvs: Vec<&mut Kv> = Vec::with_capacity(cont.len());
            for (i, s) in seats.iter_mut().enumerate() {
                if cont.contains(&i) {
                    kvs.push(&mut s.kv);
                }
            }
            let hn = self.maya.backbone.forward(&mut kvs, &x)?;
            for (r, &i) in cont.iter().enumerate() {
                seats[i].h = Some(hn.narrow(0, r, 1)?);
            }
        }
        if self.profile {
            dev.synchronize()?;
        }
        let t4 = Instant::now();
        // ended pieces: next piece (prefill) or release
        ended.sort_by(|a, b| b.0.cmp(&a.0));
        for (i, reason) in ended {
            let mut s = seats.swap_remove(i);
            match self.finish_piece(&mut s, &reason, sh) {
                Ok(true) => seats.push(s),
                Ok(false) => self.kv_pool.push(s.kv),
                Err(e) => {
                    s.job.error("internal", format!("{e:#}"));
                    self.kv_pool.push(s.kv);
                }
            }
        }
        let ms = |a: Instant, b: Instant| ((b - a).as_secs_f64() * 1e5).round() / 100.0;
        let total = ms(t0, Instant::now());
        {
            let mut tk = sh.tick_ms.lock().unwrap();
            tk.push_back(total);
            if tk.len() > 2000 {
                tk.pop_front();
            }
        }
        *sh.phase_ms.lock().unwrap() = json!({
            "batch": live.len(), "cb0": ms(t0, t1), "depth": ms(t1, t2), "mimi": ms(t2, t3), "backbone": ms(t3, t4),
            "total": total,
        });
        Ok(())
    }
}
