//! Text-side pieces of the Maya3 engine: prompt tokens, the two duration priors, long-text splitting, the loop
//! guard and its suspend windows, and script-based language detection. Ports of maya3.guards,
//! scripts/s2_serve_controls (split_text / expected_frames) and maya3.server.detect_lang.

use anyhow::{anyhow, Result};
use regex::Regex;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::OnceLock;

pub const LANGS: [&str; 12] = ["as", "bn", "en", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"];
pub const FRAME_S: f64 = 0.08;
const FPS: f64 = 12.5;
const TERMINALS: &str = "।.!?;॥";

// ---------------------------------------------------------------------------------------------- tokens
pub struct Prompter {
    tok: tokenizers::Tokenizer,
    audio_id: u32,
    lang_ids: HashMap<String, u32>,
}

impl Prompter {
    pub fn new(tokenizer_json: &std::path::Path) -> Result<Self> {
        let tok = tokenizers::Tokenizer::from_file(tokenizer_json).map_err(|e| anyhow!("tokenizer: {e}"))?;
        let id = |t: &str| tok.token_to_id(t).ok_or_else(|| anyhow!("token {t} missing from tokenizer"));
        let audio_id = id("<|audio|>")?;
        let mut lang_ids = HashMap::new();
        for l in LANGS {
            lang_ids.insert(l.to_string(), id(&format!("<|lang:{l}|>"))?);
        }
        Ok(Self { tok, audio_id, lang_ids })
    }

    /// `<|lang:xx|> {text} <|audio|>`; a named voice's text is "{Name}: text" (NamedSpeakerBuilder).
    pub fn tokens(&self, text: &str, lang: &str) -> Result<Vec<u32>> {
        let mut out = vec![*self.lang_ids.get(lang).ok_or_else(|| anyhow!("unsupported language {lang}"))?];
        let enc = self.tok.encode(text, false).map_err(|e| anyhow!("tokenize: {e}"))?;
        out.extend_from_slice(enc.get_ids());
        out.push(self.audio_id);
        Ok(out)
    }
}

// ---------------------------------------------------------------------------------------------- priors
struct FramesPrior {
    a: f64,
    b: f64,
    resid_p99: f64,
}

fn frames_prior() -> &'static HashMap<String, FramesPrior> {
    static P: OnceLock<HashMap<String, FramesPrior>> = OnceLock::new();
    P.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("../assets/text_frames_prior.json")).unwrap();
        v.as_object()
            .unwrap()
            .iter()
            .map(|(k, d)| {
                (k.clone(), FramesPrior {
                    a: d["affine_a"].as_f64().unwrap(),
                    b: d["affine_b"].as_f64().unwrap(),
                    resid_p99: d["resid_p99"].as_f64().unwrap(),
                })
            })
            .collect()
    })
}

/// (expected frames, p99 residual) for `nchar` characters (s2_serve_controls.expected_frames).
pub fn expected_frames(lang: &str, nchar: usize) -> (f64, f64) {
    let (a, b, r) = frames_prior().get(lang).map(|p| (p.a, p.b, p.resid_p99)).unwrap_or((2.0, 0.72, 90.0));
    (a + b * nchar as f64, r)
}

struct DurPrior {
    coef: HashMap<String, [f64; 5]>,
    default: [f64; 5],
}

fn dur_prior() -> &'static DurPrior {
    static P: OnceLock<DurPrior> = OnceLock::new();
    P.get_or_init(|| {
        let v: Value = serde_json::from_str(include_str!("../assets/duration_prior.json")).unwrap();
        let mut coef = HashMap::new();
        let mut sum = [0.0; 5];
        for (l, d) in v["langs"].as_object().unwrap() {
            let mut c = [0.0; 5];
            for (i, x) in d["coef"].as_array().unwrap().iter().enumerate() {
                c[i] = x.as_f64().unwrap();
                sum[i] += c[i];
            }
            coef.insert(l.clone(), c);
        }
        let n = coef.len() as f64;
        DurPrior { coef, default: sum.map(|s| s / n) }
    })
}

fn is_pause(c: char) -> bool {
    matches!(c, ',' | ';' | ':' | '.' | '!' | '?' | '।' | '॥' | '…' | '-')
}

fn text_feats(t: &str) -> [f64; 5] {
    let mut letters = 0usize;
    let mut latin = 0usize;
    let mut pauses = 0usize;
    let mut digits = 0usize;
    for ch in t.chars() {
        let o = ch as u32;
        if ch.is_alphabetic() || (0x900..=0xDFF).contains(&o) {
            letters += 1;
        }
        if ch.is_ascii_alphabetic() {
            latin += 1;
        }
        if is_pause(ch) {
            pauses += 1;
        }
        if ch.is_numeric() {
            digits += 1;
        }
    }
    [1.0, (letters - latin) as f64, latin as f64, pauses as f64, digits as f64]
}

/// Expected speech seconds (guards.DurationPrior.expected_s).
pub fn expected_s(text: &str, lang: &str) -> f64 {
    let p = dur_prior();
    let c = p.coef.get(lang).unwrap_or(&p.default);
    let f = text_feats(text);
    (0..5).map(|i| f[i] * c[i]).sum::<f64>().max(0.35)
}

pub fn cap_frames(text: &str, lang: &str, k: f64, margin_s: f64) -> usize {
    ((k * expected_s(text, lang) + margin_s) / FRAME_S) as usize
}

// ---------------------------------------------------------------------------------------------- loop guard
pub struct GuardConfig {
    pub dur_k: f64,
    pub dur_margin_s: f64,
    pub stall: usize,
    pub period_win: usize,
    pub period_thr: f64,
    pub pmin: usize,
    pub pmax: usize,
    pub start: usize,
    pub every: usize,
    pub boost: f32,
    pub backstop: usize,
    pub hist: usize,
}

pub const GUARDS: GuardConfig = GuardConfig {
    dur_k: 2.2,
    dur_margin_s: 1.2,
    stall: 32,
    period_win: 64,
    period_thr: 0.7,
    pmin: 2,
    pmax: 16,
    start: 32,
    every: 2,
    boost: 4.0,
    backstop: 24,
    hist: 96,
};

/// guards.loop_detect_batch for one seat. `w`: the last `hist` cb0 codes, newest last (-1 = empty).
pub fn loop_detect(w: &[i64], n: usize, g: &GuardConfig) -> Option<String> {
    let st = g.stall;
    if n < 12.max(st) {
        return None;
    }
    let tail = &w[w.len() - st..];
    if tail.iter().all(|&c| c == tail[0]) {
        return Some("stall".into());
    }
    if n >= 12.max(st).max(g.period_win) {
        let win = &w[w.len() - g.period_win..];
        for p in g.pmin..=g.pmax {
            let m = win.len() - p;
            let same = (0..m).filter(|&i| win[i + p] == win[i]).count();
            if same as f64 / m as f64 >= g.period_thr {
                return Some(format!("period:{p}"));
            }
        }
    }
    None
}

/// Python's re `\w`: alphanumeric or underscore.
fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// guards.suspend_windows_v2: frame windows where the loop guard takes no action.
pub fn suspend_windows(text: &str, lang: &str) -> Vec<(usize, usize)> {
    static RES: OnceLock<(Regex, Regex, Regex)> = OnceLock::new();
    let (tag_re, filler_re, speaker_re) = RES.get_or_init(|| {
        (
            Regex::new(r"\[([a-z _]+)\]").unwrap(),
            Regex::new(concat!(
                r"(?i)(?:hmm+|hm+|umm+|um|uh+|uhh|erm|ह्म्म?|हम्म|उम्म?|ಹ್ಮ್|ಉಮ್|ஹ்ம்|ம்ம்|உம்|উমম?|হুম|ଉମ୍|ହମ୍|ਹਮ|ਉਮ|",
                r"હમ્મ|ઉમ|ఉమ్|హ్మ్|ഉം|ഹ്മ്)"
            ))
            .unwrap(),
            Regex::new(r"(?:[.!?।॥]\s+|\n)([^\s:,.!?।]{1,24}):\s").unwrap(),
        )
    });
    let mut ev: Vec<(usize, f64)> = Vec::new();
    // style and event tags both hold 4 s (v2b defaults)
    for m in tag_re.find_iter(text) {
        ev.push((m.start(), 4.0));
    }
    for m in filler_re.find_iter(text) {
        let before = text[..m.start()].chars().next_back();
        let after = text[m.end()..].chars().next();
        if before.map_or(true, |c| !is_word(c)) && after.map_or(true, |c| !is_word(c)) {
            ev.push((m.start(), 2.5));
        }
    }
    for c in speaker_re.captures_iter(text) {
        ev.push((c.get(1).unwrap().start(), 2.0));
    }
    ev.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ev.into_iter()
        .map(|(pos, hold)| {
            let t = if pos > 0 { expected_s(&text[..pos], lang) } else { 0.0 };
            let a = (((0.8 * t - 0.5) / FRAME_S) as i64).max(0) as usize;
            let b = ((1.5 * t + hold) / FRAME_S) as usize;
            (a, b)
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------- splitting
/// s2_serve_controls.split_text: pieces of about `chunk_s` seconds at sentence terminals, then at whitespace.
pub fn split_text(text: &str, lang: &str, chunk_s: f64) -> Vec<String> {
    if chunk_s <= 0.0 {
        return vec![text.to_string()];
    }
    let target = chunk_s * FPS;
    let ef = |s: &str| expected_frames(lang, s.chars().count()).0;
    // sentence pieces: split at whitespace runs that follow a terminal
    let mut pieces: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut prev: Option<char> = None;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c.is_whitespace() && prev.map_or(false, |p| TERMINALS.contains(p)) {
            while chars.peek().map_or(false, |c| c.is_whitespace()) {
                chars.next();
            }
            pieces.push(std::mem::take(&mut cur));
            prev = None;
            continue;
        }
        cur.push(c);
        prev = Some(c);
    }
    pieces.push(cur);
    let pieces: Vec<String> = pieces.iter().map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect();
    let mut out = Vec::new();
    let mut cur = String::new();
    for p in pieces {
        let cand = if cur.is_empty() { p.clone() } else { format!("{cur} {p}") };
        if !cur.is_empty() && ef(&cand) > target {
            out.push(std::mem::replace(&mut cur, p));
        } else {
            cur = cand;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    let mut fin = Vec::new();
    for piece in out {
        if ef(&piece) <= target * 1.3 {
            fin.push(piece);
            continue;
        }
        let mut cur = String::new();
        for w in piece.split_whitespace() {
            let cand = if cur.is_empty() { w.to_string() } else { format!("{cur} {w}") };
            if !cur.is_empty() && ef(&cand) > target {
                fin.push(std::mem::replace(&mut cur, w.to_string()));
            } else {
                cur = cand;
            }
        }
        if !cur.is_empty() {
            fin.push(cur);
        }
    }
    fin
}

// ---------------------------------------------------------------------------------------------- language
const SCRIPTS: [(u32, u32, &str); 9] = [
    (0x0900, 0x097F, "hi"),
    (0x0980, 0x09FF, "bn"),
    (0x0A00, 0x0A7F, "pa"),
    (0x0A80, 0x0AFF, "gu"),
    (0x0B00, 0x0B7F, "or"),
    (0x0B80, 0x0BFF, "ta"),
    (0x0C00, 0x0C7F, "te"),
    (0x0C80, 0x0CFF, "kn"),
    (0x0D00, 0x0D7F, "ml"),
];

pub fn detect_lang(text: &str) -> &'static str {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    let mut bump = |l: &'static str| match counts.iter_mut().find(|(k, _)| *k == l) {
        Some(e) => e.1 += 1,
        None => counts.push((l, 1)),
    };
    for ch in text.chars() {
        let o = ch as u32;
        if o < 0x80 {
            if ch.is_ascii_alphabetic() {
                bump("en");
            }
            continue;
        }
        if let Some((_, _, l)) = SCRIPTS.iter().find(|(lo, hi, _)| (*lo..=*hi).contains(&o)) {
            bump(l);
        }
    }
    // most_common: highest count, first seen wins ties
    let mut best: Option<(&'static str, usize)> = None;
    for (l, c) in counts {
        if best.map_or(true, |(_, b)| c > b) {
            best = Some((l, c));
        }
    }
    best.map(|b| b.0).unwrap_or("en")
}
