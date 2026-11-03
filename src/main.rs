//! maya-rs: Rust/candle replacement for the Python maya3 speech worker. Same HTTP API subset the account server uses:
//!   POST /v1/audio/speech  {input, voice, language|lang, response_format: wav|pcm|mp3|opus|flac,
//!                           stream_format: audio|sse, normalize, max_frames, seed, temperature, top_k, top_p,
//!                           repetition_penalty}
//!   GET  /v1/audio/speech/ws  (WebSocket, many concurrent utterances per socket)
//!   GET  /v1/voices, GET /health, GET /v1/models
//! `maya-rs selftest <ref.json>` compares the port against tensors dumped from the Python stack.

mod audio;
mod engine;
mod graph;
mod model;
mod selftest;
mod text;

use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use audio::{Format, FORMATS};
use engine::{EngineHandle, Event, Job, Sampling, SR};
use futures::{SinkExt, StreamExt};
use std::collections::VecDeque;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const MAX_TEXT_CHARS: usize = 4000;
const MAX_FRAMES: u64 = 1500;

struct Args {
    host: String,
    port: u16,
    model_dir: PathBuf,
    mimi: PathBuf,
    capacity: usize,
    max_seats: usize,
    max_pending: usize,
    deadline_ms: u64,
    textnorm: Option<PathBuf>,
}

fn parse_args(mut it: impl Iterator<Item = String>) -> Result<Args> {
    let mut a = Args {
        host: "127.0.0.1".into(),
        port: 8809,
        model_dir: PathBuf::new(),
        mimi: PathBuf::new(),
        capacity: 2048,
        max_seats: 4,
        max_pending: 4,
        deadline_ms: 5000,
        textnorm: None,
    };
    while let Some(k) = it.next() {
        let mut v = || it.next().with_context(|| format!("{k} needs a value"));
        match k.as_str() {
            "--host" => a.host = v()?,
            "--port" => a.port = v()?.parse()?,
            "--model-dir" => a.model_dir = v()?.into(),
            "--mimi" => a.mimi = v()?.into(),
            "--capacity" => a.capacity = v()?.parse()?,
            "--max-seats" => a.max_seats = v()?.parse()?,
            "--max-pending" => a.max_pending = v()?.parse()?,
            "--queue-deadline-ms" => a.deadline_ms = v()?.parse()?,
            "--textnorm" => a.textnorm = Some(v()?.into()),
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    anyhow::ensure!((1..=32).contains(&a.max_seats), "--max-seats must be 1..32");
    anyhow::ensure!(a.model_dir.join("model.safetensors.index.json").exists(), "--model-dir must hold the maya2-native release");
    anyhow::ensure!(a.mimi.exists(), "--mimi must point at kyutai/mimi model.safetensors");
    Ok(a)
}

// ---------------------------------------------------------------------------------------------- normaliser bridge
/// The bundled maya3.text normaliser (pure standard-library Python) behind a line-JSON pipe to system python3.
struct Normaliser {
    script: PathBuf,
    proc: Option<(Child, ChildStdin, BufReader<ChildStdout>)>,
}

impl Normaliser {
    fn spawn(&mut self) -> Result<()> {
        let mut child = Command::new("python3")
            .arg("-I")
            .arg(&self.script)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().context("stdin")?;
        let mut out = BufReader::new(child.stdout.take().context("stdout")?);
        let mut line = String::new();
        out.read_line(&mut line)?;
        anyhow::ensure!(line.contains("ready"), "normaliser did not start");
        self.proc = Some((child, stdin, out));
        Ok(())
    }

    fn call(&mut self, text: &str, lang: &str) -> Result<String> {
        if self.proc.is_none() {
            self.spawn()?;
        }
        let res = (|| -> Result<String> {
            let (_, stdin, out) = self.proc.as_mut().unwrap();
            writeln!(stdin, "{}", json!({"text": text, "lang": lang}))?;
            stdin.flush()?;
            let mut line = String::new();
            anyhow::ensure!(out.read_line(&mut line)? > 0, "normaliser exited");
            let v: Value = serde_json::from_str(&line)?;
            v["out"].as_str().map(String::from).with_context(|| format!("normaliser: {}", v["error"]))
        })();
        if res.is_err() {
            if let Some((mut c, _, _)) = self.proc.take() {
                let _ = c.kill();
                let _ = c.wait(); // reap it, so a crashing normaliser does not pile up zombies
            }
        }
        res
    }
}

// ---------------------------------------------------------------------------------------------- server
struct App {
    engine: EngineHandle,
    norm: Option<Mutex<Normaliser>>,
    waiting: AtomicUsize,
    max_pending: usize,
    deadline: Duration,
}

struct ApiError {
    status: StatusCode,
    code: &'static str,
    message: String,
    param: Option<&'static str>,
}

fn bad(msg: impl Into<String>, param: Option<&'static str>) -> ApiError {
    ApiError { status: StatusCode::BAD_REQUEST, code: "invalid_request", message: msg.into(), param }
}

fn unavailable(code: &'static str, msg: impl Into<String>) -> ApiError {
    ApiError { status: StatusCode::SERVICE_UNAVAILABLE, code, message: msg.into(), param: None }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let kind = if self.status.as_u16() < 500 { "invalid_request_error" } else { "server_error" };
        let body = json!({"error": {"message": self.message, "type": kind, "code": self.code, "param": self.param}});
        let mut r = (self.status, Json(body)).into_response();
        if self.status == StatusCode::SERVICE_UNAVAILABLE {
            r.headers_mut().insert("Retry-After", HeaderValue::from_static("1"));
        }
        r
    }
}

fn opt_num(body: &Value, k: &'static str, lo: f64, hi: f64) -> Result<Option<f64>, ApiError> {
    match body.get(k) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_f64() {
            Some(x) if x.is_finite() && x >= lo && x <= hi => Ok(Some(x)),
            _ => Err(bad(format!("{k} must be a number in [{lo}, {hi}]"), Some(k))),
        },
    }
}

const OPENAI_VOICES: [&str; 13] =
    ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse", "marin", "cedar"];

/// A validated speech request (the same fields over HTTP, SSE and WebSocket).
struct Req {
    text: String,
    n_chars: usize,
    lang: String,
    fmt: Format,
    sse: bool,
    voice: Option<String>,
    max_frames: Option<usize>,
    sampling: Sampling,
    normalize: bool,
}

fn parse_req(app: &App, body: &Value) -> Result<Req, ApiError> {
    if !body.is_object() {
        return Err(bad("request body must be a JSON object", None));
    }
    let text = body.get("input").and_then(Value::as_str).map(str::trim).unwrap_or("");
    if text.is_empty() {
        return Err(bad("input must be a non-empty string", Some("input")));
    }
    let n_chars = text.chars().count();
    if n_chars > MAX_TEXT_CHARS {
        return Err(bad(format!("input is {n_chars} characters; the limit is {MAX_TEXT_CHARS}"), Some("input")));
    }
    let lang = match body.get("language").or_else(|| body.get("lang")).and_then(Value::as_str) {
        None | Some("auto") => text::detect_lang(text).to_string(),
        Some(l) if text::LANGS.contains(&l.to_lowercase().as_str()) => l.to_lowercase(),
        Some(l) => {
            return Err(bad(format!("unsupported language {l:?}; supported: {}", text::LANGS.join(", ")), Some("language")))
        }
    };
    let fmt = match body.get("response_format").and_then(Value::as_str).map(str::to_lowercase).as_deref() {
        None | Some("") => Format::Wav,
        Some(f) => Format::parse(f).ok_or_else(|| {
            bad(format!("response_format {f:?} is not available; use one of: {}", FORMATS.join(", ")), Some("response_format"))
        })?,
    };
    if let Some(s) = body.get("speed") {
        if s.as_f64() != Some(1.0) {
            return Err(bad("speed must be 1.0 (speed control is not supported)", Some("speed")));
        }
    }
    let sse = match body.get("stream_format").and_then(Value::as_str) {
        None | Some("audio") => false,
        Some("sse") => true,
        Some(_) => return Err(bad("stream_format must be 'audio' or 'sse'", Some("stream_format"))),
    };
    let voice = match body.get("voice").and_then(Value::as_str) {
        None | Some("") | Some("plain") => None,
        Some(v) => match app.engine.voices.iter().find(|x| x.id.eq_ignore_ascii_case(v)) {
            Some(x) => Some(x.id.clone()),
            None if OPENAI_VOICES.contains(&v) => None,
            None => return Err(bad(format!("unknown voice {v:?}"), Some("voice"))),
        },
    };
    let max_frames = match body.get("max_frames") {
        None | Some(Value::Null) => None,
        Some(v) => match v.as_u64() {
            Some(x) if (1..=MAX_FRAMES).contains(&x) => Some(x as usize),
            _ => return Err(bad(format!("max_frames must be an integer in 1..{MAX_FRAMES} (80 ms frames)"), Some("max_frames"))),
        },
    };
    let sampling = Sampling {
        seed: body.get("seed").and_then(Value::as_u64),
        temperature: opt_num(body, "temperature", 0.05, 2.0)?.map(|x| x as f32),
        top_k: opt_num(body, "top_k", 1.0, 2049.0)?.map(|x| x as usize),
        top_p: opt_num(body, "top_p", 1e-9, 1.0)?.map(|x| x as f32),
        repetition_penalty: opt_num(body, "repetition_penalty", 0.0, 5.0)?.map(|x| x as f32),
    };
    Ok(Req {
        text: text.to_string(),
        n_chars,
        lang,
        fmt,
        sse,
        voice,
        max_frames,
        sampling,
        normalize: body.get("normalize").and_then(Value::as_bool).unwrap_or(true),
    })
}

/// A request that has produced its first engine event (audio, normally).
struct Started {
    rid: String,
    lang: String,
    voice: String,
    fmt: Format,
    normalised: bool,
    first: Event,
    rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
    cancelled: Arc<AtomicBool>,
    log: String,
}

/// Normalise, admission-check, submit to the engine and wait (up to the queue deadline) for the first event.
async fn start_job(app: &Arc<App>, r: Req, via: &str) -> Result<Started, ApiError> {
    let sh = &app.engine.shared;
    if !sh.ready.load(Ordering::SeqCst) {
        return Err(unavailable("unavailable", "model is loading"));
    }
    let mut prepared = r.text.clone();
    let mut normalised = false;
    if r.normalize && app.norm.is_some() {
        let (t, l, app2) = (r.text.clone(), r.lang.clone(), app.clone());
        let res = tokio::task::spawn_blocking(move || app2.norm.as_ref().unwrap().lock().unwrap().call(&t, &l)).await;
        match res {
            // numbers and symbols expand when spelled out; a runaway expansion must not bypass the input limit
            Ok(Ok(out)) if out.chars().count() > 3 * MAX_TEXT_CHARS => {
                return Err(bad(format!("input expands to more than {} characters when normalised", 3 * MAX_TEXT_CHARS), Some("input")));
            }
            Ok(Ok(out)) if !out.trim().is_empty() => {
                prepared = out;
                normalised = true;
            }
            Ok(Err(e)) => eprintln!("maya: normaliser failed ({e:#}); speaking the text as written"),
            _ => {}
        }
    }
    // free seats take requests at once; beyond them at most `max_pending` may wait for one
    let limit = app.engine.max_seats + app.max_pending;
    if app.waiting.fetch_add(1, Ordering::SeqCst) >= limit {
        app.waiting.fetch_sub(1, Ordering::SeqCst);
        return Err(unavailable(
            "overloaded",
            format!("server overloaded: {limit} requests already waiting for audio; retry later"),
        ));
    }
    let rid = format!("{}-{}", &via[..1], &uuid::Uuid::new_v4().simple().to_string()[..16]);
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let cancelled = Arc::new(AtomicBool::new(false));
    let job = Job {
        id: rid.clone(),
        pieces: text::split_text(&prepared, &r.lang, engine::CHUNK_S),
        lang: r.lang.clone(),
        voice: r.voice.clone(),
        max_frames: r.max_frames,
        sampling: r.sampling,
        events: tx,
        cancelled: cancelled.clone(),
        t_submit: Instant::now(),
    };
    sh.pending.fetch_add(1, Ordering::SeqCst);
    if app.engine.tx.send(job).is_err() {
        app.waiting.fetch_sub(1, Ordering::SeqCst);
        return Err(unavailable("unavailable", "engine stopped"));
    }
    let first = tokio::time::timeout(app.deadline, rx.recv()).await;
    app.waiting.fetch_sub(1, Ordering::SeqCst);
    let first = match first {
        Err(_) => {
            cancelled.store(true, Ordering::SeqCst);
            return Err(unavailable(
                "overloaded",
                format!("no audio within the queue deadline of {} ms; retry later", app.deadline.as_millis()),
            ));
        }
        Ok(None) => return Err(unavailable("unavailable", "generation failed")),
        Ok(Some(Event::Error { code, error })) if code == "invalid_request" => return Err(bad(error, None)),
        Ok(Some(Event::Error { error, .. })) => return Err(unavailable("unavailable", error)),
        Ok(Some(e)) => e,
    };
    let voice = r.voice.unwrap_or_else(|| "plain".into());
    let log = format!("{rid} {via} lang={} voice={voice} fmt={} chars={}", r.lang, r.fmt.name(), r.n_chars);
    Ok(Started { rid, lang: r.lang, voice, fmt: r.fmt, normalised, first, rx, cancelled, log })
}

/// What a started request yields: encoded audio, then exactly one of Done / Fail.
enum Out {
    Audio(bytes::Bytes),
    Done(Value),
    Fail(String),
}

/// Engine events -> encoded output. Dropping the stream (client gone, cancel, or done) cancels the engine request.
fn pump(st: Started) -> impl futures::Stream<Item = Out> + Send {
    struct Cancel(Arc<AtomicBool>);
    impl Drop for Cancel {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }
    struct S {
        first: Option<Event>,
        rx: tokio::sync::mpsc::UnboundedReceiver<Event>,
        enc: Option<Box<dyn audio::Encoder>>,
        fmt: Format,
        queue: VecDeque<Out>,
        done: bool,
        log: String,
        _cancel: Cancel,
    }
    let s = S {
        first: Some(st.first),
        rx: st.rx,
        enc: None,
        fmt: st.fmt,
        queue: VecDeque::new(),
        done: false,
        log: st.log,
        _cancel: Cancel(st.cancelled),
    };
    futures::stream::unfold(s, |mut s| async move {
        loop {
            if let Some(o) = s.queue.pop_front() {
                return Some((o, s));
            }
            if s.done {
                return None;
            }
            let ev = match s.first.take() {
                Some(e) => Some(e),
                None => s.rx.recv().await,
            };
            if s.enc.is_none() {
                match audio::encoder(s.fmt) {
                    Ok(e) => s.enc = Some(e),
                    Err(e) => {
                        s.done = true;
                        s.queue.push_back(Out::Fail(format!("encoder: {e:#}")));
                        continue;
                    }
                }
            }
            let enc = s.enc.as_mut().unwrap();
            match ev {
                Some(Event::Audio(p)) => match enc.push(&p) {
                    Ok(b) if b.is_empty() => {}
                    Ok(b) => s.queue.push_back(Out::Audio(b.into())),
                    Err(e) => {
                        s.done = true;
                        s.queue.push_back(Out::Fail(format!("encoder: {e:#}")));
                    }
                },
                Some(Event::End(info)) => {
                    s.done = true;
                    match enc.finish() {
                        Ok(b) => {
                            if !b.is_empty() {
                                s.queue.push_back(Out::Audio(b.into()));
                            }
                            eprintln!(
                                "maya: {} ttfa_ms={} audio_s={:.2} reason={}",
                                s.log,
                                info["ttfa_s"].as_f64().map(|t| format!("{:.0}", t * 1e3)).unwrap_or_else(|| "-".into()),
                                info["audio_s"].as_f64().unwrap_or(0.0),
                                info["reason"].as_str().unwrap_or("")
                            );
                            s.queue.push_back(Out::Done(info));
                        }
                        Err(e) => s.queue.push_back(Out::Fail(format!("encoder: {e:#}"))),
                    }
                }
                Some(Event::Error { code, error }) => {
                    eprintln!("maya: {} error after audio started: {code}: {error}", s.log);
                    s.done = true;
                    s.queue.push_back(Out::Fail(error));
                }
                None => {
                    s.done = true;
                    s.queue.push_back(Out::Fail("engine dropped the request".into()));
                }
            }
        }
    })
}

fn usage(info: &Value) -> Value {
    json!({"audio_seconds": info["audio_s"], "frames": info["frames"], "ttfa_s": info["ttfa_s"], "wall_s": info["wall_s"],
           "reason": info["reason"], "pieces": info["pieces"]})
}

async fn speech(State(app): State<Arc<App>>, body: axum::body::Bytes) -> Result<Response, ApiError> {
    let body: Value = serde_json::from_slice(&body).map_err(|_| bad("request body must be a JSON object", None))?;
    let r = parse_req(&app, &body)?;
    let sse = r.sse;
    let st = start_job(&app, r, if sse { "sse" } else { "http" }).await?;
    let mut headers = HeaderMap::new();
    headers.insert("X-Sample-Rate", HeaderValue::from(SR as u32));
    headers.insert("X-Channels", HeaderValue::from_static("1"));
    headers.insert("X-Encoding", HeaderValue::from_static(if st.fmt == Format::Pcm || st.fmt == Format::Wav { "pcm_s16le" } else { st.fmt.name() }));
    headers.insert("X-Maya3-Language", HeaderValue::from_str(&st.lang).unwrap());
    headers.insert("X-Maya3-Voice", HeaderValue::from_str(&st.voice).unwrap_or(HeaderValue::from_static("plain")));
    headers.insert("X-Maya3-Normalized", HeaderValue::from_static(if st.normalised { "1" } else { "0" }));
    headers.insert("X-Request-Id", HeaderValue::from_str(&st.rid).unwrap());
    headers.insert("Cache-Control", HeaderValue::from_static("no-store"));
    if sse {
        // OpenAI's speech SSE: speech.audio.delta (base64 audio in response_format) ... speech.audio.done
        headers.insert("Content-Type", HeaderValue::from_static("text/event-stream"));
        headers.insert("X-Accel-Buffering", HeaderValue::from_static("no"));
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD;
        let body = pump(st).map(move |o| {
            let ev = match o {
                Out::Audio(b) => json!({"type": "speech.audio.delta", "audio": b64.encode(&b)}),
                Out::Done(info) => json!({"type": "speech.audio.done", "usage": usage(&info)}),
                Out::Fail(e) => json!({"type": "error", "error": {"message": e, "type": "server_error"}}),
            };
            Ok::<_, std::io::Error>(bytes::Bytes::from(format!("data: {ev}\n\n")))
        });
        return Ok((headers, Body::from_stream(body)).into_response());
    }
    headers.insert("Content-Type", HeaderValue::from_static(st.fmt.content_type()));
    // an engine error after audio started ends the body with an error, so the chunked response has no terminating
    // chunk and a truncated utterance never looks complete
    let body = pump(st)
        .take_while(|o| futures::future::ready(!matches!(o, Out::Done(_))))
        .map(|o| match o {
            Out::Audio(b) => Ok(b),
            Out::Fail(e) => Err(std::io::Error::other(e)),
            Out::Done(_) => unreachable!(),
        });
    Ok((headers, Body::from_stream(body)).into_response())
}

// ---------------------------------------------------------------------------------------------- websocket
/// `GET /v1/audio/speech/ws`: many utterances over one socket, any number in flight at once (each takes its own
/// engine seat). Client -> server text frames:
///   {"type": "speech", "id": "<client id>", <the POST /v1/audio/speech fields>}
///   {"type": "cancel", "id": "<client id>"}
/// Server -> client: text {"type":"speech.started","id","stream","request_id","format","sample_rate","language",
/// "voice","normalized"}; binary audio frames = 4-byte little-endian `stream` number + encoded bytes (in order per
/// stream); text {"type":"speech.done","id","stream","usage"} or {"type":"error","id","error":{..}}.
async fn speech_ws(State(app): State<Arc<App>>, ws: axum::extract::ws::WebSocketUpgrade) -> Response {
    ws.on_upgrade(move |sock| ws_session(app, sock))
}

async fn ws_session(app: Arc<App>, sock: axum::extract::ws::WebSocket) {
    use axum::extract::ws::Message;
    use std::collections::HashMap;
    let (mut sink, mut stream) = sock.split();
    let (out_tx, mut out_rx) = tokio::sync::mpsc::channel::<Message>(256);
    let writer = tokio::spawn(async move {
        while let Some(m) = out_rx.recv().await {
            if sink.send(m).await.is_err() {
                break;
            }
        }
    });
    let mut live: HashMap<String, tokio::task::AbortHandle> = HashMap::new();
    let mut next_stream: u32 = 0;
    let text = |v: Value| Message::Text(v.to_string().into());
    while let Some(Ok(msg)) = stream.next().await {
        let Message::Text(t) = msg else {
            if matches!(msg, Message::Close(_)) {
                break;
            }
            continue;
        };
        live.retain(|_, h| !h.is_finished());
        let v: Value = match serde_json::from_str(t.as_str()) {
            Ok(v) => v,
            Err(_) => {
                let _ = out_tx.send(text(json!({"type": "error", "error": {"message": "frame is not JSON", "code": "invalid_request"}}))).await;
                continue;
            }
        };
        let id = match v.get("id") {
            Some(Value::String(s)) => s.clone(),
            Some(Value::Number(n)) => n.to_string(),
            _ => format!("{next_stream}"),
        };
        match v.get("type").and_then(Value::as_str).unwrap_or("speech") {
            "cancel" => {
                if let Some(h) = live.remove(&id) {
                    h.abort(); // dropping the pump cancels the engine request
                    let _ = out_tx.send(text(json!({"type": "speech.cancelled", "id": id}))).await;
                }
            }
            "speech" => {
                if live.contains_key(&id) {
                    let _ = out_tx.send(text(json!({"type": "error", "id": id, "error": {"message": "id already in flight", "code": "invalid_request"}}))).await;
                    continue;
                }
                let sid = next_stream;
                next_stream = next_stream.wrapping_add(1);
                let (app2, tx, id2) = (app.clone(), out_tx.clone(), id.clone());
                let h = tokio::spawn(async move {
                    let err = |e: ApiError| {
                        json!({"type": "error", "id": id2, "stream": sid,
                               "error": {"message": e.message, "code": e.code, "param": e.param, "status": e.status.as_u16()}})
                    };
                    let st = match parse_req(&app2, &v) {
                        Ok(r) => start_job(&app2, r, "ws").await,
                        Err(e) => Err(e),
                    };
                    let st = match st {
                        Ok(s) => s,
                        Err(e) => {
                            let _ = tx.send(text(err(e))).await;
                            return;
                        }
                    };
                    let started = json!({"type": "speech.started", "id": id2, "stream": sid, "request_id": st.rid,
                        "format": st.fmt.name(), "sample_rate": SR, "language": st.lang, "voice": st.voice,
                        "normalized": st.normalised});
                    if tx.send(text(started)).await.is_err() {
                        return;
                    }
                    let mut p = std::pin::pin!(pump(st));
                    while let Some(o) = p.next().await {
                        let m = match o {
                            Out::Audio(b) => {
                                let mut f = Vec::with_capacity(4 + b.len());
                                f.extend_from_slice(&sid.to_le_bytes());
                                f.extend_from_slice(&b);
                                Message::Binary(f.into())
                            }
                            Out::Done(info) => text(json!({"type": "speech.done", "id": id2, "stream": sid, "usage": usage(&info)})),
                            Out::Fail(e) => text(json!({"type": "error", "id": id2, "stream": sid, "error": {"message": e, "code": "internal"}})),
                        };
                        if tx.send(m).await.is_err() {
                            return;
                        }
                    }
                });
                live.insert(id, h.abort_handle());
            }
            other => {
                let _ = out_tx.send(text(json!({"type": "error", "id": id, "error": {"message": format!("unknown type {other:?}"), "code": "invalid_request"}}))).await;
            }
        }
    }
    // socket closed: stop every utterance still running on it
    for (_, h) in live {
        h.abort();
    }
    drop(out_tx);
    let _ = writer.await;
}

async fn health(State(app): State<Arc<App>>) -> Response {
    let sh = &app.engine.shared;
    let ready = sh.ready.load(Ordering::SeqCst);
    let failed = sh.failed.lock().unwrap().clone();
    let mut ticks: Vec<f64> = sh.tick_ms.lock().unwrap().iter().copied().collect();
    ticks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let phase = sh.phase_ms.lock().unwrap().clone();
    let p = |q: f64| ticks.get(((ticks.len() as f64 * q) as usize).min(ticks.len().saturating_sub(1))).copied();
    let ok = ready && failed.is_none();
    let body = json!({
        "ready": ok,
        "loading": !ready && failed.is_none(),
        "failed": failed,
        "active": sh.active.load(Ordering::SeqCst),
        "pending": sh.pending.load(Ordering::SeqCst),
        "max_seats": app.engine.max_seats,
        "completed": sh.completed.load(Ordering::SeqCst),
        "tick_ms_p50": p(0.5),
        "tick_ms_p99": p(0.99),
        "last_tick_ms": phase,
        "server": {
            "impl": "maya-rs",
            "waiting": app.waiting.load(Ordering::SeqCst),
            "max_pending": app.max_pending,
            "queue_deadline_ms": app.deadline.as_millis() as u64,
            "normalise": app.norm.is_some(),
            "formats": FORMATS,
            "stream_formats": ["audio", "sse", "websocket"],
        }
    });
    (if ok { StatusCode::OK } else { StatusCode::SERVICE_UNAVAILABLE }, Json(body)).into_response()
}

async fn voices(State(app): State<Arc<App>>) -> Response {
    let mut out = vec![json!({"id": "plain", "mode": "plain", "ref_lang": null})];
    for v in &app.engine.voices {
        out.push(json!({"id": v.id, "mode": "named", "ref_lang": null, "meta": v.meta}));
    }
    Json(json!({"voices": out})).into_response()
}

async fn models() -> Response {
    Json(json!({"object": "list", "data": [{"id": "maya3", "object": "model", "owned_by": "maya"}]})).into_response()
}

fn main() -> Result<()> {
    let mut argv = std::env::args().skip(1).peekable();
    if argv.peek().map(String::as_str) == Some("selftest") {
        argv.next();
        let reference = argv.next().context("selftest <ref.json> --model-dir D --mimi F")?;
        let a = parse_args(argv)?;
        return selftest::run(&reference, &a.model_dir, &a.mimi);
    }
    let a = parse_args(argv)?;
    let norm = a.textnorm.clone().and_then(|script| {
        let mut n = Normaliser { script, proc: None };
        match n.spawn() {
            Ok(()) => Some(Mutex::new(n)),
            Err(e) => {
                eprintln!("maya: text normaliser unavailable ({e:#}); serving without normalisation");
                None
            }
        }
    });
    let engine = engine::start(engine::Paths { model_dir: a.model_dir.clone(), mimi_file: a.mimi.clone() }, a.capacity, a.max_seats)?;
    let app = Arc::new(App {
        engine,
        norm,
        waiting: AtomicUsize::new(0),
        max_pending: a.max_pending,
        deadline: Duration::from_millis(a.deadline_ms),
    });
    let router = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/voices", get(voices))
        .route("/v1/audio/speech", post(speech))
        .route("/v1/audio/speech/ws", get(speech_ws))
        .with_state(app);
    tokio::runtime::Runtime::new()?.block_on(async move {
        let listener = tokio::net::TcpListener::bind((a.host.as_str(), a.port)).await?;
        eprintln!("maya: listening on http://{}:{}", a.host, a.port);
        axum::serve(listener, router).await?;
        Ok(())
    })
}
