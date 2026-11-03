# maya-rs: Maya2 Native speech worker in Rust

A Rust + [candle](https://github.com/huggingface/candle) CUDA server for
[Maya2 Native](https://huggingface.co/maya-research/maya2-native), Maya Research's streaming Indic/English TTS
(Qwen3-1.7B backbone, depth codebook decoder, Kyutai Mimi codec). It replaces the upstream Python/torch server with
one binary: no torch, no Python ML stack, ~1.4 GB RAM, ready in seconds.

Compared with the Python server as I ran it (RTX 3060, 1 seat):

| | Python maya3 | maya-rs |
|---|---|---|
| ready after start | 89-130 s | 1.4-6.6 s |
| RAM resident (peak) | ~6.7 GB (15.8 GB) | ~1.4 GB |
| disk besides the 5 GB weights | 6.7 GB venv + caches | 31 MB binary (+ CUDA runtime) |
| first audio, idle | 41-46 ms | ~40 ms |
| concurrent real-time streams | 1 (others queue, 0.5-4 s waits) | 6 configured, 8 tested |

OpenAI-compatible speech API on `127.0.0.1:8809`:

- `POST /v1/audio/speech` `{input, voice, language|lang, response_format, stream_format, normalize, max_frames,
  seed, temperature, top_k, top_p, repetition_penalty}`, streamed as it is generated, same `X-*` headers
  - `response_format`: `wav` (default, open-ended RIFF), `pcm` (s16le 24 kHz mono), `mp3` (LAME 64 kbps),
    `opus` (Ogg/Opus 32 kbps, 20 ms packets), `flac` (lossless, one frame per 80 ms)
  - `stream_format: "sse"`: OpenAI-style `speech.audio.delta` events (base64 audio) then `speech.audio.done`
- `GET /v1/audio/speech/ws` WebSocket, any number of utterances in flight on one socket (see below)
- `GET /v1/voices` (plain + the 20 named voices), `GET /health` (503 until loaded), `GET /v1/models`

Not served: reference-audio voice cloning (`POST /v1/voices`, ECAPA).

## Several streams at once

One GPU thread runs continuous batching over `--max-seats N` seats (default 4). Each 80 ms tick
does cb0 for all seats in one head call, depth codebooks batched, backbone step batched (GQA attention per seat
without K/V copies), Mimi decode per seat. New requests are prefilled when they get a seat, at most 2 per tick
while others are speaking. Beyond the seats, `--max-pending` requests may wait (up to `--queue-deadline-ms` for
first audio); more get 503 `overloaded`.

RTX 3060, 2026-11-03, CUDA graphs on (llama-server sharing the GPU, idle):

| streams | tick ms | real-time factor | first audio |
|---|---|---|---|
| 1 | 30-32 | 0.39 | ~40 ms |
| 4 | 43 | ~0.55 | 40-150 ms |
| 6 | 48 | 0.65 | 40-250 ms |

## CUDA graphs

`src/graph.rs` captures, at start, one graph per batch size (1..max_seats) for the depth stack (codebooks 1..31,
~1900 kernels) plus the feedback embedding: candle 0.11's h2d parameter cache is enabled, the body runs once eagerly,
then once under stream capture; inputs/outputs are fixed tensors, temporaries come from the stream-ordered allocator
inside the graph, and the sampling noise is drawn outside the graph every tick. The engine uses its own CUDA stream
with event tracking off (required for capture). `MAYA_CUDA_GRAPHS=0` runs eagerly; selftest checks graph == eager
bit for bit. `MAYA_PROFILE=1` synchronises after each tick phase so `/health` `last_tick_ms` shows GPU time per
phase.

Where a single stream's 30 ms goes (GPU time): backbone ~12 ms (3.4 GB bf16 weights read per step; 9.5 ms is the
3060's bandwidth floor), depth ~16 ms (~2.7 GB fp16 weights per frame, floor ~7.6 ms), Mimi ~2 ms. Both are
bandwidth-bound, so more graphs would not help; extra seats share those weight reads, which is why batching scales.
Mimi decode is still per seat (~2 ms each) and is the next cost to batch.

VRAM: ~4.9 GB for weights + ~0.2 GB KV per seat (`--capacity 2048`).

## WebSocket protocol

Client sends text frames: `{"type":"speech","id":"u1", <POST fields>}` and `{"type":"cancel","id":"u1"}`.
Server sends `{"type":"speech.started","id","stream","request_id","format","sample_rate","language","voice",
"normalized"}`, then binary frames `[u32 LE stream][encoded audio]`, then `{"type":"speech.done","id","stream",
"usage"}`, or `{"type":"error","id","error":{message,code,status}}` / `{"type":"speech.cancelled","id"}`.
Closing the socket cancels everything on it.

## What it runs

`src/model.rs` Qwen3-1.7B backbone (bf16) + cb0 head, DepthV2 codebooks 1..31 (fp16, speaker cross-attention folded
to constants), audio input embedding, speaker anchor. `src/engine.rs` seats: prompt prefill, per-frame controlled
stop + cb0 sampling (presence penalty 0.8 / 12 frames, temp 0.9, top-k 64), depth sampling (temp 0.7, top-k 64,
on GPU), streaming Mimi decode (candle), duration cap + loop guard, 20 s text pieces with 0.15 s gaps.
`src/audio.rs` streaming encoders. `textnorm/` is the upstream `maya3.text` normaliser (standard-library Python
only), run through system `python3` as a line-JSON subprocess; without python3 the server speaks the text as written.

## Quick start

```bash
scripts/download_models.sh            # ~5 GB into models/ (needs `hf`, pip install -U huggingface_hub)
source build-env.sh                   # CUDA_ROOT (default /usr/local/cuda), GPU arch from nvidia-smi
cargo build --release
./target/release/maya-rs --model-dir models/maya2-native --mimi models/mimi/model.safetensors \
    --max-seats 6 --textnorm textnorm/serve.py
curl -s localhost:8809/v1/audio/speech -H 'Content-Type: application/json' \
    -d '{"input":"नमस्ते, आप कैसे हैं?","voice":"Ahana","response_format":"mp3"}' -o hello.mp3
```

Build needs a CUDA 12.x toolkit (nvcc) for candle's kernels, cmake and a C compiler for the bundled libopus/LAME.
Runtime needs the NVIDIA driver and the CUDA 12 cuBLAS/cuRAND libraries (system, or copied into `cuda-lib/`, which
`build-env.sh` rpath-links when present). Tested on Linux, RTX 3060 (sm_86), CUDA 12.8.

Flags: `--host`, `--port` (8809), `--max-seats` (4), `--max-pending` (4), `--queue-deadline-ms` (5000),
`--capacity` (2048 KV positions per seat), `--textnorm <serve.py>` (omit to speak text as written).

## Check against the Python stack

`scripts/make_reference.py` (run inside the upstream maya3 Python environment) dumps a greedy decode; `ref.json`\nis one. The selftest compares tokens, hidden states, logits, depth codes, feedback and Mimi PCM against it.

```bash
./target/release/maya-rs selftest ref.json --model-dir models/maya2-native --mimi models/mimi/model.safetensors
```

It also checks a batched two-seat backbone step against stepping each seat alone. It needs ~5 GB of free VRAM,
so stop the server first. `MAYA_BENCH=1` adds depth/sort micro-benchmarks.

## Run as a service

`maya-tts.service` is an example systemd user unit (edit the paths).

```bash
install -m 644 maya-tts.service ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user restart maya-tts
journalctl --user -u maya-tts -f
curl http://127.0.0.1:8809/health
```

## Not done yet

- Reference-audio voice cloning (needs Mimi's encoder + the ECAPA speaker model).
- Text normalisation still runs the upstream pure-stdlib Python module through `python3`.
- Mimi decode runs per seat; batching it is the next step for more concurrent streams.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`). Weights are not included: Maya2 Native is Apache-2.0, Kyutai Mimi is
CC-BY-4.0.
