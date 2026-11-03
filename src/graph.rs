//! CUDA graphs for the per-frame depth stack + feedback embedding (31 codebook levels x 4 layers: ~1900 small
//! kernels per tick that are launch-bound when issued one by one). One graph per batch size, captured at start.
//!
//! Capture protocol (candle 0.11): with the h2d parameter cache enabled, run the body once eagerly (loads kernel
//! modules, cuBLAS workspaces, and caches every layout/stride upload), then run it again under stream capture.
//! Inputs and outputs live in tensors allocated before capture, so replays see the same addresses; temporaries are
//! allocated and freed inside the graph (stream-ordered allocator). Sampling noise is drawn outside the graph each
//! tick (curand's offset is host state, a captured generator would replay the same noise).

use crate::model::{Maya, HIDDEN, N_Q};
use anyhow::{Context, Result};
use candle_core::cuda::cudarc::driver::{sys, CudaGraph};
use candle_core::{DType, Device, Tensor};

pub struct FrameGraph {
    h: Tensor,     // [B, 2048] bf16
    cb0: Tensor,   // [B] u32
    u: Tensor,     // [B, 31, 2048] f32
    codes: Tensor, // [B, 32] u32
    fb: Tensor,    // [B, 1, 2048] bf16
    graph: CudaGraph,
}

/// What the graph runs: depth codes for every row, then the backbone input embedding of those codes.
fn body(maya: &Maya, h: &Tensor, cb0: &Tensor, u: &Tensor, codes_out: &Tensor, fb_out: &Tensor) -> Result<()> {
    let codes = maya.depth.frame_with(h, cb0, u)?;
    let fb = maya.feedback(&codes)?;
    codes_out.slice_set(&codes, 0, 0)?;
    fb_out.slice_set(&fb.contiguous()?, 0, 0)?;
    Ok(())
}

impl FrameGraph {
    pub fn capture(maya: &Maya, b: usize) -> Result<Self> {
        let dev = &maya.dev;
        let cuda = dev.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let h = Tensor::zeros((b, HIDDEN), DType::BF16, dev)?;
        let cb0 = Tensor::zeros(b, DType::U32, dev)?;
        let u = maya.depth.noise(b, dev)?;
        let codes = Tensor::zeros((b, N_Q), DType::U32, dev)?;
        let fb = Tensor::zeros((b, 1, HIDDEN), DType::BF16, dev)?;
        let _cache = cuda.enable_cuda_graph_htod_cache();
        body(maya, &h, &cb0, &u, &codes, &fb)?; // warm-up: fills the h2d cache, loads kernels
        stream.synchronize()?;
        stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
        let res = body(maya, &h, &cb0, &u, &codes, &fb);
        let graph = stream.end_capture(sys::CUgraphInstantiate_flags(0));
        res.context("depth graph capture")?;
        let graph = graph?.context("empty depth graph")?;
        graph.upload()?;
        Ok(Self { h, cb0, u, codes, fb, graph })
    }

    /// Run one frame for `h` [B, 2048] and `cb0` [B]. Returns (codes [B, 32] u32, feedback [B, 1, 2048] bf16), copies
    /// that stay valid after the next run.
    pub fn run(&self, h: &Tensor, cb0: &Tensor, noise: &Tensor) -> Result<(Tensor, Tensor)> {
        self.h.slice_set(h, 0, 0)?;
        self.cb0.slice_set(cb0, 0, 0)?;
        self.u.slice_set(noise, 0, 0)?;
        self.graph.launch()?;
        Ok((self.codes.copy()?, self.fb.copy()?))
    }
}

/// Graphs for batch sizes 1..=max_seats, or None on CPU / when disabled with MAYA_CUDA_GRAPHS=0.
pub fn capture_all(maya: &Maya, max_seats: usize) -> Result<Option<Vec<FrameGraph>>> {
    if !matches!(maya.dev, Device::Cuda(_)) || std::env::var("MAYA_CUDA_GRAPHS").as_deref() == Ok("0") {
        return Ok(None);
    }
    let t0 = std::time::Instant::now();
    let gs = (1..=max_seats).map(|b| FrameGraph::capture(maya, b)).collect::<Result<Vec<_>>>()?;
    eprintln!("maya: captured {} frame graphs in {:.1}s", gs.len(), t0.elapsed().as_secs_f64());
    Ok(Some(gs))
}
