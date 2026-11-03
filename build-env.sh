# source this before `cargo build`. Needs a CUDA 12.x toolkit (nvcc) for candle's kernels and cmake for the bundled
# libopus / LAME. Override anything below by exporting it first or in build-env.local.sh (not committed).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$HERE/build-env.local.sh" ] && source "$HERE/build-env.local.sh"
export CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda}
export CUDA_PATH=$CUDA_ROOT
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_ROOT
export PATH=$CUDA_ROOT/bin:$PATH
export CPATH=$CUDA_ROOT/targets/x86_64-linux/include${CPATH:+:$CPATH}
export LIBRARY_PATH=$CUDA_ROOT/lib64:$CUDA_ROOT/lib${LIBRARY_PATH:+:$LIBRARY_PATH}
# GPU architecture for the kernels, e.g. 86 = RTX 30xx, 89 = RTX 40xx, 80 = A100
export CUDA_COMPUTE_CAP=${CUDA_COMPUTE_CAP:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}
# bundled libopus (audiopus_sys) still declares cmake_minimum_required < 3.5
export CMAKE_POLICY_VERSION_MINIMUM=3.5
# optional: ship cuBLAS/cuRAND next to the binary in ./cuda-lib and rpath-link against it
if [ -d "$HERE/cuda-lib" ]; then
  export LIBRARY_PATH=$HERE/cuda-lib:$LIBRARY_PATH
  export RUSTFLAGS="-C link-arg=-Wl,-rpath,$HERE/cuda-lib"
fi
# candle always links -lnvrtc; some toolkits (conda) ship only libnvrtc.so.12. Link-time stub, --as-needed drops it.
if ! ls "$CUDA_ROOT"/lib*/libnvrtc.so >/dev/null 2>&1; then
  NV=$(ls "$CUDA_ROOT"/lib*/libnvrtc.so.12 2>/dev/null | head -1)
  if [ -n "$NV" ]; then
    mkdir -p "$HERE/target/linkstubs" && ln -sf "$NV" "$HERE/target/linkstubs/libnvrtc.so"
    export LIBRARY_PATH=$HERE/target/linkstubs:$LIBRARY_PATH
  fi
fi
