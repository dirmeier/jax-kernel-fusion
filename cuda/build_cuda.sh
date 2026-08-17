#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

{ read -r NVIDIA_ROOT; read -r JAX_INC; } < <(
  uv run --project .. --extra cuda python -c '
import os, jax
try:
  import nvidia; print(os.path.dirname(nvidia.__file__))
except ImportError: print()
print(jax.ffi.include_dir())'
)

NVCC="$NVIDIA_ROOT/cuda_nvcc/bin/nvcc"
[ -x "$NVCC" ] || NVCC=$(command -v nvcc) || {
  echo "nvcc not found. It ships with the CUDA extra: uv sync --extra cuda"
  exit 1
}

HOSTCC="${CUDA_HOST_COMPILER:-$(command -v g++-14 g++-13 g++-12 g++-11 | head -1 || true)}"
ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ' || true)}"
ARCH="${ARCH:-90}"

echo "==> nvcc: $NVCC"
"$NVCC" --version | tail -1
echo "==> host compiler: ${HOSTCC:-nvcc default}"
echo "==> target architecture: sm_$ARCH"

# cuda_runtime.h and the cccl headers live in sibling wheels.
INCLUDES=(-I"$JAX_INC")
for d in cuda_runtime cuda_cccl cuda_nvcc; do
  [ -d "$NVIDIA_ROOT/$d/include" ] && INCLUDES+=(-I"$NVIDIA_ROOT/$d/include")
done

echo "==> building libqk_norm_rope_cuda.so"
"$NVCC" -std=c++17 -O3 --shared -Xcompiler -fPIC \
  -gencode "arch=compute_${ARCH},code=sm_${ARCH}" \
  ${HOSTCC:+-ccbin "$HOSTCC"} \
  "${INCLUDES[@]}" \
  qk_norm_rope.cu qk_norm_rope_tuned.cu -o libqk_norm_rope_cuda.so

echo "built cuda/libqk_norm_rope_cuda.so"
