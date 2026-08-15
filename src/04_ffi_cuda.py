"""CUDA kernel, called through `jax.ffi`."""

import ctypes
import pathlib

import jax

import kernel

ROOT = pathlib.Path(__file__).resolve().parent.parent
LIB_PATH = ROOT / "cuda" / "libqk_norm_rope_cuda.so"

_TARGET = "qk_norm_rope_cuda"


def load_library():
  if not LIB_PATH.exists():
    print(f"missing {LIB_PATH}")
    print("build it first:\n\n  bash cuda/build_cuda.sh\n")
    raise SystemExit(1)
  lib = ctypes.cdll.LoadLibrary(str(LIB_PATH))
  jax.ffi.register_ffi_target(
    _TARGET, jax.ffi.pycapsule(lib.QkNormRopeCuda), platform="CUDA"
  )
  return lib


def cuda_qk_norm_rope(queries, gamma, cos, sin):
  """QK-Norm+RoPE."""
  out_type = jax.ShapeDtypeStruct(queries.shape, queries.dtype)
  return jax.ffi.ffi_call(_TARGET, out_type)(queries, gamma, cos, sin)


def main():
  if jax.default_backend() != "gpu":
    print("no GPU found :)")
    return

  load_library()

  err = kernel.check(jax.jit(cuda_qk_norm_rope))
  print(f"shape = {kernel.CHECK_SHAPE}  max abs error: {err:.3e}")

  ms = kernel.bench(cuda_qk_norm_rope)
  print(f"shape = {kernel.BENCH_SHAPE}")
  print(f"time: {ms:.3f} ms   {kernel.fused_bytes() / ms * 1e-6:.1f} GB/s")


if __name__ == "__main__":
  main()
