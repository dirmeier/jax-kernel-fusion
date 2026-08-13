"""Pallas kernel: one head vector per program, rotation through SMEM."""

import jax
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp

import kernel


def _kernel(queries_ref, gamma_ref, cos_ref, sin_ref, o_ref, smem):
  queries = queries_ref[0]
  d = queries.shape[0]
  half = d // 2

  total = jnp.sum(queries * queries)
  qn = queries * jax.lax.rsqrt(total / d + 1e-6) * gamma_ref[0]

  # needed cause a Pallas thread is 128 CUDA lanes.
  smem[0:d] = qn
  smem[d : 2 * d] = qn
  rot = smem[half : half + d]

  o_ref[0] = qn * cos_ref[0] + rot * sin_ref[0]


def pallas_qk_norm_rope(queries, gamma, cos, sin, interpret=None):
  """QK-Norm+RoPE, one program per head vector."""
  B, S, H, D = queries.shape
  if interpret is None:
    interpret = jax.default_backend() != "gpu"
  n = B * S * H

  cos_full = jnp.concatenate([cos, cos], axis=-1)
  sin_full = jnp.concatenate([-sin, sin], axis=-1)

  out = pl.pallas_call(
    _kernel,
    out_shape=jax.ShapeDtypeStruct((n, D), queries.dtype),
    grid=(n,),
    in_specs=[
      pl.BlockSpec((1, D), lambda i: (i, 0)),
      pl.BlockSpec((1, D), lambda i: (0, 0)),
      pl.BlockSpec((1, D), lambda i: ((i // H) % S, 0)),
      pl.BlockSpec((1, D), lambda i: ((i // H) % S, 0)),
    ],
    out_specs=pl.BlockSpec((1, D), lambda i: (i, 0)),
    scratch_shapes=[plgpu.SMEM((2 * D,), queries.dtype)],
    interpret=interpret,
  )(queries.reshape(n, D), gamma.reshape(1, D), cos_full, sin_full)
  return out.reshape(B, S, H, D)


def main():
  err = kernel.check(pallas_qk_norm_rope)
  print(f"shape = {kernel.CHECK_SHAPE}  max abs error: {err:.3e}")

  if jax.default_backend() != "gpu":
    print("no GPU found :)")
    return

  ms = kernel.bench(pallas_qk_norm_rope)
  print(f"shape = {kernel.BENCH_SHAPE}")
  print(f"time: {ms:.3f} ms   {kernel.fused_bytes() / ms * 1e-6:.1f} GB/s")


if __name__ == "__main__":
  main()
