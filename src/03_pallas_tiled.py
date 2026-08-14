"""Pallas kernel: a [64, D] tile per program, rotation in registers."""

import functools

import jax
import jax.experimental.mosaic.gpu as mgpu
import jax.experimental.pallas as pl
import jax.experimental.pallas.mosaic_gpu as plgpu
import jax.numpy as jnp

import kernel

TILE_ROWS = 64

_LANE = plgpu.CompilerParams(lowering_semantics=mgpu.LoweringSemantics.Lane)
_TRANSFORMS = (plgpu.TilingTransform((8, 32)), plgpu.SwizzleTransform(128))


def _kernel(queries_ref, gamma_ref, cos_ref, sin_ref, o_ref, *, cast):
  queries = queries_ref[...]
  d = queries.shape[1]
  half = d // 2

  if cast:
    queries = plgpu.layout_cast(queries, plgpu.Layout.WGMMA)

  scale = jax.lax.rsqrt(jnp.sum(queries * queries, axis=1) / d + 1e-6)
  scale = jax.lax.broadcast_in_dim(scale, (TILE_ROWS, d), (0,))
  qn = queries * scale * gamma_ref[...]

  lo, hi = qn[:, :half], qn[:, half:]
  c, s = cos_ref[...], sin_ref[...]
  o_ref[:, :half] = lo * c - hi * s
  o_ref[:, half:] = hi * c + lo * s


def tiled_qk_norm_rope(queries, gamma, cos, sin, interpret=None):
  """QK-Norm+RoPE over a `[64, D]` tile per program."""
  B, S, H, D = queries.shape
  if S % TILE_ROWS:
    raise ValueError(f"sequence length {S} is not a multiple of {TILE_ROWS}")
  if interpret is None:
    interpret = jax.default_backend() != "gpu"

  nblk = S // TILE_ROWS
  if interpret:
    spec = pl.BlockSpec
    params = None
  else:
    spec = functools.partial(plgpu.BlockSpec, transforms=_TRANSFORMS)
    params = _LANE

  out = pl.pallas_call(
    functools.partial(_kernel, cast=not interpret),
    out_shape=jax.ShapeDtypeStruct((B * S, H * D), queries.dtype),
    grid=(B, H, nblk),
    in_specs=[
      spec((TILE_ROWS, D), lambda b, h, s: (b * nblk + s, h)),
      spec((TILE_ROWS, D), lambda b, h, s: (0, 0)),
      spec((TILE_ROWS, D // 2), lambda b, h, s: (s, 0)),
      spec((TILE_ROWS, D // 2), lambda b, h, s: (s, 0)),
    ],
    out_specs=spec((TILE_ROWS, D), lambda b, h, s: (b * nblk + s, h)),
    interpret=interpret,
    compiler_params=params,
  )(
    queries.reshape(B * S, H * D),
    jnp.broadcast_to(gamma, (TILE_ROWS, D)),
    cos,
    sin,
  )
  return out.reshape(B, S, H, D)


def main():
  B, _, H, D = kernel.CHECK_SHAPE
  shape = (B, TILE_ROWS, H, D)
  err = kernel.check(tiled_qk_norm_rope, shape=shape)
  print(f"shape = {shape}  max abs error: {err:.3e}")

  if jax.default_backend() != "gpu":
    print("no GPU found :)")
    return

  ms = kernel.bench(tiled_qk_norm_rope)
  print(f"shape = {kernel.BENCH_SHAPE}")
  print(f"time: {ms:.3f} ms   {kernel.fused_bytes() / ms * 1e-6:.1f} GB/s")


if __name__ == "__main__":
  main()
