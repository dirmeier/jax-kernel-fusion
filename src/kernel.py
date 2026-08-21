"""QK-Norm+RoPE referemce."""

import time

import jax
import jax.numpy as jnp
import numpy as np

BENCH_SHAPE = (128, 1024, 16, 128)
CHECK_SHAPE = (2, 32, 4, 128)


def rope_tables(seq_len, head_dim, dtype=jnp.float32):
  """Builds rotary cosine and sine tables."""
  inv_freq = 10_000 ** (-jnp.arange(0, head_dim, 2, dtype=dtype) / head_dim)
  angles = jnp.arange(seq_len, dtype=dtype)[:, None] * inv_freq[None, :]
  return jnp.cos(angles), jnp.sin(angles)


def make_inputs(shape, seed=0):
  _, seq_len, _, head_dim = shape
  queries = jax.random.normal(jax.random.key(seed), shape, jnp.float32)
  gamma = jnp.linspace(0.5, 1.5, head_dim, dtype=jnp.float32)
  cos, sin = rope_tables(seq_len, head_dim)
  return queries, gamma, cos, sin


def qk_norm_rope(queries, gamma, cos, sin):
  """QK-Norm+RoPE."""
  mean_square = jnp.mean(jnp.square(queries), axis=-1, keepdims=True)
  qn = queries * jax.lax.rsqrt(mean_square + 1e-6) * gamma

  half = queries.shape[-1] // 2
  lo, hi = qn[..., :half], qn[..., half:]
  c = cos[None, :, None, :]
  s = sin[None, :, None, :]
  return jnp.concatenate([lo * c - hi * s, hi * c + lo * s], axis=-1)


def check(fn, shape=CHECK_SHAPE, atol=1e-5):
  args = make_inputs(shape)
  got = np.asarray(jax.block_until_ready(fn(*args)))
  want = np.asarray(qk_norm_rope(*args))
  err = float(np.abs(got - want).max())
  assert np.allclose(got, want, atol=atol), f"max abs error {err:.3e} > {atol}"
  return err


def bench(fn, shape=BENCH_SHAPE, warmup=3, repeats=20):
  args = make_inputs(shape)
  compiled = jax.jit(fn)
  for _ in range(warmup):
    jax.block_until_ready(compiled(*args))
  start = time.perf_counter()
  for _ in range(repeats):
    out = compiled(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - start) / repeats * 1e3


def fused_bytes(shape=BENCH_SHAPE):
  return 8 * np.prod(shape)
