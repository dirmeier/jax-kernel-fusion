"""XLA-based kernel fusion."""

import jax

import kernel


def main():
  args = kernel.make_inputs(kernel.CHECK_SHAPE)
  print(f"shape = {kernel.CHECK_SHAPE}")

  if jax.default_backend() != "gpu":
    print("no GPU found :)")

  print(jax.make_jaxpr(kernel.qk_norm_rope)(*args))

  ms = kernel.bench(kernel.qk_norm_rope)
  print(f"\nshape = {kernel.BENCH_SHAPE}")
  print(f"time: {ms:.3f} ms   {kernel.fused_bytes() / ms * 1e-6:.1f} GB/s")


if __name__ == "__main__":
  main()
