# jax-kernel-fusion

[![ci](https://github.com/dirmeier/jax-kernel-fusion/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/jax-kernel-fusion/actions/workflows/ci.yaml)

> Evaluating JAX kernel fusion using XLA, Pallas and CUDA FFI

`jax-kernel-fusion` implements fused QK-Norm+RoPE operations five ways:
- vanilla XLA (i.e., just implement the ops in Python and let the XLA compiler do the rest),
- JAX Pallas using Mosaic GPU, with one kernel per head vector,
- JAX Pallas using Mosaic GPU, with one kernel per `[64, D]` tile,
- manual CUDA using `jax.ffi`,
- manual CUDA using `jax.ffi`, with `gamma` and the rotary tables in SMEM.

Please find a blog post which goes into detail and explains how all this relates to HLO and kernel fusions [here](https://simon-dirmeier.com/blog/jax-kernel-fusion).

## Installation

```bash
uv sync --extra cuda
bash cuda/build_cuda.sh
```

## Usage

The scripts require a Hopper GPU (i.e., `sm_90`).

```bash
uv run python src/01_baseline.py       # XLA: jaxpr, optimised HLO, timing
uv run python src/02_pallas_smem.py    # Pallas: one head vector per program
uv run python src/03_pallas_tiled.py   # Pallas: [64, D] tile per program
uv run python src/04_ffi_cuda.py       # CUDA
uv run python src/05_ffi_cuda_tuned.py # CUDA: gamma and RoPE tables in SMEM

uv run python src/dump_ir.py           # write StableHLO and HLO IRs

# Alternatively
bash run_on_gpu.sh
```

## Development

The project uses uv for everything:

```bash
uv sync --all-extras
uv run pre-commit install -t pre-commit -t commit-msg
uv run pre-commit run --all-files
```
