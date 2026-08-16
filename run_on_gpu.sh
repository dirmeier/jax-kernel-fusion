#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")"

OUT=gpu_results.txt
exec > >(tee "$OUT") 2>&1

echo "### environment "###
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
uv run nvcc --version | tail -2
uv sync --extra cuda

echo
echo "### jax devices "###
uv run python -c "import jax; print(jax.devices()); print(jax.default_backend())"

echo
echo "### build cuda "###
bash cuda/build_cuda.sh

for script in 01_baseline 02_pallas_smem 03_pallas_tiled 04_ffi_cuda dump_ir; do
  echo
  echo "### $script"
  if ! uv run python "src/$script.py"; then
    echo "### $script FAILED"
  fi
done
