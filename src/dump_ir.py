"""Writes StableHLO and optimised HLO for every implementation."""

import importlib.util
import pathlib
import re

import jax

import kernel

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "ir"
_MAX_BLOB = 200


def truncate_blobs(text):
  def repl(match):
    body = match.group(1)
    if len(body) <= _MAX_BLOB:
      return match.group(0)
    return f'"<truncated {len(body)} bytes>"'

  return re.sub(r'"((?:[^"\\]|\\.)*)"', repl, text)


def load(filename, attribute):
  path = pathlib.Path(__file__).resolve().parent / filename
  spec = importlib.util.spec_from_file_location(path.stem, str(path))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return getattr(module, attribute)


def entry_block(hlo):
  lines = []
  inside = False
  for line in hlo.splitlines():
    if line.startswith("ENTRY"):
      inside = True
    if inside:
      lines.append(line.split(", metadata=")[0])
    if inside and line.strip() == "}":
      break
  return lines


def dump(name, fn, args):
  lowered = jax.jit(fn).lower(*args)
  stablehlo = truncate_blobs(lowered.as_text())
  hlo = truncate_blobs(lowered.compile().as_text())

  (OUT_DIR / f"{name}.stablehlo").write_text(stablehlo)
  (OUT_DIR / f"{name}.hlo").write_text(hlo)

  print(f"### {name}: optimised HLO, ENTRY block")
  for line in entry_block(hlo):
    print(line)
  print(
    f"fusion instructions: {hlo.count('fusion(')}   "
    f"custom_call: {hlo.count('custom-call(')}   "
    f"total lines: {len(hlo.splitlines())}"
  )


def main():
  if jax.default_backend() != "gpu":
    print("no GPU found :)")
    return

  OUT_DIR.mkdir(exist_ok=True)
  args = kernel.make_inputs(kernel.BENCH_SHAPE)

  cuda_fn = load("04_ffi_cuda.py", "cuda_qk_norm_rope")
  load("04_ffi_cuda.py", "load_library")()

  for name, fn in (
    ("xla", kernel.qk_norm_rope),
    ("pallas_smem", load("02_pallas_smem.py", "pallas_qk_norm_rope")),
    ("pallas_tiled", load("03_pallas_tiled.py", "tiled_qk_norm_rope")),
    ("cuda", cuda_fn),
  ):
    try:
      dump(name, fn, args)
    except Exception as exc:  # noqa: BLE001 - report, do not hide
      print(f"\n{name}: FAILED {type(exc).__name__}: {str(exc)[:300]}")


if __name__ == "__main__":
  main()
