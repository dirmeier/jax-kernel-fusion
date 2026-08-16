# jax-kernel-fusion

[![ci](https://github.com/dirmeier/jax-kernel-fusion/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/jax-kernel-fusion/actions/workflows/ci.yaml)

> Evaluating JAX kernel fusion using XLA, Pallas and CUDA FFI

`jax-kernel-fusion` implements fused QK-Norm+ROPE operations four ways:
- vanilla XLA (i.e., just implement the ops in Python and let the XLA compiler do the rest),
- JAX Pallas using Mosaic GPU, with one kernel per head vector,
- JAX Pallas using Mosaic GPU, with one kernel per `[64, D]` tile,
- manual CUDA using `jax.ffi`.

The repository implements the following: Let $q \in \mathbb{R}^{B \times S \times H \times D}$ be the query vector of a self-attention mechanism. If $x = q[b, s, h, :]$ is a single attention head vector, QK-Norm+ROPE first normalises over the head dimension

$$\mu = \frac{1}{D}\sum_d x_d^2, \qquad \hat{x}_d = \frac{x_d}{\sqrt{\mu+\varepsilon}}\,g_d,$$

and then rotates each channel $i$ against its partner $i+m$, wgere $m = D / 2$, by an angle that
depends on the sequence position $s$,

$$\begin{pmatrix} y_i \\ y_{i+m}\end{pmatrix} = \begin{pmatrix} \cos s\theta_i & -\sin s\theta_i \\ \sin s\theta_i & \cos s\theta_i\end{pmatrix}\begin{pmatrix} \hat{x}_i \\ \hat{x}_{i+m}\end{pmatrix},$$

where $\theta_i = \Theta^{-2i/D}$ and $\Theta$ is some constant.

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
uv run python src/04_ffi_cuda.py       # CUDA, and the comparison table
```

## Results

The different implementations have been measured on a GH200 (which uses a H100 Hopper GPU) using the query dimensionality $(4, 1024, 16, 128)$:

| Mode      | ms    | GB/s  | vs XLA |
| ------------------- | ----- | ----- | ------ |
| XLA, compiler-fused | 0.157 | 427.2 | 1.00x  |
| Pallas, SMEM        | 0.189 | 354.7 | 0.83x  |
| Pallas, tiled       | 0.165 | 405.7 | 0.95x  |
| CUDA FFI, manual  | 0.129 | 518.4 | 1.21x  |

So CUDA > XLA > Pallas, tiled > Pallas, SMEM. Why is that (I think)?

**Pallas, SMEM**. This Pallas kernel utilizes Shared Memory (SMEM) as a workaround to calculate rotary embeddings without violating hardware memory constraints. This is required because Mosaic GPU's `WG_STRIDED` layout partitions an array equally among the 128 CUDA lanes that form a Pallas thread, so a value needs a multiple of 128 elements to sit in registers. At $D = 128$, we have two 64-wide rotation halves, which fall below that, so the kernel keeps the head vector whole and reaches the partner channel ($i + m$) through SMEM instead.

**Pallas, tiled**. The tiled Pallas kernel takes 64 sequence positions of one head instead, which makes the halves large enough to slice in registers. It needs neither shared memory nor rebuilt tables, and lands within 5% of XLA.

**CUDA**. CUDA wins by assigning one warp (32 threads) to each head vector, rather than forcing it across 128 lanes. This ensures all 128 channels live safely in the registers of a single warp, requiring no workarounds.

### HLO

Let's look into the different HLOs, i.e., the specific intermediate representations (IRs) of the XLA compiler. But first, let's do a small intro to what happens under the hood of XLA when seeing a JAX function.

#### From `jaxpr` to HLO to PTX/SASS

In plain JAX, the QK-Norm+ROPE operation looks like this:

```python
def qk_norm_rope(query, gamma, cos, sin, EPS=1e-6):
  mean_square = jnp.mean(jnp.square(query), axis=-1, keepdims=True)
  qn = query * jax.lax.rsqrt(mean_square + EPS) * gamma

  half = query.shape[-1] // 2
  lo, hi = qn[..., :half], qn[..., half:]
  c = cos[None, :, None, :]
  s = sin[None, :, None, :]
  return jnp.concatenate([lo * c - hi * s, hi * c + lo * s], axis=-1)
```

The traced JAX expression (`jaxpr`) then has the following form.

```
{ lambda ; a:f32[2,32,4,128] b:f32[128] c:f32[32,64] d:f32[32,64]. let
    e:f32[2,32,4,128] = square a
    f:f32[2,32,4] = reduce_sum[axes=(3,) out_sharding=None] e
    g:f32[2,32,4,1] = broadcast_in_dim[broadcast_dimensions=(0, 1, 2)] f
    h:f32[2,32,4,1] = div g 128.0:f32[]
    i:f32[2,32,4,1] = add h 9.999999974752427e-07:f32[]
    j:f32[2,32,4,1] = rsqrt i
    k:f32[2,32,4,128] = mul a j
    l:f32[1,1,1,128] = broadcast_in_dim[broadcast_dimensions=(3,)] b
    m:f32[2,32,4,128] = mul k l
    n:f32[2,32,4,64] = slice[
      limit_indices=(2, 32, 4, 64)
      start_indices=(0, 0, 0, 0)
      strides=None
    ] m
    o:f32[2,32,4,64] = slice[
      limit_indices=(2, 32, 4, 128)
      start_indices=(0, 0, 0, 64)
      strides=None
    ] m
    p:f32[1,32,1,64] = broadcast_in_dim[broadcast_dimensions=(1, 3)] c
    q:f32[1,32,1,64] = broadcast_in_dim[broadcast_dimensions=(1, 3)] d
    r:f32[2,32,4,64] = mul n p
    s:f32[2,32,4,64] = mul o q
    t:f32[2,32,4,64] = sub r s
    u:f32[2,32,4,64] = mul o p
    v:f32[2,32,4,64] = mul n q
    w:f32[2,32,4,64] = add u v
    x:f32[2,32,4,128] = concatenate[dimension=3] t w
  in (x,) }
```

The `jaxpr` shows that JAX records 20 primitive operations. Note, that we could execute the `jaxpr` graph using:

```python
cj = jax.make_jaxpr(kernel.qk_norm_rope)(*args)
out = jax.core.eval_jaxpr(cj.jaxpr, cj.literals, *args)[0]
```

From the `jaxpr`, every implementation takes the same "path" to the GPU:

```
Python -> jaxpr -> StableHLO -> HLO -> PTX/SASS
```

However, the four implementations take 3 different "routes" in order to produce machine code (shown below with the help of Gemini):

```
  route 1                route 2                  route 3
  XLA, compiler-fused    Pallas (smem, tiled)     CUDA via FFI

  jnp ops in Python      kernel fn in Python      qk_norm_rope.cu in C/CUDA
        |                        |                       |
      jaxpr                pallas_call             nvcc, at BUILD time
        |                        |                       |
    StableHLO           StableHLO custom_call   StableHLO custom_call
        |                 @mosaic_gpu_v2         @qk_norm_rope_cuda
        |                        :                       :
        |                        :                       :
        |                        :                       :
        |                        :                       :
    HLO passes                  HLO                     HLO
    (fusion decided)      custom-call, target=   custom-call, target=
        |                 mosaic_gpu_v2          qk_norm_rope_cuda
        |                        |                       |
        |                        |                       |
        |                        |                       |
    Triton IR              Mosaic GPU                    |
    (on GH200)             MLIR -> NVVM                  |
        |                        |                       |
        |                        |                       |
     LLVM IR                     |                       |
        +----------------------- + ----------------------+
                                 |
                                PTX          virtual ISA
                                 |
                               ptxas         compiles/optimizes PTX for a specific architecture
                                 |
                                SASS         real machine code for sm_90
```

The HLO (at least with some caveats) shows how many kernels each implementation launches per call. Highlighted lines below are the launches. However, a `fusion` becomes a single kernel while a `custom-call` is opaque to XLA and it may launch multipe kernels, too.

> [!NOTE]
> Here the mapping actually does hold, for both calls. The CUDA handler in
> `cuda/qk_norm_rope.cu` contains a single `<<<>>>`. And Mosaic spawns one
> `gpu.launch` per `pallas_call`.

### HLOs for each implementation

When we `jit` the `qk_norm_rope` function on an GH200, XLA fuses these into a single kernel operation using OpenAI's Triton with 8 warps (8 x 32 threads) and an output
tensor shape of `[4, 2, 16, 64]`. The HLO:

```diff
  ENTRY %main.2 (q, g, cos, sin) -> f32[4,1024,16,128] {
+   ROOT %fusion.9 = f32[4,1024,16,128] fusion(%sin.1, %cos.1, %g.1, %q.1),
+       kind=kCustom, calls=%fused_computation.7,
        backend_config={"fusion_backend_config":{
          "kind":"__triton",
          "block_level_fusion_config":{
            "num_warps":"8","output_tiles":[{"sizes":["4","2","16","64"]}]}}}
  }
```

Interestingly, on my Mac M1, XLA produces 3 fused kernels:

```diff
  ENTRY %main.2 (q.1: f32[4,1024,16,128], g.1: f32[128], cos.1: f32[1024,64], sin.1: f32[1024,64]) -> f32[4,1024,16,128] {
    %q.1 = f32[4,1024,16,128]{3,2,1,0} parameter(0), metadata={op_name="q"}
    %g.1 = f32[128]{0} parameter(1), metadata={op_name="g"}
    %cos.1 = f32[1024,64]{1,0} parameter(2), metadata={op_name="cos"}
    %sin.1 = f32[1024,64]{1,0} parameter(3), metadata={op_name="sin"}
+   %ynn_fusion = f32[4,1024,16,128]{3,2,1,0} fusion(%q.1, %g.1), kind=kCustom, calls=%fused_computation, metadata={op_name="jit(qk_norm_rope)/reduce_sum" stack_frame_id=3}, backend_config={"fusion_config":{"kind":"__ynn_fusion"},"outer_dimension_partitions":[]}
+   %multiply_subtract_fusion = f32[4,1024,16,64]{3,2,1,0} fusion(%sin.1, %ynn_fusion, %cos.1), kind=kLoop, calls=%fused_computation.2, metadata={op_name="jit(qk_norm_rope)/sub" stack_frame_id=12}, backend_config={"outer_dimension_partitions":["3"]}
+   %multiply_add_fusion = f32[4,1024,16,64]{3,2,1,0} fusion(%sin.1, %ynn_fusion, %cos.1), kind=kLoop, calls=%fused_computation.1, metadata={op_name="jit(qk_norm_rope)/add" stack_frame_id=14}, backend_config={"outer_dimension_partitions":["3"]}
    ROOT %concatenate.1 = f32[4,1024,16,128]{3,2,1,0} concatenate(%multiply_subtract_fusion, %multiply_add_fusion), dimensions={3}, metadata={op_name="jit(qk_norm_rope)/concatenate" stack_frame_id=16}, backend_config={"outer_dimension_partitions":["3"]}
  }
```

The baseline is therefore highly optimized, which is why the manual CUDA code in the table above is _only_ 1.21x faster instead of being _significantly_ faster. Let's have a look at the CUDA HLO:

```diff
  ENTRY %main.1 (q.1: f32[4,1024,16,128], g.1: f32[128], cos.1: f32[1024,64], sin.1: f32[1024,64]) -> f32[4,1024,16,128] {
    %sin.1 = f32[1024,64]{1,0} parameter(3)
    %cos.1 = f32[1024,64]{1,0} parameter(2)
    %g.1 = f32[128]{0} parameter(1)
    %q.1 = f32[4,1024,16,128]{3,2,1,0} parameter(0)
+   ROOT %ffi_call.1 = f32[4,1024,16,128]{3,2,1,0} custom-call(%q.1, %g.1, %cos.1, %sin.1), custom_call_target="qk_norm_rope_cuda", operand_layout_constraints={f32[4,1024,16,128]{3,2,1,0}, f32[128]{0}, f32[1024,64]{1,0}, f32[1024,64]{1,0}}, api_version=API_VERSION_TYPED_FFI, metadata={op_name="jit(cuda_qk_norm_rope)/ffi_call" scheduling_name="ffi_call.1" stack_frame_id=4}, backend_config={}
}
```
As expected (since it was defined this was), a single primitive.


Pallas, SMEM, HLO:
```diff
  ENTRY %main.1 (q.1: f32[4,1024,16,128], g.1: f32[128], cos.1: f32[1024,64], sin.1: f32[1024,64]) -> f32[4,1024,16,128] {
    %sin.1 = f32[1024,64]{1,0} parameter(3)
    %cos.1 = f32[1024,64]{1,0} parameter(2)
    %g.1 = f32[128]{0} parameter(1)
    %bitcast.1.0 = f32[1,128]{1,0} bitcast(%g.1), metadata={op_name="g" scheduling_name="bitcast.1.0"}
  %q.1 = f32[4,1024,16,128]{3,2,1,0} parameter(0), metadata={op_name="q" scheduling_name="q.1"}
  %bitcast.3 = f32[65536,128]{1,0} bitcast(%q.1), metadata={op_name="q" scheduling_name="bitcast.3"}
+ %wrapped_concatenate = f32[1024,128]{1,0} fusion(%cos.1), kind=kLoop, calls=%wrapped_concatenate_computation, metadata={op_name="jit(pallas_qk_norm_rope)/concatenate" scheduling_name="wrapped_concatenate" stack_frame_id=6}, backend_config={"device_type":"DEVICE_TYPE_INVALID","force_earliest_schedule":false,"native_emitter_backend_config":{"type":"NATIVE_EMITTER_TYPE_INVALID","unroll_factor":0},"operation_queue_id":"0","reification_cost":[]}
+ %input_concatenate_fusion = f32[1024,128]{1,0} fusion(%sin.1), kind=kInput, calls=%fused_concatenate, metadata={op_name="jit(pallas_qk_norm_rope)/concatenate" scheduling_name="input_concatenate_fusion" stack_frame_id=4}, backend_config={"device_type":"DEVICE_TYPE_INVALID","force_earliest_schedule":false,"native_emitter_backend_config":{"type":"NATIVE_EMITTER_TYPE_INVALID","unroll_factor":0},"operation_queue_id":"0","reification_cost":[]}
+ %pallas_call.1 = f32[65536,128]{1,0} custom-call(%bitcast.3, %bitcast.1.0, %wrapped_concatenate, %input_concatenate_fusion), custom_call_target="mosaic_gpu_v2", operand_layout_constraints={f32[65536,128]{1,0}, f32[1,128]{1,0}, f32[1024,128]{1,0}, f32[1024,128]{1,0}}, api_version=API_VERSION_TYPED_FFI, metadata={op_name="jit(pallas_qk_norm_rope)/pallas_call" scheduling_name="pallas_call.1" stack_frame_id=7}, backend_config={kernel_hash = "O\85I\9D\F3\9DF\E5\C8\08Y\9BC3\02\F7Y\D5\1F\DA\06\A3\EE\ED\1A2\B6\CA\85\CC\13\B1", module = "<truncated 37057 bytes>", use_custom_barrier = false, uses_xla_collective_metadata = false}
  ROOT %bitcast.2.0 = f32[4,1024,16,128]{3,2,1,0} bitcast(%pallas_call.1), metadata={op_name="jit(pallas_qk_norm_rope)/pallas_call" scheduling_name="bitcast.2.0" stack_frame_id=7}
}
```

Pallas, tiled, HLO:
```diff
  ENTRY %main.1 (q.1: f32[4,1024,16,128], g.1: f32[128], cos.1: f32[1024,64], sin.1: f32[1024,64]) -> f32[4,1024,16,128] {
    %sin.1 = f32[1024,64]{1,0} parameter(3)
    %cos.1 = f32[1024,64]{1,0} parameter(2)
    %g.1 = f32[128]{0} parameter(1)
    %q.1 = f32[4,1024,16,128]{3,2,1,0} parameter(0)
    %bitcast.2 = f32[4096,2048]{1,0} bitcast(%q.1), metadata={op_name="q" scheduling_name="bitcast.2"}
+ %wrapped_broadcast = f32[64,128]{1,0} fusion(%g.1), kind=kLoop, calls=%wrapped_broadcast_computation, metadata={op_name="jit(tiled_qk_norm_rope)/broadcast_in_dim" scheduling_name="wrapped_broadcast" stack_frame_id=4}, backend_config={"device_type":"DEVICE_TYPE_INVALID","force_earliest_schedule":false,"native_emitter_backend_config":{"type":"NATIVE_EMITTER_TYPE_INVALID","unroll_factor":0},"operation_queue_id":"0","reification_cost":[]}
+ %pallas_call.1 = f32[4096,2048]{1,0} custom-call(%bitcast.2, %wrapped_broadcast, %cos.1, %sin.1), custom_call_target="mosaic_gpu_v2", operand_layout_constraints={f32[4096,2048]{1,0}, f32[64,128]{1,0}, f32[1024,64]{1,0}, f32[1024,64]{1,0}}, api_version=API_VERSION_TYPED_FFI, metadata={op_name="jit(tiled_qk_norm_rope)/pallas_call" scheduling_name="pallas_call.1" stack_frame_id=5}, backend_config={kernel_hash = "\F2\87\85\CC\81T\C3\EB\E6\F6\09\B3\0E \FBS?\90\0B\A1\D3\A7\97\FAs\9B\F4\1FT\F1\E9>", module = "<truncated 149369 bytes>", use_custom_barrier = false, uses_xla_collective_metadata = false}
  ROOT %bitcast.1.0 = f32[4,1024,16,128]{3,2,1,0} bitcast(%pallas_call.1), metadata={op_name="jit(tiled_qk_norm_rope)/pallas_call" scheduling_name="bitcast.1.0" stack_frame_id=5}
}
```

Hope reading this was informative to some 🙂🦉.
