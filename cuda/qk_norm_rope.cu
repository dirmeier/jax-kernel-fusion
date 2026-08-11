
#include <cuda_runtime.h>

#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

constexpr float kEps = 1e-6f;
constexpr unsigned kHeadDim = 128;
constexpr unsigned kVec4 = kHeadDim / 4;
constexpr unsigned kHalfVec4 = kHeadDim / 8;
constexpr unsigned kWarpsPerBlock = 8;
constexpr unsigned kFullMask = 0xffffffffu;

__device__ __forceinline__ float4 shfl_xor_f4(float4 v, int lane_mask) {
  float4 r;
  r.x = __shfl_xor_sync(kFullMask, v.x, lane_mask);
  r.y = __shfl_xor_sync(kFullMask, v.y, lane_mask);
  r.z = __shfl_xor_sync(kFullMask, v.z, lane_mask);
  r.w = __shfl_xor_sync(kFullMask, v.w, lane_mask);
  return r;
}

__global__ void qk_norm_rope_kernel(const float4* __restrict__ q,
                                    const float4* __restrict__ g,
                                    const float4* __restrict__ cos_tab,
                                    const float4* __restrict__ sin_tab,
                                    float4* __restrict__ out, unsigned seq,
                                    unsigned heads, unsigned vectors,
                                    float eps) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned warp = threadIdx.x >> 5;
  const unsigned vec = blockIdx.x * kWarpsPerBlock + warp;
  if (vec >= vectors) return;

  const unsigned base = vec * kVec4;
  const float4 v = q[base + lane];
  float acc = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;

#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    acc += __shfl_xor_sync(kFullMask, acc, off);
  }
  const float inv = rsqrtf(acc / static_cast<float>(kHeadDim) + eps);

  const float4 gg = g[lane];
  float4 qn;
  qn.x = v.x * inv * gg.x;
  qn.y = v.y * inv * gg.y;
  qn.z = v.z * inv * gg.z;
  qn.w = v.w * inv * gg.w;

  const float4 partner = shfl_xor_f4(qn, 16);
  const unsigned s = (vec / heads) % seq;
  const unsigned row = s * kHalfVec4 + (lane & 15u);
  const float4 c = cos_tab[row];
  const float4 sn = sin_tab[row];
  const float sign = (lane < 16u) ? -1.0f : 1.0f;

  float4 r;
  r.x = qn.x * c.x + sign * partner.x * sn.x;
  r.y = qn.y * c.y + sign * partner.y * sn.y;
  r.z = qn.z * c.z + sign * partner.z * sn.z;
  r.w = qn.w * c.w + sign * partner.w * sn.w;
  out[base + lane] = r;
}

ffi::Error QkNormRopeCudaImpl(cudaStream_t stream, ffi::Buffer<ffi::F32> q,
                              ffi::Buffer<ffi::F32> g,
                              ffi::Buffer<ffi::F32> cos,
                              ffi::Buffer<ffi::F32> sin,
                              ffi::Result<ffi::Buffer<ffi::F32>> out) {
  const auto dims = q.dimensions();
  if (dims.size() != 4) {
    return ffi::Error::InvalidArgument("q must have rank 4 [B, S, H, D]");
  }

  const auto batch = static_cast<unsigned>(dims[0]);
  const auto seq = static_cast<unsigned>(dims[1]);
  const auto heads = static_cast<unsigned>(dims[2]);
  const auto head_dim = static_cast<unsigned>(dims[3]);

  if (head_dim != kHeadDim) {
    return ffi::Error::InvalidArgument("need to have head dimension of 128");
  }
  if (g.element_count() != static_cast<int64_t>(head_dim)) {
    return ffi::Error::InvalidArgument("g must have shape [D]");
  }
  const auto half = static_cast<int64_t>(head_dim) / 2;
  if (cos.element_count() != static_cast<int64_t>(seq) * half ||
      sin.element_count() != static_cast<int64_t>(seq) * half) {
    return ffi::Error::InvalidArgument("cos/sin must have shape [S, D/2]");
  }

  const unsigned vectors = batch * seq * heads;
  const unsigned blocks = (vectors + kWarpsPerBlock - 1) / kWarpsPerBlock;

  qk_norm_rope_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
      reinterpret_cast<const float4*>(q.typed_data()),
      reinterpret_cast<const float4*>(g.typed_data()),
      reinterpret_cast<const float4*>(cos.typed_data()),
      reinterpret_cast<const float4*>(sin.typed_data()),
      reinterpret_cast<float4*>(out->typed_data()), seq, heads, vectors, kEps);

  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return ffi::Error::Internal(std::string("kernel launch failed: ") +
                                cudaGetErrorString(err));
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(QkNormRopeCuda, QkNormRopeCudaImpl,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()  // q
                                  .Arg<ffi::Buffer<ffi::F32>>()  // g
                                  .Arg<ffi::Buffer<ffi::F32>>()  // cos
                                  .Arg<ffi::Buffer<ffi::F32>>()  // sin
                                  .Ret<ffi::Buffer<ffi::F32>>());
