// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// The pre-expert RMSNorm of the Muse Spark MoE layer and the NVFP4 quantization of its
// output in one launch, for the large-batch muse decode kernel (9 to 128 tokens):
//
//   out  = bf16(((x * rsqrt(mean(x^2) + eps)) * (f32(w) + gain)))          [T, 4096]
//   xq   = the NVFP4 codes [T, 2048] and linear e4m3 scales [T, 256] of `out`, with the
//          arithmetic of the activation quantization of the large-batch kernel's
//          preparation launch (cvt_warp_fp16_to_fp4, gs[0] = 1 / a13)
//
// `out` is bit for bit what vLLM's Triton `_rmsnorm_fwd_kernel` writes for the
// pre_expert_norm (D = 4096, weight applied after the normalization, bT = 2,
// num_warps = 8, single tile), and the codes and scales are those the preparation
// launch of nvfp4_decode_moe_large would write for that `out`, so the large-batch kernel
// can take them with flag 64 (external xq) and skip the preparation launch.
//
// The RMSNorm half mirrors the SASS of the Triton kernel (Triton 3.7.1, ptxas 13.x), not
// its source: ptxas contracts some of the multiplies of the sum of squares into FMAs and
// the reduction tree is the one Triton generates for the block layout. One CTA of 256
// threads handles one row; thread t owns the 16-byte packs t and 256 + t of the row. The
// chain over its 16 elements is round(x1 * x1), then fma over x0, x2 .. x7 of the first
// pack and x0 .. x7 of the second; the tree is a xor butterfly over the 32 lanes
// (offsets 16 .. 1), the 8 warp partials of the row through shared memory and a butterfly
// of 4, 2, 1; then fma(sum_sq, 2^-12, eps), rsqrt.approx.ftz.f32 and
// (x * rs) * (f32(w) + gain). It is the same code as the all-gather + RMSNorm epilogue of
// the host topology plugin (csrc/boundary.cuh there), proven bit for bit against the
// Triton kernel; everything is written with explicit rounded intrinsics so that nvcc and
// ptxas cannot contract or reorder it.
//
// The quantization pairs two lanes (xor 1) per 16-element block: the two packs 2m and
// 2m + 1 of the row, which the RMSNorm layout keeps in adjacent lanes of one warp.

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <initializer_list>

#include "muse_c_ops.h"
#include "nvfp4_utils.cuh"

namespace muse_spark {
namespace {

constexpr int kNormHidden = 4096;
constexpr int kNormRowBytes = kNormHidden / 2;   // packed fp4 bytes of a row
constexpr int kNormSfCols = kNormHidden / 16;    // scales of a row
constexpr int kNormThreads = 256;
constexpr int kNormPacks = 2;                    // 16-byte packs of a thread
constexpr int kNormMaxTokens = 128;

__device__ __forceinline__ float rsqrt_ftz(float x) {
  float r;
  asm("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return r;
}
__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) {
  return __uint_as_float(w & 0xffff0000u);
}

// The 8 bf16 of a 16-byte pack as floats (exact widening).
__device__ __forceinline__ void unpack_bf16x8(const uint4& p, float (&v)[8]) {
  v[0] = bf_lo(p.x);
  v[1] = bf_hi(p.x);
  v[2] = bf_lo(p.y);
  v[3] = bf_hi(p.y);
  v[4] = bf_lo(p.z);
  v[5] = bf_hi(p.z);
  v[6] = bf_lo(p.w);
  v[7] = bf_hi(p.w);
}
__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&v);
}
__device__ __forceinline__ uint4 pack_bf16x8(const float (&v)[8]) {
  return make_uint4(pack_bf16x2(v[0], v[1]), pack_bf16x2(v[2], v[3]),
                    pack_bf16x2(v[4], v[5]), pack_bf16x2(v[6], v[7]));
}

// Thread chain of the first 8 elements of a pack: the odd element 1 starts the chain as a
// plain product, element 0 is the first FMA.
__device__ __forceinline__ float chain8(const float (&v)[8]) {
  float acc = __fmul_rn(v[1], v[1]);
  acc = __fmaf_rn(v[0], v[0], acc);
#pragma unroll
  for (int k = 2; k < 8; ++k) acc = __fmaf_rn(v[k], v[k], acc);
  return acc;
}
// Continuation over a second pack (elements 0..7 in order).
__device__ __forceinline__ float chain8_cont(float acc, const float (&v)[8]) {
#pragma unroll
  for (int k = 0; k < 8; ++k) acc = __fmaf_rn(v[k], v[k], acc);
  return acc;
}

template <int FROM>
__device__ __forceinline__ float bfly(float v) {
#pragma unroll
  for (int o = FROM; o >= 1; o >>= 1) {
    v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, o));
  }
  return v;
}

// Load of the weight packs ahead of griddepcontrol.wait of a programmatic dependent: a
// volatile asm with a memory clobber, so that ptxas keeps it above the wait.
__device__ __forceinline__ uint4 ld_early(const void* p) {
  uint4 v;
  asm volatile("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

__global__ void __launch_bounds__(kNormThreads)
    k_rmsnorm_nvfp4(const __nv_bfloat16* __restrict__ x,
                    const __nv_bfloat16* __restrict__ w, float eps, float gain,
                    const float* __restrict__ gs, __nv_bfloat16* __restrict__ out,
                    uint8_t* __restrict__ codes, uint8_t* __restrict__ sf) {
  __shared__ float sm[8];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  // The weight never changes: load it before the wait for the producer of x. Let the
  // large-batch kernel launch now, it waits for this kernel to finish before it reads
  // anything written here.
  uint4 wx[kNormPacks];
#pragma unroll
  for (int c = 0; c < kNormPacks; ++c) wx[c] = ld_early(w + 8 * (kNormThreads * c + tid));
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");

  float v[kNormPacks][8];
#pragma unroll
  for (int c = 0; c < kNormPacks; ++c) {
    const uint4 pk = *reinterpret_cast<const uint4*>(
        x + static_cast<size_t>(row) * kNormHidden + 8 * (kNormThreads * c + tid));
    unpack_bf16x8(pk, v[c]);
  }
  float p = chain8_cont(chain8(v[0]), v[1]);
  p = bfly<16>(p);
  if (lane == 0) sm[warp] = p;
  __syncthreads();
  const float sum_sq = bfly<4>(sm[lane & 7]);
  const float rs = rsqrt_ftz(__fmaf_rn(sum_sq, 0.000244140625f, eps));

  const float gs0 = gs[0];
#pragma unroll
  for (int c = 0; c < kNormPacks; ++c) {
    float wv[8], y[8];
    unpack_bf16x8(wx[c], wv);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      y[k] = __fmul_rn(__fmul_rn(v[c][k], rs), __fadd_rn(wv[k], gain));
    }
    const uint4 o = pack_bf16x8(y);
    const int idx = kNormThreads * c + tid;  // 8-element vector of the row
    *reinterpret_cast<uint4*>(out + static_cast<size_t>(row) * kNormHidden + 8 * idx) = o;
    // The NVFP4 quantization of the rounded output, a lane pair per 16-element block.
    PackedVec<__nv_bfloat16> vec;
    *reinterpret_cast<uint4*>(&vec) = o;
    uint8_t* sf_out =
        (lane & 1) ? nullptr : sf + static_cast<size_t>(row) * kNormSfCols + (idx >> 1);
    const uint32_t q = cvt_warp_fp16_to_fp4<__nv_bfloat16, 2>(vec, gs0, sf_out);
    *reinterpret_cast<uint32_t*>(codes + static_cast<size_t>(row) * kNormRowBytes + idx * 4) =
        q;
  }
}

void check_tensor(at::Tensor const& t, at::ScalarType type, char const* name,
                  std::initializer_list<int64_t> shape) {
  TORCH_CHECK(t.is_cuda() && t.is_contiguous(), name,
              " must be a contiguous CUDA tensor");
  TORCH_CHECK(t.scalar_type() == type, name, " has the wrong dtype");
  TORCH_CHECK(t.dim() == static_cast<int64_t>(shape.size()), name,
              " has the wrong rank");
  int d = 0;
  for (int64_t s : shape) {
    TORCH_CHECK(s < 0 || t.size(d) == s, name, " has the wrong shape in dim ", d);
    ++d;
  }
}

}  // namespace

void rmsnorm_nvfp4_quant(at::Tensor const& x, at::Tensor const& w, double eps,
                         double gain, at::Tensor const& gs, at::Tensor& out,
                         at::Tensor& xq_data, at::Tensor& xq_sf, bool pdl) {
  const int64_t T = x.size(0);
  check_tensor(x, at::kBFloat16, "x", {-1, kNormHidden});
  check_tensor(w, at::kBFloat16, "w", {kNormHidden});
  check_tensor(gs, at::kFloat, "gs", {-1});
  check_tensor(out, at::kBFloat16, "out", {T, kNormHidden});
  check_tensor(xq_data, at::kByte, "xq_data", {T, kNormRowBytes});
  check_tensor(xq_sf, at::kByte, "xq_sf", {T, kNormSfCols});
  TORCH_CHECK(T >= 1 && T <= kNormMaxTokens, "token count ", T, " is outside 1..",
              kNormMaxTokens);
  for (const at::Tensor* t : std::initializer_list<const at::Tensor*>{&x, &w, &out, &xq_data, &xq_sf}) {
    TORCH_CHECK(reinterpret_cast<uintptr_t>(t->data_ptr()) % 16 == 0,
                "the operands must be 16-byte aligned");
  }
  const c10::cuda::OptionalCUDAGuard guard(x.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned>(T));
  cfg.blockDim = dim3(kNormThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(
      &cfg, k_rmsnorm_nvfp4, reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), static_cast<float>(eps),
      static_cast<float>(gain), reinterpret_cast<const float*>(gs.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
      reinterpret_cast<uint8_t*>(xq_data.data_ptr()),
      reinterpret_cast<uint8_t*>(xq_sf.data_ptr())));
}

}  // namespace muse_spark
