// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Router GEMV of a Muse Spark decode layer: out[t][n] = sum_k x[t][k] * w[n][k]
// for up to 8 token rows, fp32 activations x [T, 8192] and fp32 weights w [N,
// 8192] (N a multiple of the rows per CTA), fp32 out [T, N]. The router of the
// model is a [T, 8192] x [8192, 256] fp32 product that cuBLAS runs as a dot-
// product kernel plus a reduction (T = 1: 3.7 + 1.2 us with the weights in L2)
// or as a SIMT sgemm plus a split-K reduction (T = 2 .. 8: 6.5 + 2.6 .. 3.3
// us), and that the routing kernel waits for; here it is one kernel that reads
// the 8 MiB of weights once, with all of its loads in flight.
//
// Decomposition: a CTA of W warps owns W / WPR output rows; WPR warps share a
// row, warp `wr` takes the slice [wr * K / WPR, (wr + 1) * K / WPR) of it, lane
// l of a warp loads the 16-byte chunks l, l + 32, ... of its slice (a warp
// instruction reads 512 contiguous bytes). The weights do not depend on any
// earlier kernel: a launch as a programmatic dependent (pdl) loads them before
// griddepcontrol.wait and holds them in registers; the model launches it
// without (pdl = false) today, because its predecessor on the aux stream (the
// previous layer's expert prefetch) does not trigger early.
//
// Order of the additions of one output element (row n, token t), fixed and
// independent of T, the grid and the run:
//   thread: four accumulators a[e], e = element position inside the chunk;
//     chunks in ascending j; a[e] = fma(w, x, a[e]); thread value (a0 + a1) +
//     (a2 + a3)
//   warp:   xor butterfly over the 32 lanes (16, 8, 4, 2, 1)
//   row:    the WPR warp values added in ascending warp order.
// This is not the order of cuBLAS: the logits differ from the cuBLAS ones in
// the last bits (deterministic; accuracy as good as the fp32 dot product of
// cuBLAS, see tests/kernels/core/test_muse_spark_router_gemv.py).

// Only c10 headers: <ATen/cuda/CUDAContext.h> needs cusparse.h and
// cusolverDn.h, which the stock runtime images do not ship.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <cstdint>

#include "muse_c_ops.h"

namespace muse_spark {
namespace rgemv {

constexpr int kHidden = 8192;

// Weights: streamed once, no L1 allocation. A plain load (not ld.global.nc):
// ptxas moves a .nc load below griddepcontrol.wait.
__device__ __forceinline__ float4 ld_w(const float* p) {
  float4 v;
  asm volatile("ld.global.L1::no_allocate.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
               : "l"(p)
               : "memory");
  return v;
}
// Activations: written by the kernel in front of this one; a plain load that
// stays behind griddepcontrol.wait.
__device__ __forceinline__ float4 ld_x(const float* p) {
  float4 v;
  asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];"
      : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
      : "l"(p)
      : "memory");
  return v;
}

template <int T, int W, int WPR>
__global__ void __launch_bounds__(W * 32)
    router_gemv_kernel(const float* x, const float* __restrict__ w,
                       float* __restrict__ out, int N) {
  constexpr int K = kHidden;
  constexpr int CH = K / 4;       // 16-byte chunks per row
  constexpr int SLICE = CH / WPR;  // chunks per warp slice
  constexpr int LPL = SLICE / 32;  // chunks per lane
  constexpr int RPC = W / WPR;     // rows per CTA
  static_assert(SLICE % 32 == 0 && T >= 1 && T <= 8, "geometry");
  __shared__ float part[W][T];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int rg = warp / WPR, wr = warp % WPR;
  const int row = blockIdx.x * RPC + rg;

  float4 wv[LPL];
  const float* wp = w + static_cast<int64_t>(row) * K + 4 * (wr * SLICE + lane);
#pragma unroll
  for (int j = 0; j < LPL; ++j) wv[j] = ld_w(wp + 4 * 32 * j);
  // The routing kernel after this one (Triton, a programmatic dependent that
  // waits for this grid to finish) may be launched now; without a dependent
  // this does nothing.
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  asm volatile("griddepcontrol.wait;" ::: "memory");

  float acc[T][4];
#pragma unroll
  for (int t = 0; t < T; ++t)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[t][e] = 0.f;
  const float* xp = x + 4 * (wr * SLICE + lane);
#pragma unroll
  for (int j = 0; j < LPL; ++j) {
#pragma unroll
    for (int t = 0; t < T; ++t) {
      const float4 xv = ld_x(xp + static_cast<int64_t>(t) * K + 4 * 32 * j);
      acc[t][0] = __fmaf_rn(wv[j].x, xv.x, acc[t][0]);
      acc[t][1] = __fmaf_rn(wv[j].y, xv.y, acc[t][1]);
      acc[t][2] = __fmaf_rn(wv[j].z, xv.z, acc[t][2]);
      acc[t][3] = __fmaf_rn(wv[j].w, xv.w, acc[t][3]);
    }
  }
#pragma unroll
  for (int t = 0; t < T; ++t) {
    float v = (acc[t][0] + acc[t][1]) + (acc[t][2] + acc[t][3]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_xor_sync(0xffffffffu, v, off);
    acc[t][0] = v;
  }
  if (lane < T) {
    float v = 0.f;
#pragma unroll
    for (int t = 0; t < T; ++t)
      if (lane == t) v = acc[t][0];
    part[warp][lane] = v;
  }
  __syncthreads();
  if (threadIdx.x < RPC * T) {
    const int r = threadIdx.x / T, t = threadIdx.x % T;
    float s = part[r * WPR][t];
#pragma unroll
    for (int k = 1; k < WPR; ++k) s += part[r * WPR + k][t];
    out[static_cast<int64_t>(t) * N + blockIdx.x * RPC + r] = s;
  }
}

template <int T, int W, int WPR>
void launch(const float* x, const float* w, float* out, int N, bool pdl,
            cudaStream_t stream) {
  constexpr int RPC = W / WPR;
  TORCH_CHECK(N % RPC == 0, "router_gemv: N must be a multiple of ", RPC);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(N / RPC);
  cfg.blockDim = dim3(W * 32);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = attr;
  cfg.numAttrs = pdl ? 1 : 0;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, router_gemv_kernel<T, W, WPR>, x, w,
                                    out, N));
}

// Geometry per batch size: the activations are re-read by every CTA, so the
// larger batches take fewer, larger CTAs (4 rows each instead of 2).
template <int T>
void launch_t(const float* x, const float* w, float* out, int N, bool pdl,
              cudaStream_t stream) {
  if constexpr (T <= 2) {
    launch<T, 8, 4>(x, w, out, N, pdl, stream);
  } else {
    launch<T, 16, 4>(x, w, out, N, pdl, stream);
  }
}

}  // namespace rgemv

void router_gemv(at::Tensor const& x, at::Tensor const& w, at::Tensor& out,
                 bool pdl) {
  TORCH_CHECK(x.is_cuda() && w.is_cuda() && out.is_cuda(),
              "router_gemv: CUDA tensors expected");
  TORCH_CHECK(x.scalar_type() == at::kFloat && w.scalar_type() == at::kFloat &&
                  out.scalar_type() == at::kFloat,
              "router_gemv: float32 tensors expected");
  TORCH_CHECK(x.dim() == 2 && x.size(1) == rgemv::kHidden &&
                  x.is_contiguous(),
              "router_gemv: x must be a contiguous [T, ",
              rgemv::kHidden, "] tensor");
  const int64_t T = x.size(0);
  TORCH_CHECK(T >= 1 && T <= 8, "router_gemv: 1 <= T <= 8, got ", T);
  TORCH_CHECK(w.dim() == 2 && w.size(1) == rgemv::kHidden &&
                  w.is_contiguous(),
              "router_gemv: w must be a contiguous [N, ", rgemv::kHidden,
              "] tensor");
  const int64_t N = w.size(0);
  TORCH_CHECK(out.dim() == 2 && out.size(0) == T && out.size(1) == N &&
                  out.is_contiguous(),
              "router_gemv: out must be a contiguous [T, N] tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(w.data_ptr()) % 16 == 0,
              "router_gemv: operands must be 16-byte aligned");
  const c10::cuda::CUDAGuard guard(x.device());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  const float* xp = x.data_ptr<float>();
  const float* wp = w.data_ptr<float>();
  float* op = out.data_ptr<float>();
  const int n = static_cast<int>(N);
  switch (T) {
    case 1: rgemv::launch_t<1>(xp, wp, op, n, pdl, stream); break;
    case 2: rgemv::launch_t<2>(xp, wp, op, n, pdl, stream); break;
    case 3: rgemv::launch_t<3>(xp, wp, op, n, pdl, stream); break;
    case 4: rgemv::launch_t<4>(xp, wp, op, n, pdl, stream); break;
    case 5: rgemv::launch_t<5>(xp, wp, op, n, pdl, stream); break;
    case 6: rgemv::launch_t<6>(xp, wp, op, n, pdl, stream); break;
    case 7: rgemv::launch_t<7>(xp, wp, op, n, pdl, stream); break;
    default: rgemv::launch_t<8>(xp, wp, op, n, pdl, stream); break;
  }
}

}  // namespace muse_spark
