// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Ported from vLLM csrc/muse_spark_mul_tanh_soft_cap_kernel.cu
// (vllm-project/vllm#62132) for the `_muse_C` extension: the torch::stable
// boundary becomes at::Tensor and the alignment-aware vectorized loop of
// vLLM's quantization/vectorization_utils.cuh is inlined in the form this
// kernel needs. The kernel math is unchanged.

// Only c10 headers: <ATen/cuda/CUDAContext.h> needs cusparse.h and
// cusolverDn.h, which the stock runtime images do not ship.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "muse_c_ops.h"

namespace muse_spark {
namespace soft_cap {

// 16 B of output per thread; low-precision inputs are widened in registers.
static constexpr int kVecSize = 4;

template <typename T, int N>
struct __align__(N * sizeof(T)) VecN {
  T val[N];
};

// out[i] = op(in[i]) over `len` elements, `stride` threads cooperating. Takes
// the vector path where both pointers allow it and the scalar path for the
// misaligned head and the tail (vllm::vectorize_with_alignment).
template <int VEC_SIZE, typename InT, typename OutT, typename Op>
__device__ __forceinline__ void transform_with_alignment(const InT* in,
                                                         OutT* out, int len,
                                                         int tid, int stride,
                                                         Op&& op) {
  constexpr int WIDTH = VEC_SIZE * sizeof(InT);
  constexpr int OUT_WIDTH = VEC_SIZE * sizeof(OutT);
  uintptr_t const addr = reinterpret_cast<uintptr_t>(in);
  uintptr_t const out_addr = reinterpret_cast<uintptr_t>(out);

  using vin_t = VecN<InT, VEC_SIZE>;
  using vout_t = VecN<OutT, VEC_SIZE>;

  // Fast path when input and output are both fully aligned.
  bool const can_vec = ((addr & (WIDTH - 1)) == 0) &&
                       ((out_addr & (OUT_WIDTH - 1)) == 0) &&
                       ((len & (VEC_SIZE - 1)) == 0);
  if (can_vec) {
    int const num_vec = len / VEC_SIZE;
    auto const* v_in = reinterpret_cast<const vin_t*>(in);
    auto* v_out = reinterpret_cast<vout_t*>(out);
    for (int i = tid; i < num_vec; i += stride) {
      vin_t src = v_in[i];
      vout_t tmp;
#pragma unroll
      for (int j = 0; j < VEC_SIZE; ++j) op(tmp.val[j], src.val[j]);
      v_out[i] = tmp;
    }
    return;
  }

  int const misalignment_offset = addr & (WIDTH - 1);
  int const alignment_bytes = WIDTH - misalignment_offset;
  int prefix_elems = (alignment_bytes & (WIDTH - 1)) / sizeof(InT);
  prefix_elems = min(prefix_elems, len);

  // The prefix aligns `in`; if that does not also align `out`, vectorizing is
  // impossible and the whole copy stays scalar.
  if (((out_addr + prefix_elems * sizeof(OutT)) & (OUT_WIDTH - 1)) != 0) {
    for (int i = tid; i < len; i += stride) op(out[i], in[i]);
    return;
  }

  for (int i = tid; i < prefix_elems; i += stride) op(out[i], in[i]);

  in += prefix_elems;
  out += prefix_elems;
  len -= prefix_elems;

  int const num_vec = len / VEC_SIZE;
  auto const* v_in = reinterpret_cast<const vin_t*>(in);
  auto* v_out = reinterpret_cast<vout_t*>(out);
  for (int i = tid; i < num_vec; i += stride) {
    vin_t src = v_in[i];
    vout_t tmp;
#pragma unroll
    for (int j = 0; j < VEC_SIZE; ++j) op(tmp.val[j], src.val[j]);
    v_out[i] = tmp;
  }

  int const tail_start = num_vec * VEC_SIZE;
  for (int i = tid + tail_start; i < len; i += stride) op(out[i], in[i]);
}

// A functor rather than a lambda: a device lambda needs --extended-lambda.
template <typename scalar_t>
struct SoftCapOp {
  float multiplier;
  float soft_cap_temp;

  __device__ __forceinline__ void operator()(float& dst,
                                             const scalar_t& src) const {
    // Multiply then divide, as MulTanhSoftCap.forward does, so fp32 rounding
    // matches the eager path.
    dst = soft_cap_temp *
          tanhf(static_cast<float>(src) * multiplier / soft_cap_temp);
  }
};

// blockIdx.x walks rows, blockIdx.y splits a row. Rows are the sampled tokens
// (often one, at decode) while a row is the whole vocab, so splitting the row
// is what keeps the GPU busy; a row-per-block grid would leave a single-token
// decode on one SM.
template <typename scalar_t>
__global__ void muse_spark_mul_tanh_soft_cap_kernel(
    float* __restrict__ out, const scalar_t* __restrict__ input, int d,
    float multiplier, float soft_cap_temp) {
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * d;
  const scalar_t* __restrict__ in_row = input + row_offset;
  float* __restrict__ out_row = out + row_offset;

  const int tid = blockIdx.y * blockDim.x + threadIdx.x;
  const int stride = gridDim.y * blockDim.x;

  transform_with_alignment<kVecSize>(
      in_row, out_row, d, tid, stride,
      SoftCapOp<scalar_t>{multiplier, soft_cap_temp});
}

template <typename scalar_t>
void launch(at::Tensor& out, at::Tensor const& input, double multiplier,
            double soft_cap_temp, dim3 grid, dim3 block,
            cudaStream_t stream) {
  const int d = static_cast<int>(input.size(input.dim() - 1));
  muse_spark_mul_tanh_soft_cap_kernel<scalar_t><<<grid, block, 0, stream>>>(
      out.data_ptr<float>(),
      reinterpret_cast<const scalar_t*>(input.const_data_ptr()), d,
      static_cast<float>(multiplier), static_cast<float>(soft_cap_temp));
}

}  // namespace soft_cap

// out = soft_cap_temp * tanh(input * multiplier / soft_cap_temp), fp32.
void muse_spark_mul_tanh_soft_cap(at::Tensor& out, at::Tensor const& input,
                                  double multiplier, double soft_cap_temp) {
  TORCH_CHECK(input.is_cuda() && out.is_cuda() &&
                  out.get_device() == input.get_device(),
              "muse_spark_mul_tanh_soft_cap: input and out must be CUDA "
              "tensors on the same device");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Float ||
                  input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "muse_spark_mul_tanh_soft_cap: input must be float32, float16 "
              "or bfloat16");
  TORCH_CHECK(out.scalar_type() == at::ScalarType::Float,
              "muse_spark_mul_tanh_soft_cap: out must be float32");
  TORCH_CHECK(input.is_contiguous(),
              "muse_spark_mul_tanh_soft_cap: input must be contiguous");
  TORCH_CHECK(out.is_contiguous(),
              "muse_spark_mul_tanh_soft_cap: out must be contiguous");
  TORCH_CHECK(out.numel() == input.numel(),
              "muse_spark_mul_tanh_soft_cap: out and input must have the "
              "same number of elements");
  TORCH_CHECK(soft_cap_temp != 0.0,
              "muse_spark_mul_tanh_soft_cap: soft_cap_temp must be nonzero");

  if (input.numel() == 0) {
    return;
  }

  const int d = static_cast<int>(input.size(input.dim() - 1));
  const int64_t num_tokens = input.numel() / d;

  constexpr int kThreads = 128;
  const int64_t vecs_per_row =
      (d + soft_cap::kVecSize - 1) / soft_cap::kVecSize;
  const int64_t blocks_to_cover = (vecs_per_row + kThreads - 1) / kThreads;
  constexpr int64_t kTargetBlocks = 32768;
  const int64_t blocks_per_row = std::clamp<int64_t>(
      kTargetBlocks / num_tokens, 1, std::max<int64_t>(blocks_to_cover, 1));

  dim3 grid(static_cast<unsigned>(num_tokens),
            static_cast<unsigned>(blocks_per_row));
  dim3 block(kThreads);

  const c10::cuda::CUDAGuard device_guard(input.get_device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  switch (input.scalar_type()) {
    case at::ScalarType::Float:
      soft_cap::launch<float>(out, input, multiplier, soft_cap_temp, grid,
                              block, stream);
      break;
    case at::ScalarType::Half:
      soft_cap::launch<c10::Half>(out, input, multiplier, soft_cap_temp, grid,
                                  block, stream);
      break;
    default:
      soft_cap::launch<c10::BFloat16>(out, input, multiplier, soft_cap_temp,
                                      grid, block, stream);
      break;
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace muse_spark
