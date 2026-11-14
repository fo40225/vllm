// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Plan of the exact GEMM1 of the CUTLASS NVFP4 grouped-GEMM path.
//
// The gate and up halves of an expert's w13 carry different global scales, and
// the grouped GEMM takes one alpha per group. Instead of folding the ratio into
// a later scale (which rounds the up half to bf16 in the wrong scale), every
// half is its own group: w13 [E, 2N, K / 2] is read as [2E, N, K / 2], group
// 2e + h being half h of expert e. All of them multiply the same activation
// rows, so the rows are stored twice (the grouped GEMM derives the output rows
// from the same offsets as the activation rows): the group of half h reads and
// writes the rows [h * R + offset_e, h * R + offset_e + M_e), with R the
// capacity of one copy.

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <initializer_list>

#include "muse_c_ops.h"

namespace muse_spark {

namespace {

constexpr int kThreads = 256;

__global__ void __launch_bounds__(kThreads)
    split_gemm1_prepare_kernel(int4 const* __restrict__ a,
                               int4* __restrict__ a2, int64_t rows,
                               int64_t row_vecs,
                               int32_t const* __restrict__ expert_offsets,
                               int32_t const* __restrict__ blockscale_offsets,
                               int32_t const* __restrict__ problem_sizes,
                               int32_t* __restrict__ expert_offsets2,
                               int32_t* __restrict__ blockscale_offsets2,
                               int32_t* __restrict__ problem_sizes2,
                               int num_experts) {
  // The tables of the 2E groups.
  if (blockIdx.x == 0) {
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
      int32_t const offset = expert_offsets[e];
      int32_t const sf_offset = blockscale_offsets[e];
      int32_t const m = problem_sizes[3 * e];
      int32_t const n = problem_sizes[3 * e + 1] / 2;
      int32_t const k = problem_sizes[3 * e + 2];
      for (int h = 0; h < 2; ++h) {
        int const g = 2 * e + h;
        expert_offsets2[g] = offset + h * static_cast<int32_t>(rows);
        // The activation scale factors are shared by the two halves.
        blockscale_offsets2[g] = sf_offset;
        problem_sizes2[3 * g] = m;
        problem_sizes2[3 * g + 1] = n;
        problem_sizes2[3 * g + 2] = k;
      }
    }
  }
  // Both copies of the rows that experts own (the rows after them are unused).
  int64_t const valid =
      min(rows, static_cast<int64_t>(expert_offsets[num_experts]));
  int64_t const total = valid * row_vecs;
  int64_t const second = rows * row_vecs;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    int4 const v = a[i];
    a2[i] = v;
    a2[second + i] = v;
  }
}

}  // namespace

void split_gemm1_prepare(at::Tensor const& a_fp4,
                         at::Tensor const& expert_offsets,
                         at::Tensor const& blockscale_offsets,
                         at::Tensor const& problem_sizes, at::Tensor& a_fp4_x2,
                         at::Tensor& expert_offsets_x2,
                         at::Tensor& blockscale_offsets_x2,
                         at::Tensor& problem_sizes_x2) {
  using Tensors = std::initializer_list<at::Tensor const*>;
  for (at::Tensor const* t :
       Tensors{&a_fp4, &expert_offsets, &blockscale_offsets, &problem_sizes,
               &a_fp4_x2, &expert_offsets_x2, &blockscale_offsets_x2,
               &problem_sizes_x2}) {
    TORCH_CHECK(t->is_cuda() && t->is_contiguous(),
                "all tensors must be contiguous CUDA tensors");
    TORCH_CHECK(t->device() == a_fp4.device(),
                "all tensors must be on the same device");
  }
  TORCH_CHECK(a_fp4.scalar_type() == at::kByte &&
                  a_fp4_x2.scalar_type() == at::kByte,
              "the activations must be packed fp4 (uint8)");
  for (at::Tensor const* t :
       Tensors{&expert_offsets, &blockscale_offsets, &problem_sizes,
               &expert_offsets_x2, &blockscale_offsets_x2, &problem_sizes_x2}) {
    TORCH_CHECK(t->scalar_type() == at::kInt, "the tables must be int32");
  }
  TORCH_CHECK(a_fp4.dim() == 2 && a_fp4_x2.dim() == 2, "rows must be 2D");
  int64_t const rows = a_fp4.size(0);
  int64_t const row_bytes = a_fp4.size(1);
  TORCH_CHECK(row_bytes % 16 == 0, "a row must be a multiple of 16 bytes");
  TORCH_CHECK(a_fp4_x2.size(0) == 2 * rows && a_fp4_x2.size(1) == row_bytes,
              "a_fp4_x2 must hold two copies of a_fp4");
  int64_t const num_experts = problem_sizes.size(0);
  TORCH_CHECK(problem_sizes.dim() == 2 && problem_sizes.size(1) == 3,
              "problem_sizes must be [num_experts, 3]");
  TORCH_CHECK(expert_offsets.numel() == num_experts + 1 &&
                  blockscale_offsets.numel() == num_experts + 1,
              "the offsets must have num_experts + 1 entries");
  TORCH_CHECK(expert_offsets_x2.numel() == 2 * num_experts &&
                  blockscale_offsets_x2.numel() == 2 * num_experts &&
                  problem_sizes_x2.dim() == 2 &&
                  problem_sizes_x2.size(0) == 2 * num_experts &&
                  problem_sizes_x2.size(1) == 3,
              "the tables of the split groups must have 2 * num_experts rows");
  TORCH_CHECK(rows < (int64_t{1} << 30), "too many rows for int32 offsets");

  const c10::cuda::CUDAGuard device_guard(a_fp4.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(a_fp4.get_device()).stream();
  int64_t const row_vecs = row_bytes / 16;
  // A grid that covers a few full waves of the largest copy; the rest is a
  // grid-stride loop. The grid does not depend on the number of valid rows,
  // which is only known on the device.
  int64_t const max_blocks = (rows * row_vecs + kThreads - 1) / kThreads;
  int const grid = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(max_blocks, 4096)));
  split_gemm1_prepare_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<int4 const*>(a_fp4.data_ptr()),
      reinterpret_cast<int4*>(a_fp4_x2.data_ptr()), rows, row_vecs,
      expert_offsets.data_ptr<int32_t>(),
      blockscale_offsets.data_ptr<int32_t>(), problem_sizes.data_ptr<int32_t>(),
      expert_offsets_x2.data_ptr<int32_t>(),
      blockscale_offsets_x2.data_ptr<int32_t>(),
      problem_sizes_x2.data_ptr<int32_t>(), static_cast<int>(num_experts));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace muse_spark
