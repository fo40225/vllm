/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Adapted from vLLM csrc/quantization/fp4/nvfp4_experts_quant.cu (v0.32.0):
// the fused SiLU+Mul NVFP4 experts quantization for activations laid out as
// [up | gate] instead of [gate | up], and for the "split" layout of the exact
// GEMM1 (two [rows, N] matrices, one per half, stacked in one buffer). Only the
// gate/up load offsets differ from the stock kernel; the expert lookup,
// scale-factor layout and numerics are unchanged. Build with the stock flags:
// no --use_fast_math, and NVFP4_ENABLE_ELTS16 undefined (8 elements per
// thread).
//
// It also holds the row-gather variant of the stock activation quantization
// (gather_scaled_fp4_experts_quant), which reads the rows of the permuted
// activation matrix straight from the token rows instead of from a shuffled
// copy. It runs the same per-element arithmetic, so its bytes are those of the
// stock quantization of the shuffled copy.
//
// The last pair (pack_nvfp4_rows / gather_packed_fp4_experts) splits that
// quantization around a collective: the rows are quantized once on the rank
// that owns them, travel as packed fp4 codes plus linear fp8 scales, and the
// consumer only moves the bytes into the permuted rows and the swizzled
// scale-factor layout. The quantization of a row does not depend on the expert
// (one static global scale), so the bytes equal those of the gather variant.

// Only c10 headers: <ATen/cuda/CUDAContext.h> needs cusparse.h and cusolverDn.h,
// which the stock runtime images do not ship.
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <cuda_fp8.h>

#include <algorithm>
#include <atomic>

#include "launch_bounds_utils.h"
#include "muse_c_ops.h"
#include "nvfp4_utils.cuh"

namespace muse_spark {

// Number of SMs of `device`, queried once per device.
static int multi_processor_count(int device) {
  constexpr int kMaxCachedDevices = 64;
  // Static storage is zero-initialized: 0 means not queried yet.
  static std::atomic<int> cache[kMaxCachedDevices];
  bool const cacheable = device >= 0 && device < kMaxCachedDevices;
  if (cacheable) {
    int const cached = cache[device].load(std::memory_order_relaxed);
    if (cached > 0) {
      return cached;
    }
  }
  int value = 0;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount,
                                        device));
  if (cacheable) {
    cache[device].store(value, std::memory_order_relaxed);
  }
  return value;
}

// Where the up and gate values of row r, column vector c are, in units of
// PackedVec: `in[r * row_stride + c + up_off]` and `in[r * row_stride + c +
// gate_off]`.
struct InputLayout {
  int64_t row_stride;
  int64_t up_off;
  int64_t gate_off;
};

// NVFP4 quantization kernel for experts (low-latency path).
// Reads up and gate where `layout` puts them (up||gate: the halves of the stock
// kernel's gate||up layout exchanged) and fuses up * SiLU(gate) before
// quantization. Use UE4M3 scale factors.
template <class Type, bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_mul_and_silu_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                            float const* SFScale, uint32_t* out,
                            uint32_t* SFout, uint32_t* input_offset_by_experts,
                            uint32_t* output_scale_offset_by_experts,
                            int n_experts, bool low_latency,
                            InputLayout layout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  int32_t const numValidRows =
      min(numRows,
          static_cast<int32_t>(__ldca(&input_offset_by_experts[n_experts])));

  // Each global thread processes one element
  for (int globalIdx = tid; globalIdx < numValidRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    // Calculate which row and column this global thread should process
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // Find index within the experts using different strategies based on expert
    // count
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    if constexpr (SMALL_NUM_EXPERTS) {
      for (int i = 0; i < n_experts; i++) {
        uint32_t current_offset = __ldca(&input_offset_by_experts[i]);
        uint32_t next_offset = __ldca(&input_offset_by_experts[i + 1]);
        if (rowIdx >= current_offset && rowIdx < next_offset) {
          rowIdx_in_expert = rowIdx - current_offset;
          expert_idx = i;
          break;
        }
      }
    } else {
      // Load input offsets into registers first, then do the computation.
      // Local array size set to 17 because of register limit.
      uint32_t local_offsets[17];
      for (int chunk_start = 0; chunk_start < n_experts; chunk_start += 16) {
        *reinterpret_cast<int4*>(local_offsets) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start]));
        *reinterpret_cast<int4*>(local_offsets + 4) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start + 4]));
        *reinterpret_cast<int4*>(local_offsets + 8) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start + 8]));
        *reinterpret_cast<int4*>(local_offsets + 12) =
            __ldca(reinterpret_cast<const int4*>(
                &input_offset_by_experts[chunk_start + 12]));
        local_offsets[16] = __ldca(&input_offset_by_experts[chunk_start + 16]);

// Check against the 16 loaded offsets
#pragma unroll
        for (int i = 0; i < 16; i++) {
          if (rowIdx >= local_offsets[i] && rowIdx < local_offsets[i + 1]) {
            rowIdx_in_expert = rowIdx - local_offsets[i];
            expert_idx = chunk_start + i;
            break;
          }
        }
      }
    }

    // Load up and gate from where the layout puts them, then apply the fused
    // SiLU+Mul
    int64_t inOffset =
        static_cast<int64_t>(rowIdx) * layout.row_stride + colIdx;
    PackedVec in_vec_up =
        reinterpret_cast<PackedVec const*>(in)[inOffset + layout.up_off];
    PackedVec in_vec_gate =
        reinterpret_cast<PackedVec const*>(in)[inOffset + layout.gate_off];
    PackedVec quant_input = compute_silu_mul(in_vec_gate, in_vec_up);

    // Get the output tensor offset.
    // Same as inOffset because 8 elements are packed into one uint32_t.
    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    // Get the global scaling factor, which will be applied to the SF.
    // Note SFScale is the same as next GEMM's alpha, which is
    // (448.f / (Alpha_A / 6.f)).
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF>(
        quant_input, SFScaleVal, sf_out);
  }
}

// NVFP4 quantization kernel for LARGE_M_TOPK = true (large m_topk optimized
// version). Reads up and gate where `layout` puts them and fuses
// up * SiLU(gate) before quantization.
template <class Type, bool SMALL_NUM_EXPERTS = false>
__global__ void __launch_bounds__(1024, VLLM_BLOCKS_PER_SM(1024))
    cvt_mul_and_silu_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                            float const* SFScale, uint32_t* out,
                            uint32_t* SFout, uint32_t* input_offset_by_experts,
                            uint32_t* output_scale_offset_by_experts,
                            int n_experts, InputLayout layout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  extern __shared__ uint32_t shared_input_offsets[];

  // Load input offsets into shared memory.
  // If n_experts is larger than 4, use vectorized int4 to save instructions.
  // If n_experts is smaller than 4, read directly.
  if constexpr (SMALL_NUM_EXPERTS) {
    for (int i = threadIdx.x; i < n_experts + 1; i += blockDim.x) {
      shared_input_offsets[i] = input_offset_by_experts[i];
    }
  } else {
    for (int i = threadIdx.x * 4; i < n_experts; i += blockDim.x * 4) {
      *reinterpret_cast<int4*>(&shared_input_offsets[i]) =
          *reinterpret_cast<const int4*>(&input_offset_by_experts[i]);
    }
    if (threadIdx.x == 0) {
      shared_input_offsets[n_experts] = input_offset_by_experts[n_experts];
    }
  }

  __syncthreads();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  int32_t const numValidRows =
      min(numRows, static_cast<int32_t>(shared_input_offsets[n_experts]));

  // Each global thread processes one element
  for (int globalIdx = tid; globalIdx < numValidRows * colsPerRow;
       globalIdx += gridDim.x * blockDim.x) {
    // Calculate which row and column this global thread should process
    int rowIdx = globalIdx / colsPerRow;
    int colIdx = globalIdx % colsPerRow;

    // Find expert using binary search for better performance with large m_topk
    int rowIdx_in_expert = 0;
    int expert_idx = 0;

    // Binary search through experts using shared memory
    int left = 0, right = n_experts - 1;
    while (left <= right) {
      int mid = (left + right) / 2;
      // Get offsets: shared_input_offsets[i] corresponds to
      // input_offset_by_experts[i]
      uint32_t mid_offset = shared_input_offsets[mid];
      uint32_t next_offset = shared_input_offsets[mid + 1];

      if (rowIdx >= mid_offset && rowIdx < next_offset) {
        rowIdx_in_expert = rowIdx - mid_offset;
        expert_idx = mid;
        break;
      } else if (rowIdx < mid_offset) {
        right = mid - 1;
      } else {
        left = mid + 1;
      }
    }

    // Load up and gate from where the layout puts them, then apply the fused
    // SiLU+Mul
    int64_t inOffset =
        static_cast<int64_t>(rowIdx) * layout.row_stride + colIdx;
    PackedVec in_vec_up =
        reinterpret_cast<PackedVec const*>(in)[inOffset + layout.up_off];
    PackedVec in_vec_gate =
        reinterpret_cast<PackedVec const*>(in)[inOffset + layout.gate_off];
    PackedVec quant_input = compute_silu_mul(in_vec_gate, in_vec_up);

    int64_t outOffset = rowIdx * colsPerRow + colIdx;
    auto& out_pos = out[outOffset];

    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];

    uint32_t* SFout_in_expert =
        SFout + output_scale_offset_by_experts[expert_idx] * numKTiles;

    auto sf_out =
        cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                           CVT_FP4_NUM_THREADS_PER_SF>(
            rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);

    out_pos = cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF>(
        quant_input, SFScaleVal, sf_out);
  }
}

template <typename T>
void quant_impl(void* output, void* output_scale, void* input,
                void* input_global_scale, void* input_offset_by_experts,
                void* output_scale_offset_by_experts, int m_topk, int k,
                int n_experts, int multiProcessorCount, InputLayout layout,
                cudaStream_t stream) {
  // Grid, Block size.
  // Each thread converts 8 values.
  int const workSizePerRow = k / CVT_FP4_ELTS_PER_THREAD;
  int const totalWorkSize = m_topk * workSizePerRow;
  dim3 block(std::min(workSizePerRow, 512));
  // Get number of blocks per SM
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  dim3 grid(std::min(static_cast<int>((totalWorkSize + block.x - 1) / block.x),
                     multiProcessorCount * numBlocksPerSM));
  while (grid.x <= multiProcessorCount && block.x > 64) {
    grid.x *= 2;
    block.x = (block.x + 1) / 2;
  }

  int const blockRepeat =
      (totalWorkSize + block.x * grid.x - 1) / (block.x * grid.x);
  if (blockRepeat > 1) {
    size_t shared_mem_size = (n_experts + 1) * sizeof(uint32_t);
    // The shared-memory vectorized offset load only handles full 4-expert
    // chunks. Use the scalar specialization for the remainder cases.
    if (n_experts >= 4 && n_experts % 4 == 0) {
      cvt_mul_and_silu_to_fp4<T, false>
          <<<grid, block, shared_mem_size, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts, layout);
    } else {
      cvt_mul_and_silu_to_fp4<T, true>
          <<<grid, block, shared_mem_size, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts, layout);
    }
  } else {
    // The low-latency vectorized expert lookup only handles full 16-expert
    // chunks. Fall back to the scalar lookup path for the remainder cases.
    if (n_experts >= 16 && n_experts % 16 == 0) {
      cvt_mul_and_silu_to_fp4<T, false>
          <<<grid, block, 0, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts, /* bool low_latency */ true, layout);
    } else {
      cvt_mul_and_silu_to_fp4<T, true>
          <<<grid, block, 0, stream>>>(
              m_topk, k, reinterpret_cast<T*>(input),
              reinterpret_cast<float*>(input_global_scale),
              reinterpret_cast<uint32_t*>(output),
              reinterpret_cast<uint32_t*>(output_scale),
              reinterpret_cast<uint32_t*>(input_offset_by_experts),
              reinterpret_cast<uint32_t*>(output_scale_offset_by_experts),
              n_experts, /* bool low_latency */ true, layout);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// NVFP4 quantization of gathered rows: row r of the output is the quantization
// of row `row_map[r]` of `in`, with the global scale and the scale-factor
// layout of the expert that owns row r. Only the first
// input_offset_by_experts[n_experts] rows (the local routes of an expert-parallel
// rank) are read; the rows past them are neither read nor written, which is what
// a shuffled copy of the activations would spend most of its time on.
//
// A block handles whole rows (grid-stride), so the expert lookup is done once per
// row, and every thread converts 8 elements per row, like the stock kernel. The
// block size divides the row width, so each warp is wholly inside or outside a
// row (the scale factor needs the max of a thread pair).
template <class Type>
__global__ void __launch_bounds__(512)
    cvt_gather_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                      int32_t const* rowMap, float const* SFScale,
                      uint32_t* out, uint32_t* SFout,
                      int32_t const* input_offset_by_experts,
                      int32_t const* output_scale_offset_by_experts,
                      int n_experts) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  extern __shared__ int32_t shared_row_starts[];
  for (int i = threadIdx.x; i <= n_experts; i += blockDim.x) {
    shared_row_starts[i] = input_offset_by_experts[i];
  }
  __syncthreads();

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;
  int const colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  int32_t const numValidRows = min(numRows, shared_row_starts[n_experts]);

  for (int row = blockIdx.x; row < numValidRows; row += gridDim.x) {
    // The expert of the row: the last one that starts at or before it (an
    // expert without rows shares its start with the next one).
    int left = 0, right = n_experts - 1;
    while (left < right) {
      int mid = (left + right + 1) >> 1;
      if (shared_row_starts[mid] <= row) {
        left = mid;
      } else {
        right = mid - 1;
      }
    }
    int const expert_idx = left;
    int const rowIdx_in_expert = row - shared_row_starts[expert_idx];
    int64_t const srcRow = rowMap[row];
    float const SFScaleVal = SFScale == nullptr ? 1.0f : SFScale[expert_idx];
    uint32_t* SFout_in_expert =
        SFout + static_cast<uint32_t>(output_scale_offset_by_experts[expert_idx]) *
                    numKTiles;

    for (int colIdx = threadIdx.x; colIdx < colsPerRow; colIdx += blockDim.x) {
      PackedVec in_vec =
          reinterpret_cast<PackedVec const*>(in)[srcRow * colsPerRow + colIdx];
      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx_in_expert, colIdx, numKTiles, SFout_in_expert);
      out[static_cast<int64_t>(row) * colsPerRow + colIdx] =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF>(
              in_vec, SFScaleVal, sf_out);
    }
  }
}

template <typename T>
void gather_quant_impl(void* output, void* output_scale, void* input,
                       void* row_map, void* input_global_scale,
                       void* input_offset_by_experts,
                       void* output_scale_offset_by_experts, int m_topk, int k,
                       int n_experts, int multiProcessorCount,
                       cudaStream_t stream) {
  if (m_topk == 0) {
    return;
  }
  int const colsPerRow = k / CVT_FP4_ELTS_PER_THREAD;
  // The largest power of two <= 512 that divides the row width (the caller
  // checked that it is a multiple of 32).
  int block = 512;
  while (colsPerRow % block != 0) {
    block >>= 1;
  }
  // Rows past the local ones return at once, so the grid only has to cover the
  // rows that can exist; a few waves of blocks keep every SM fed.
  int const grid = std::min(m_topk, multiProcessorCount * 8);
  size_t const shared_mem_size = (n_experts + 1) * sizeof(int32_t);
  cvt_gather_to_fp4<T><<<grid, block, shared_mem_size, stream>>>(
      m_topk, k, reinterpret_cast<T*>(input),
      reinterpret_cast<int32_t*>(row_map),
      reinterpret_cast<float*>(input_global_scale),
      reinterpret_cast<uint32_t*>(output),
      reinterpret_cast<uint32_t*>(output_scale),
      reinterpret_cast<int32_t*>(input_offset_by_experts),
      reinterpret_cast<int32_t*>(output_scale_offset_by_experts), n_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// NVFP4 quantization of dense rows into a packed wire format: row r of `out`
// (row_stride bytes) holds the numCols / 2 bytes of fp4 codes, then the numCols / 16
// linear (row-major, unswizzled) e4m3 scale factors, then whatever the caller keeps
// in the tail. The arithmetic is that of cvt_gather_to_fp4 with one global scale.
//
// A block handles whole rows (grid-stride) and every thread converts 8 elements
// per row. The block size divides the row width, so each warp is wholly inside
// or outside a row (the scale factor needs the max of a thread pair).
template <class Type>
__global__ void __launch_bounds__(512)
    cvt_rows_to_packed_fp4(int32_t numRows, int32_t numCols, Type const* in,
                           float const* SFScale, uint8_t* out,
                           int64_t rowStride) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  int const colsPerRow = numCols / CVT_FP4_ELTS_PER_THREAD;
  float const SFScaleVal = SFScale[0];
  for (int row = blockIdx.x; row < numRows; row += gridDim.x) {
    uint8_t* row_out = out + static_cast<int64_t>(row) * rowStride;
    uint8_t* row_sf = row_out + numCols / 2;
    for (int colIdx = threadIdx.x; colIdx < colsPerRow; colIdx += blockDim.x) {
      PackedVec in_vec =
          reinterpret_cast<PackedVec const*>(in)[static_cast<int64_t>(row) *
                                                     colsPerRow +
                                                 colIdx];
      // One thread of a pair writes the scale factor (the block size is even,
      // so the parity of the thread is that of the column).
      uint8_t* sf_out =
          threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF == 0
              ? row_sf + colIdx / CVT_FP4_NUM_THREADS_PER_SF
              : nullptr;
      reinterpret_cast<uint32_t*>(row_out)[colIdx] =
          cvt_warp_fp16_to_fp4<Type, CVT_FP4_NUM_THREADS_PER_SF>(
              in_vec, SFScaleVal, sf_out);
    }
  }
}

template <typename T>
void pack_rows_impl(void* packed, void* input, void* input_global_scale,
                    int num_rows, int k, int64_t row_stride,
                    int multiProcessorCount, cudaStream_t stream) {
  if (num_rows == 0) {
    return;
  }
  int const colsPerRow = k / CVT_FP4_ELTS_PER_THREAD;
  // The largest power of two <= 512 that divides the row width (the caller
  // checked that it is a multiple of 32).
  int block = 512;
  while (colsPerRow % block != 0) {
    block >>= 1;
  }
  int const grid = std::min(num_rows, multiProcessorCount * 8);
  cvt_rows_to_packed_fp4<T><<<grid, block, 0, stream>>>(
      num_rows, k, reinterpret_cast<T*>(input),
      reinterpret_cast<float*>(input_global_scale),
      reinterpret_cast<uint8_t*>(packed), row_stride);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Row r of the experts' activations from the packed rows of cvt_rows_to_packed_fp4:
// the fp4 codes of row `row_map[r]` go to `out` row r, and its scale factors to
// the swizzled layout of the expert that owns row r (the layout of
// cvt_quant_to_fp4_get_sf_out_offset, written as the 4 scale factors of one
// k-tile at a time). Only the first input_offset_by_experts[n_experts] rows are
// written. A block of 128 threads handles one row at a time.
__global__ void __launch_bounds__(128)
    gather_packed_to_fp4(int32_t numRows, int32_t numCols,
                         uint8_t const* packed, int64_t packedStride,
                         int32_t const* rowMap, uint8_t* out, uint32_t* SFout,
                         int32_t const* input_offset_by_experts,
                         int32_t const* output_scale_offset_by_experts,
                         int n_experts) {
  extern __shared__ int32_t shared_row_starts[];
  for (int i = threadIdx.x; i <= n_experts; i += blockDim.x) {
    shared_row_starts[i] = input_offset_by_experts[i];
  }
  __syncthreads();

  int32_t const numKTiles = (numCols + 63) / 64;
  int const codeVecs = numCols / 32;  // 16 B vectors of fp4 codes per row
  int32_t const numValidRows = min(numRows, shared_row_starts[n_experts]);

  for (int row = blockIdx.x; row < numValidRows; row += gridDim.x) {
    // The expert of the row: the last one that starts at or before it (an
    // expert without rows shares its start with the next one).
    int left = 0, right = n_experts - 1;
    while (left < right) {
      int mid = (left + right + 1) >> 1;
      if (shared_row_starts[mid] <= row) {
        left = mid;
      } else {
        right = mid - 1;
      }
    }
    int const expert_idx = left;
    int32_t const mIdx = row - shared_row_starts[expert_idx];
    uint8_t const* src =
        packed + static_cast<int64_t>(rowMap[row]) * packedStride;
    uint32_t* SFout_in_expert =
        SFout + static_cast<uint32_t>(output_scale_offset_by_experts[expert_idx]) *
                    numKTiles;
    int32_t const mTileIdx = mIdx >> 7;
    int32_t const outerMIdx = mIdx & 31;
    int32_t const innerMIdx = (mIdx >> 5) & 3;

    for (int v = threadIdx.x; v < codeVecs; v += blockDim.x) {
      reinterpret_cast<int4*>(out + static_cast<int64_t>(row) * (numCols / 2))[v] =
          reinterpret_cast<int4 const*>(src)[v];
    }
    uint32_t const* src_sf = reinterpret_cast<uint32_t const*>(src + numCols / 2);
    for (int kTile = threadIdx.x; kTile < numKTiles; kTile += blockDim.x) {
      // Byte offset (mTile * numKTiles + kTile) * 512 + outerM * 16 + innerM * 4
      // of the 4 scale factors kTile * 4 .. kTile * 4 + 3, as a word index.
      int64_t const word =
          (((static_cast<int64_t>(mTileIdx) * numKTiles + kTile) << 9) |
           (outerMIdx << 4) | (innerMIdx << 2)) >>
          2;
      SFout_in_expert[word] = src_sf[kTile];
    }
  }
}

#define MUSE_CHECK_INPUT(x, m)                          \
  TORCH_CHECK(x.is_cuda(), m, " must be a CUDA tensor"); \
  TORCH_CHECK(x.is_contiguous(), m, " must be contiguous");

// Same validation as the stock fp4 experts quantization entry points.
static void validate_fp4_experts_quant_inputs(
    at::Tensor const& output, at::Tensor const& output_scale,
    at::Tensor const& input, at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts, int64_t m_topk,
    int64_t k) {
  MUSE_CHECK_INPUT(output, "output");
  MUSE_CHECK_INPUT(output_scale, "output_scale");
  MUSE_CHECK_INPUT(input, "input");
  MUSE_CHECK_INPUT(input_global_scale, "input_global_scale");
  MUSE_CHECK_INPUT(input_offset_by_experts, "input_offset_by_experts");
  MUSE_CHECK_INPUT(output_scale_offset_by_experts,
                   "output_scale_offset_by_experts");

  TORCH_CHECK(output.device() == input.device() &&
                  output_scale.device() == input.device() &&
                  input_global_scale.device() == input.device() &&
                  input_offset_by_experts.device() == input.device() &&
                  output_scale_offset_by_experts.device() == input.device(),
              "all tensors must be on the same device");

  TORCH_CHECK(output.dim() == 2);
  TORCH_CHECK(output_scale.dim() == 2);
  TORCH_CHECK(input.dim() == 2);
  TORCH_CHECK(input_global_scale.dim() == 1);
  TORCH_CHECK(input_offset_by_experts.dim() == 1);
  TORCH_CHECK(output_scale_offset_by_experts.dim() == 1);

  TORCH_CHECK(input.scalar_type() == at::kHalf ||
              input.scalar_type() == at::kBFloat16);
  TORCH_CHECK(input_global_scale.scalar_type() == at::kFloat);
  TORCH_CHECK(input_offset_by_experts.scalar_type() == at::kInt);
  TORCH_CHECK(output_scale_offset_by_experts.scalar_type() == at::kInt);
  // output is uint8 (two nvfp4 values are packed into one uint8)
  // output_scale is int32 (four fp8 values are packed into one int32)
  TORCH_CHECK(output.scalar_type() == at::kByte);
  TORCH_CHECK(output_scale.scalar_type() == at::kInt);

  const int BLOCK_SIZE = 16;
  TORCH_CHECK(k % BLOCK_SIZE == 0, "k must be a multiple of 16");
  auto n_experts = input_global_scale.size(0);
  TORCH_CHECK(input_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output_scale_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output.size(0) == m_topk);
  TORCH_CHECK(output.size(1) == k / 2);
  int scales_k = k / BLOCK_SIZE;
  // 4 means the swizzle requirement by nvidia nvfp4.
  int padded_k = (scales_k + (4 - 1)) / 4 * 4;
  // 4 means 4 fp8 values are packed into one int32
  TORCH_CHECK(output_scale.size(1) * 4 == padded_k);
}

// Quantizes up * silu(gate) of the m_topk rows of `input`, which `layout`
// describes, for k columns per half.
static void silu_mul_quant(at::Tensor& output, at::Tensor& output_scale,
                           at::Tensor const& input,
                           at::Tensor const& input_global_scale,
                           at::Tensor const& input_offset_by_experts,
                           at::Tensor const& output_scale_offset_by_experts,
                           int64_t m_topk, int64_t k, InputLayout layout) {
  validate_fp4_experts_quant_inputs(output, output_scale, input,
                                    input_global_scale, input_offset_by_experts,
                                    output_scale_offset_by_experts, m_topk, k);

  auto n_experts = input_global_scale.size(0);
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  int const multiProcessorCount = multi_processor_count(input.get_device());

  if (input.scalar_type() == at::kHalf) {
    quant_impl<half>(output.data_ptr(), output_scale.data_ptr(),
                     input.data_ptr(), input_global_scale.data_ptr(),
                     input_offset_by_experts.data_ptr(),
                     output_scale_offset_by_experts.data_ptr(), m_topk, k,
                     n_experts, multiProcessorCount, layout, stream);
  } else {
    quant_impl<__nv_bfloat16>(
        output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
        input_global_scale.data_ptr(), input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(), m_topk, k, n_experts,
        multiProcessorCount, layout, stream);
  }
}

void mul_and_silu_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  auto m_topk = input.size(0);
  // Input has up || gate layout, so k = input.size(1) / 2
  auto k_times_2 = input.size(1);
  TORCH_CHECK(k_times_2 % 2 == 0, "input width must be even (up || gate)");
  auto k = k_times_2 / 2;
  // Row-major [m_topk, 2k]: the up half first, the gate half k columns later.
  int64_t const cols_per_half = k / CVT_FP4_ELTS_PER_THREAD;
  InputLayout const layout{2 * cols_per_half, 0, cols_per_half};
  silu_mul_quant(output, output_scale, input, input_global_scale,
                 input_offset_by_experts, output_scale_offset_by_experts,
                 m_topk, k, layout);
}

void silu_and_mul_split_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts, bool gate_first) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  TORCH_CHECK(input.size(0) % 2 == 0,
              "input must hold the two halves as stacked [m_topk, k] matrices");
  // Row r of the first half is row r of the input, row r of the second half
  // is row m_topk + r.
  auto m_topk = input.size(0) / 2;
  auto k = input.size(1);
  int64_t const cols = k / CVT_FP4_ELTS_PER_THREAD;
  int64_t const second = m_topk * cols;
  InputLayout const layout{cols, gate_first ? second : 0,
                           gate_first ? 0 : second};
  silu_mul_quant(output, output_scale, input, input_global_scale,
                 input_offset_by_experts, output_scale_offset_by_experts,
                 m_topk, k, layout);
}

void gather_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& row_map, at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts) {
  TORCH_CHECK(input.dim() == 2, "input must be 2D");
  auto k = input.size(1);
  auto m_topk = row_map.size(0);

  validate_fp4_experts_quant_inputs(output, output_scale, input,
                                    input_global_scale, input_offset_by_experts,
                                    output_scale_offset_by_experts, m_topk, k);
  MUSE_CHECK_INPUT(row_map, "row_map");
  TORCH_CHECK(row_map.dim() == 1 && row_map.scalar_type() == at::kInt &&
                  row_map.device() == input.device(),
              "row_map must be a 1D int32 tensor on the device of input");
  TORCH_CHECK((k / CVT_FP4_ELTS_PER_THREAD) % 32 == 0,
              "k must be a multiple of 256");

  auto n_experts = input_global_scale.size(0);
  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  int const multiProcessorCount = multi_processor_count(input.get_device());

  if (input.scalar_type() == at::kHalf) {
    gather_quant_impl<half>(
        output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
        row_map.data_ptr(), input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(), m_topk, k, n_experts,
        multiProcessorCount, stream);
  } else {
    gather_quant_impl<__nv_bfloat16>(
        output.data_ptr(), output_scale.data_ptr(), input.data_ptr(),
        row_map.data_ptr(), input_global_scale.data_ptr(),
        input_offset_by_experts.data_ptr(),
        output_scale_offset_by_experts.data_ptr(), m_topk, k, n_experts,
        multiProcessorCount, stream);
  }
}

void pack_nvfp4_rows(at::Tensor& packed, at::Tensor const& input,
                     at::Tensor const& input_global_scale) {
  MUSE_CHECK_INPUT(packed, "packed");
  MUSE_CHECK_INPUT(input, "input");
  MUSE_CHECK_INPUT(input_global_scale, "input_global_scale");
  TORCH_CHECK(input.dim() == 2 && packed.dim() == 2, "input and packed must be 2D");
  TORCH_CHECK(packed.device() == input.device() &&
                  input_global_scale.device() == input.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(input.scalar_type() == at::kHalf ||
              input.scalar_type() == at::kBFloat16);
  TORCH_CHECK(input_global_scale.scalar_type() == at::kFloat &&
                  input_global_scale.numel() == 1,
              "input_global_scale must be one float");
  TORCH_CHECK(packed.scalar_type() == at::kByte, "packed must be uint8");
  auto const num_rows = input.size(0);
  auto const k = input.size(1);
  TORCH_CHECK((k / CVT_FP4_ELTS_PER_THREAD) % 32 == 0,
              "k must be a multiple of 256");
  TORCH_CHECK(packed.size(0) == num_rows, "packed must have one row per row");
  TORCH_CHECK(packed.size(1) >= k / 2 + k / CVT_FP4_SF_VEC_SIZE &&
                  packed.size(1) % 16 == 0,
              "a packed row holds k / 2 code bytes and k / 16 scale bytes, and "
              "its width is a multiple of 16");

  const c10::cuda::CUDAGuard device_guard(input.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(input.get_device()).stream();
  int const multiProcessorCount = multi_processor_count(input.get_device());
  if (input.scalar_type() == at::kHalf) {
    pack_rows_impl<half>(packed.data_ptr(), input.data_ptr(),
                         input_global_scale.data_ptr(), num_rows, k,
                         packed.size(1), multiProcessorCount, stream);
  } else {
    pack_rows_impl<__nv_bfloat16>(packed.data_ptr(), input.data_ptr(),
                                  input_global_scale.data_ptr(), num_rows, k,
                                  packed.size(1), multiProcessorCount, stream);
  }
}

void gather_packed_fp4_experts(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& packed,
    at::Tensor const& row_map, at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts, int64_t k) {
  MUSE_CHECK_INPUT(output, "output");
  MUSE_CHECK_INPUT(output_scale, "output_scale");
  MUSE_CHECK_INPUT(packed, "packed");
  MUSE_CHECK_INPUT(row_map, "row_map");
  MUSE_CHECK_INPUT(input_offset_by_experts, "input_offset_by_experts");
  MUSE_CHECK_INPUT(output_scale_offset_by_experts,
                   "output_scale_offset_by_experts");
  TORCH_CHECK(output.device() == packed.device() &&
                  output_scale.device() == packed.device() &&
                  row_map.device() == packed.device() &&
                  input_offset_by_experts.device() == packed.device() &&
                  output_scale_offset_by_experts.device() == packed.device(),
              "all tensors must be on the same device");
  TORCH_CHECK(packed.dim() == 2 && output.dim() == 2 && output_scale.dim() == 2 &&
              row_map.dim() == 1 && input_offset_by_experts.dim() == 1 &&
              output_scale_offset_by_experts.dim() == 1);
  TORCH_CHECK(packed.scalar_type() == at::kByte && output.scalar_type() == at::kByte &&
              output_scale.scalar_type() == at::kInt &&
              row_map.scalar_type() == at::kInt &&
              input_offset_by_experts.scalar_type() == at::kInt &&
              output_scale_offset_by_experts.scalar_type() == at::kInt);
  TORCH_CHECK((k / CVT_FP4_ELTS_PER_THREAD) % 32 == 0,
              "k must be a multiple of 256");
  TORCH_CHECK(packed.size(1) >= k / 2 + k / CVT_FP4_SF_VEC_SIZE &&
                  packed.size(1) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(packed.data_ptr()) % 16 == 0,
              "packed rows hold k / 2 code bytes and k / 16 scale bytes, and "
              "their width and base address are multiples of 16");
  auto const m_topk = row_map.size(0);
  auto const n_experts = input_offset_by_experts.size(0) - 1;
  TORCH_CHECK(n_experts >= 1 &&
              output_scale_offset_by_experts.size(0) == n_experts + 1);
  TORCH_CHECK(output.size(0) == m_topk && output.size(1) == k / 2);
  TORCH_CHECK(output_scale.size(1) * 4 == (k / 16 + 3) / 4 * 4);
  if (m_topk == 0) {
    return;
  }
  const c10::cuda::CUDAGuard device_guard(packed.device());
  const cudaStream_t stream =
      c10::cuda::getCurrentCUDAStream(packed.get_device()).stream();
  int const multiProcessorCount = multi_processor_count(packed.get_device());
  int const grid = std::min<int64_t>(m_topk, multiProcessorCount * 8);
  size_t const shared_mem_size = (n_experts + 1) * sizeof(int32_t);
  gather_packed_to_fp4<<<grid, 128, shared_mem_size, stream>>>(
      m_topk, k, reinterpret_cast<uint8_t const*>(packed.data_ptr()),
      packed.size(1), reinterpret_cast<int32_t const*>(row_map.data_ptr()),
      reinterpret_cast<uint8_t*>(output.data_ptr()),
      reinterpret_cast<uint32_t*>(output_scale.data_ptr()),
      reinterpret_cast<int32_t const*>(input_offset_by_experts.data_ptr()),
      reinterpret_cast<int32_t const*>(output_scale_offset_by_experts.data_ptr()),
      n_experts);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace muse_spark
