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

// Subset of vLLM csrc/quantization/fp4/nvfp4_utils.cuh (v0.32.0) for the
// NVFP4 experts quantization kernel. It keeps the configuration that
// nvfp4_experts_quant.cu uses (8 elements per thread, UE4M3 scale factors);
// the 16-element, UE8M0 (MXFP4) and non-experts helpers are dropped.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp8.h>

#include "cuda_vec_utils.cuh"

constexpr int CVT_FP4_ELTS_PER_THREAD = 8;
constexpr int CVT_FP4_SF_VEC_SIZE = 16;

namespace muse_spark {

// Convert 4 float2 values into 8 e2m1 values (represented as one uint32_t).
__device__ __forceinline__ uint32_t fp32_vec8_to_e2m1(float2 (&array)[4]) {
  uint32_t val;
  asm volatile(
      "{\n"
      ".reg .b8 byte0;\n"
      ".reg .b8 byte1;\n"
      ".reg .b8 byte2;\n"
      ".reg .b8 byte3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte1, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte2, %6, %5;\n"
      "cvt.rn.satfinite.e2m1x2.f32   byte3, %8, %7;\n"
      "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
      "}\n"
      : "=r"(val)
      : "f"(array[0].x), "f"(array[0].y), "f"(array[1].x), "f"(array[1].y),
        "f"(array[2].x), "f"(array[2].y), "f"(array[3].x), "f"(array[3].y));
  return val;
}

// Fast reciprocal.
__device__ __forceinline__ float reciprocal_approximate_ftz(float a) {
  float b;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
  return b;
}

// Compute SF output offset for swizzled tensor core layout.
// SF layout: [numMTiles, numKTiles, 32, 4, 4]
// Caller must precompute: numKTiles = (numCols + 63) / 64
template <class SFType, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ uint8_t* cvt_quant_to_fp4_get_sf_out_offset(
    int rowIdx, int colIdx, int32_t numKTiles, SFType* SFout) {
  static_assert(CVT_FP4_NUM_THREADS_PER_SF == 1 ||
                CVT_FP4_NUM_THREADS_PER_SF == 2);

  // One pair of threads write one SF to global memory.
  // TODO: stage through smem for packed STG.32
  // is it better than STG.8 from 4 threads ?
  if (threadIdx.x % CVT_FP4_NUM_THREADS_PER_SF != 0) {
    return nullptr;
  }

  // SF vector index (16 elements share one SF in the K dimension).
  int32_t kIdx = colIdx / CVT_FP4_NUM_THREADS_PER_SF;
  int32_t mIdx = rowIdx;

  // Decompose indices using bitwise ops (all divisors are powers of 2).
  // SF layout [numMTiles, numKTiles, 32 (mTile), 4 (mTile), 4(kTile)]
  int32_t mTileIdx = mIdx >> 7;         // mIdx / 128
  int32_t outerMIdx = mIdx & 31;        // mIdx % 32
  int32_t innerMIdx = (mIdx >> 5) & 3;  // (mIdx / 32) % 4
  int32_t kTileIdx = kIdx >> 2;         // kIdx / 4
  int32_t innerKIdx = kIdx & 3;         // kIdx % 4

  // Compute global SF offset: mTileIdx * (numKTiles * 512) + kTileIdx * 512 +
  //                           outerMIdx * 16 + innerMIdx * 4 + innerKIdx
  // Use bitwise OR for non-overlapping lower bits.
  int64_t SFOffset = (static_cast<int64_t>(mTileIdx) * numKTiles + kTileIdx)
                         << 9 |
                     (outerMIdx << 4) | (innerMIdx << 2) | innerKIdx;

  return reinterpret_cast<uint8_t*>(SFout) + SFOffset;
}

// Quantizes the provided PackedVec into the uint32_t output
template <class Type, int CVT_FP4_NUM_THREADS_PER_SF>
__device__ __forceinline__ uint32_t cvt_warp_fp16_to_fp4(
    PackedVec<Type>& vec, float SFScaleVal, uint8_t* SFout) {
  // Get absolute maximum values among the local 8 values.
  auto localMax = __habs2(vec.elts[0]);

  // Local maximum value.
#pragma unroll
  for (int i = 1; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    localMax = __hmax2(localMax, __habs2(vec.elts[i]));
  }

  // Get the absolute maximum among all 16 values (two threads).

  if constexpr (CVT_FP4_NUM_THREADS_PER_SF == 2) {
    localMax = __hmax2(__shfl_xor_sync(0xffffffffu, localMax, 1), localMax);
  }
  // Get the final absolute maximum values.
  float vecMax = float(__hmax(localMax.x, localMax.y));

  // 8 bits representation of the SF.
  float SFValue;
  uint8_t fp8SFVal;

  // NVFP4 path: scale = max / 6.0, stored as E4M3.
  SFValue = SFScaleVal * (vecMax * reciprocal_approximate_ftz(6.0f));
  // Here SFValue is always positive, so E4M3 is the same as UE4M3.
  __nv_fp8_e4m3 tmp = __nv_fp8_e4m3(SFValue);
  reinterpret_cast<__nv_fp8_e4m3&>(fp8SFVal) = tmp;
  // Convert back to fp32.
  SFValue = float(tmp);

  // Write the SF to global memory (STG.8).
  if (SFout) *SFout = fp8SFVal;

  // Get the output scale: use fast approximate reciprocal.
  float outputScale =
      SFValue != 0.0f
          ? reciprocal_approximate_ftz(SFValue *
                                       reciprocal_approximate_ftz(SFScaleVal))
          : 0.0f;

  // Convert the input to float.
  float2 fp2Vals[CVT_FP4_ELTS_PER_THREAD / 2];

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; i++) {
    fp2Vals[i] = cast_to_float2(vec.elts[i]);
    fp2Vals[i].x *= outputScale;
    fp2Vals[i].y *= outputScale;
  }

  // Convert to e2m1 values.
  return fp32_vec8_to_e2m1(fp2Vals);
}

// silu in float32
__device__ __forceinline__ float silu(float x) {
  return __fdividef(x, (1.f + __expf(-x)));
}

__device__ __forceinline__ float2 silu2(float2 x) {
  return make_float2(silu(x.x), silu(x.y));
}

template <class Type>
__inline__ __device__ PackedVec<Type> compute_silu_mul(
    const PackedVec<Type>& x_vec, const PackedVec<Type>& y_vec) {
  PackedVec<Type> result;

#pragma unroll
  for (int i = 0; i < CVT_FP4_ELTS_PER_THREAD / 2; ++i) {
    // silu_mul in float32
    using packed_t = typename PackedTypeConverter<Type>::Type;
    float2 silu_vec = silu2(cast_to_float2(x_vec.elts[i]));
    float2 y_f2 = cast_to_float2(y_vec.elts[i]);
    result.elts[i] = cast_to_packed<packed_t>(
        make_float2(silu_vec.x * y_f2.x, silu_vec.y * y_f2.y));
  }
  return result;
}

}  // namespace muse_spark
