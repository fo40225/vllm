// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Subset of vLLM csrc/cuda_vec_utils.cuh (v0.32.0): only the 128-bit packed
// vector types and conversions used by the NVFP4 experts quantization kernel.
// The ld/st helpers, the 256-bit path and the c10 type converters are dropped.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace muse_spark {

// PackedTypeConverter - map a CUDA scalar type to its packed pair type
//   half <-> half2,  __nv_bfloat16 <-> __nv_bfloat162
template <typename T>
struct PackedTypeConverter {
  static_assert(sizeof(T) == 0,
                "PackedTypeConverter is not specialized for this type.");
};

template <>
struct PackedTypeConverter<half> {
  using Type = half2;
};

template <>
struct PackedTypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

// PackedVec - 128-bit vector of packed element pairs.
//   Type is the CUDA scalar type (e.g. half, __nv_bfloat16).
template <class Type>
struct alignas(16) PackedVec {
  static constexpr int NUM_ELTS =
      16 / sizeof(typename PackedTypeConverter<Type>::Type);
  typename PackedTypeConverter<Type>::Type elts[NUM_ELTS];
};

template <typename packed_t>
__device__ __forceinline__ float2 cast_to_float2(const packed_t& val) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
    return __bfloat1622float2(val);
  } else if constexpr (std::is_same_v<packed_t, __half2>) {
    return __half22float2(val);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return float2(val);
  }
}

template <typename packed_t>
__device__ __forceinline__ packed_t cast_to_packed(const float2& val) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
    return __float22bfloat162_rn(val);
  } else if constexpr (std::is_same_v<packed_t, __half2>) {
    return __float22half2_rn(val);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return float2(val);
  }
}

}  // namespace muse_spark
