// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Declarations of the kernels exposed through the `_muse_C` torch library.
// Register each one in muse_c_bindings.cpp.

#pragma once

#include <ATen/core/Tensor.h>

namespace muse_spark {

// Fused mul-and-silu + per-expert NVFP4 quantization. The input halves are
// [up | gate], so the result is quant(up * silu(gate)). This is the stock
// silu_and_mul_scaled_fp4_experts_quant with the two halves exchanged.
void mul_and_silu_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts);

}  // namespace muse_spark
