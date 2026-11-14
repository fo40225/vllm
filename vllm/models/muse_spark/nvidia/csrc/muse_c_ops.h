// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Declarations of the kernels exposed through the `_muse_C` torch library.
// Register each one in muse_c_bindings.cpp.

#pragma once

#include <ATen/core/Tensor.h>

#include <optional>
#include <string>

namespace muse_spark {

// Fused mul-and-silu + per-expert NVFP4 quantization. The input halves are
// [up | gate], so the result is quant(up * silu(gate)). This is the stock
// silu_and_mul_scaled_fp4_experts_quant with the two halves exchanged.
void mul_and_silu_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts);

// Muse Spark attention-side ops, ported from vllm-project/vllm#62132 (the
// stock `torch.ops._C` versions of the same ops). Schemas are identical.

// out = soft_cap_temp * tanh(input * multiplier / soft_cap_temp), in fp32.
// input is float32, float16 or bfloat16; out is float32 with the same shape.
void muse_spark_mul_tanh_soft_cap(at::Tensor& out, at::Tensor const& input,
                                  double multiplier, double soft_cap_temp);

// In-place weightless QK RMSNorm + GPT-J RoPE over [q | k | v | o_gate] rows,
// optionally writing k/v into a paged [nb, nkv, bs, 2*head_dim] KV cache
// (bf16/fp16, or fp8 with k_scale/v_scale) and Q as scaled E4M3 into
// q_fp8_out.
void fused_muse_spark_qknorm_rope_kv_insert(
    at::Tensor& qkv, at::Tensor const& positions, int64_t num_heads,
    int64_t num_kv_heads, int64_t head_dim, double eps,
    std::optional<at::Tensor> const& cos_sin_cache, int64_t rotary_dim,
    std::optional<at::Tensor> const& slot_mapping,
    std::optional<at::Tensor> const& kv_cache, int64_t block_size,
    std::string const& kv_cache_dtype,
    std::optional<at::Tensor> const& k_scale,
    std::optional<at::Tensor> const& v_scale,
    std::optional<at::Tensor> const& q_fp8_out,
    std::optional<at::Tensor> const& q_scale);

}  // namespace muse_spark
