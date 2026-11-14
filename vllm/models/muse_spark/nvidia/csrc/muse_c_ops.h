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

// SiLU * up + per-expert NVFP4 quantization of the exact GEMM1's output, which
// holds the two halves as stacked matrices: input [2 * m_topk, k], the first
// half in rows [0, m_topk) and the second in rows [m_topk, 2 * m_topk). The
// first half is the gate if gate_first, else the up half. Quantizes
// up * silu(gate) to output [m_topk, k / 2], bit-identical to the stock
// silu_and_mul_scaled_fp4_experts_quant on the same values laid out
// [gate | up].
void silu_and_mul_split_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts, bool gate_first);

// Tables of the exact GEMM1 (muse_nvfp4_split_gemm1.cu): the grouped GEMM
// reads w13 [E, 2N, K / 2] as [2E, N, K / 2], group 2e + h being half h of
// expert e, on two copies of the activation rows. From the tables of the E
// experts (expert_offsets and blockscale_offsets [E + 1], problem_sizes [E, 3]
// = (M, 2N, K), as get_cutlass_moe_mm_data makes them) and the activations
// a_fp4 [R, K / 2], it writes expert_offsets_x2 and blockscale_offsets_x2
// [2E], problem_sizes_x2 [2E, 3] = (M, N, K), and the rows that experts own,
// twice, to a_fp4_x2 [2R, K / 2] (copy h at rows [h * R, (h + 1) * R)); the
// other rows are untouched.
void split_gemm1_prepare(at::Tensor const& a_fp4,
                         at::Tensor const& expert_offsets,
                         at::Tensor const& blockscale_offsets,
                         at::Tensor const& problem_sizes, at::Tensor& a_fp4_x2,
                         at::Tensor& expert_offsets_x2,
                         at::Tensor& blockscale_offsets_x2,
                         at::Tensor& problem_sizes_x2);


// The stock scaled_fp4_experts_quant of an activation matrix permuted by row_map,
// without the permuted copy: row r of the result is the quantization of row
// input[row_map[r]], with the global scale and scale-factor layout of the expert
// that owns row r. input [T, K] bf16/fp16 (K a multiple of 256), row_map [M]
// int32, output [M, K / 2] u8; only the first input_offset_by_experts[E] rows
// are written. It gives the bytes the stock op gives on the shuffled copy.
void gather_scaled_fp4_experts_quant(
    at::Tensor& output, at::Tensor& output_scale, at::Tensor const& input,
    at::Tensor const& row_map, at::Tensor const& input_global_scale,
    at::Tensor const& input_offset_by_experts,
    at::Tensor const& output_scale_offset_by_experts);

// Routing data of the CUTLASS grouped GEMMs of one expert-parallel rank
// (muse_moe_route.cu): what remap_topk_to_local and the stock
// get_cutlass_moe_mm_data give for the NVFP4 path, from the global ids.
//   topk_ids [T, top_k] int32 (global ids; -1 and ids outside the map are
//   remote), expert_map [E_global] int32 (local id or -1; null: the ids are
//   local already). Outputs, all int32: expert_offsets and blockscale_offsets
//   [E + 1] (rows of every expert, and the same padded to whole 128-row tiles),
//   problem_sizes1 [E, 3] = (m, 2n, k), problem_sizes2 [E, 3] = (m, k, n),
//   a_map [T * top_k] (token of every local row, -1 past them) and c_map
//   [T * top_k] (row of every route, -1 for a remote one).
void moe_route_local(at::Tensor const& topk_ids,
                     std::optional<at::Tensor> const& expert_map,
                     int64_t num_local_experts, int64_t n, int64_t k,
                     at::Tensor& expert_offsets, at::Tensor& blockscale_offsets,
                     at::Tensor& problem_sizes1, at::Tensor& problem_sizes2,
                     at::Tensor& a_map, at::Tensor& c_map);

// NVFP4 (W4A4) MoE experts for a few tokens on SM120 (muse_nvfp4_decode_moe.cu).
// One persistent launch: routing + activation quantization + GEMM1 + SiLU * up +
// NVFP4 requantization into hq, then GEMM2 into route-major unweighted bf16
// `rows` [T_PAD * 8, 4096] (rows of non-local routes are left untouched) and
// `perm_idx` [T_PAD * 8] (route index, or -1 for a non-local route).
//   hidden [T, 4096] bf16, topk_ids [T, 8] int32 (global ids), expert_map
//   [E_global] int32 (null: identity), w13 [E_local, 8192, 2048] u8 and w2
//   [E_local, 4096, 2048] u8 (packed e2m1), sf13 [E_local, 8192, 256] / sf2
//   [E_local, 4096, 256] e4m3 swizzled, prm [E_local, 4] fp32 = (alpha_g, alpha_u,
//   alpha2, 0), gs [2] fp32 = (1 / a13, 1 / a2). w13 rows are [up; gate] if
//   up_first, else [gate; up]. hq_data [64, 2048] u8, hq_sf [64, 256] u8 and
//   plan (int32, >= 708 entries) are scratch; ctl (int32, >= 66 entries) holds
//   the persistent kernel's counters: zero it once, every launch leaves it zero.
//   dbg_acc optionally receives the raw GEMM1 sums. flags: 1 = two launches
//   (plan B), 2 = zero the counters with a memset node first, 4 = zero the rows
//   of non-local routes, 8 = no programmatic dependent launch (of K2 with 1, of
//   the persistent kernel otherwise); bits 16.. override the grid size (0 = one
//   CTA per SM).
//   xq_data [T, 2048] u8 / xq_sf [T, 256] u8 (linear scales), when given, are the
//   activations quantized by the stock op (ops.scaled_fp4_quant with
//   is_sf_swizzled_layout=False and gs[0]) and replace the in-kernel quantization.
void nvfp4_decode_moe(at::Tensor const& hidden, at::Tensor const& topk_ids,
                      std::optional<at::Tensor> const& expert_map,
                      at::Tensor const& w13, at::Tensor const& sf13,
                      at::Tensor const& w2, at::Tensor const& sf2,
                      at::Tensor const& prm, at::Tensor const& gs, bool up_first,
                      at::Tensor& rows, at::Tensor& perm_idx,
                      at::Tensor& hq_data, at::Tensor& hq_sf, at::Tensor& plan,
                      at::Tensor& ctl, std::optional<at::Tensor> const& dbg_acc,
                      int64_t flags, std::optional<at::Tensor> const& xq_data,
                      std::optional<at::Tensor> const& xq_sf);

// The in-kernel activation quantization of nvfp4_decode_moe on its own, for
// tests: hidden [T, 4096] bf16 -> codes [T, 2048] u8 and linear e4m3 scales
// [T, 256] u8, gs[0] = 1 / a13.
void nvfp4_decode_act_quant(at::Tensor const& hidden, at::Tensor const& gs,
                            at::Tensor& data, at::Tensor& sf);

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
