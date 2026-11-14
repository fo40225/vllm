// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// torch library registration for the `_muse_C` extension. To add a kernel:
// declare it in muse_c_ops.h, add its .cu file to _SOURCES in
// vllm/models/muse_spark/nvidia/ops/muse_c.py and register it here.

#include <torch/library.h>

#include "muse_c_ops.h"

TORCH_LIBRARY(_muse_C, m) {
  // [up | gate] variant of silu_and_mul_scaled_fp4_experts_quant.
  m.def(
      "mul_and_silu_scaled_fp4_experts_quant(Tensor! output, Tensor! "
      "output_scale, Tensor input, Tensor input_global_scale, "
      "Tensor input_offset_by_experts, "
      "Tensor output_scale_offset_by_experts) -> ()");

  // Exact GEMM1 of the CUTLASS grouped-GEMM path (separate gate / up alphas).
  m.def(
      "silu_and_mul_split_scaled_fp4_experts_quant(Tensor! output, Tensor! "
      "output_scale, Tensor input, Tensor input_global_scale, "
      "Tensor input_offset_by_experts, "
      "Tensor output_scale_offset_by_experts, bool gate_first) -> ()");
  m.def(
      "split_gemm1_prepare(Tensor a_fp4, Tensor expert_offsets, "
      "Tensor blockscale_offsets, Tensor problem_sizes, Tensor! a_fp4_x2, "
      "Tensor! expert_offsets_x2, Tensor! blockscale_offsets_x2, "
      "Tensor! problem_sizes_x2) -> ()");

  // Stock scaled_fp4_experts_quant of the rows input[row_map[r]].
  m.def(
      "gather_scaled_fp4_experts_quant(Tensor! output, Tensor! output_scale, "
      "Tensor input, Tensor row_map, Tensor input_global_scale, "
      "Tensor input_offset_by_experts, "
      "Tensor output_scale_offset_by_experts) -> ()");

  // Routing data of the CUTLASS grouped GEMMs of an EP rank.
  m.def(
      "moe_route_local(Tensor topk_ids, Tensor? expert_map, "
      "int num_local_experts, int n, int k, Tensor! expert_offsets, "
      "Tensor! blockscale_offsets, Tensor! problem_sizes1, "
      "Tensor! problem_sizes2, Tensor! a_map, Tensor! c_map) -> ()");

  // NVFP4 (W4A4) MoE experts for a few tokens (decode), see muse_c_ops.h.
  m.def(
      "nvfp4_decode_moe(Tensor hidden, Tensor topk_ids, Tensor? expert_map, "
      "Tensor w13, Tensor sf13, Tensor w2, Tensor sf2, Tensor prm, "
      "Tensor gs, bool up_first, Tensor! rows, Tensor! perm_idx, "
      "Tensor! hq_data, Tensor! hq_sf, Tensor! plan, Tensor! ctl, "
      "Tensor? dbg_acc=None, int flags=0, Tensor? xq_data=None, "
      "Tensor? xq_sf=None) -> ()");
  m.def(
      "nvfp4_decode_act_quant(Tensor hidden, Tensor gs, Tensor! data, "
      "Tensor! sf) -> ()");

  // Muse Spark attention ops of vllm-project/vllm#62132; the schemas are those
  // of the stock torch.ops._C ops.
  m.def(
      "muse_spark_mul_tanh_soft_cap(Tensor! out, Tensor input, "
      "float multiplier, float soft_cap_temp) -> ()");
  m.def(
      "fused_muse_spark_qknorm_rope_kv_insert("
      "Tensor! qkv, Tensor positions, int num_heads, int num_kv_heads, "
      "int head_dim, float eps, Tensor? cos_sin_cache, int rotary_dim, "
      "Tensor? slot_mapping, Tensor!? kv_cache, int block_size, "
      "str kv_cache_dtype, Tensor? k_scale=None, Tensor? v_scale=None, "
      "Tensor!? q_fp8_out=None, Tensor? q_scale=None) "
      "-> ()");
}

TORCH_LIBRARY_IMPL(_muse_C, CUDA, m) {
  m.impl("mul_and_silu_scaled_fp4_experts_quant",
         &muse_spark::mul_and_silu_scaled_fp4_experts_quant);
  m.impl("silu_and_mul_split_scaled_fp4_experts_quant",
         &muse_spark::silu_and_mul_split_scaled_fp4_experts_quant);
  m.impl("split_gemm1_prepare", &muse_spark::split_gemm1_prepare);
  m.impl("gather_scaled_fp4_experts_quant",
         &muse_spark::gather_scaled_fp4_experts_quant);
  m.impl("moe_route_local", &muse_spark::moe_route_local);
  m.impl("nvfp4_decode_moe", &muse_spark::nvfp4_decode_moe);
  m.impl("nvfp4_decode_act_quant", &muse_spark::nvfp4_decode_act_quant);
  m.impl("muse_spark_mul_tanh_soft_cap",
         &muse_spark::muse_spark_mul_tanh_soft_cap);
  m.impl("fused_muse_spark_qknorm_rope_kv_insert",
         &muse_spark::fused_muse_spark_qknorm_rope_kv_insert);
}
