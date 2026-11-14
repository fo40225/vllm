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
}

TORCH_LIBRARY_IMPL(_muse_C, CUDA, m) {
  m.impl("mul_and_silu_scaled_fp4_experts_quant",
         &muse_spark::mul_and_silu_scaled_fp4_experts_quant);
}
