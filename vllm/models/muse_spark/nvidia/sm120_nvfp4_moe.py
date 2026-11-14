# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x native-FP4 (NVFP4 W4A4) MoE experts for SM120.

Grouped CUTLASS GEMMs on the FP4 tensor cores. The experts stop after GEMM2 and
hand the unweighted rows to the post-GEMM2 hook (the post-expert norm), which
also does the weighted top-k sum.

MuseSpark checkpoints carry separate global scales for the gate (w1) and up
(w3) halves of ``w13``, while the grouped GEMM takes one alpha per expert. The
up-side gain is folded into the intermediate quantization scale instead (see
``fold_sm120_nvfp4_scales``), which yields the same FP4 codes as two alphas.
"""

from typing import NamedTuple

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
    nvfp4_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.moe_output import (
    UnfinalizedMoEOutput,
    route_major_moe_output,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    remap_topk_to_local,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.models.muse_spark.common.moe import needs_pre_norm_all_reduce
from vllm.platforms import current_platform

# The grouped GEMM tiles are 128 wide in N and K, and the block scales are
# swizzled in 128 x 4 tiles.
_SHAPE_ALIGNMENT = 128


class Sm120NvFp4Scales(NamedTuple):
    """Per-expert fp32 ``[num_local_experts]`` scales of the grouped GEMMs."""

    a1_gscale: torch.Tensor
    g1_alphas: torch.Tensor
    a2_gscale: torch.Tensor
    g2_alphas: torch.Tensor


def fold_sm120_nvfp4_scales(
    w13_weight_scale_2: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> Sm120NvFp4Scales:
    """Single-alpha GEMM1 scales from the checkpoint's split w1/w3 scales.

    ``w13_weight_scale_2`` is ``[E, 2]`` (gate, up) and the two columns differ.
    The input scales collapse to their global max, as on the TRT-LLM path. With
    the gate alpha for both halves GEMM1 yields ``up * s1 / s3``; quantizing the
    intermediate with ``s3 / s1`` folded into its global scale undoes that:

        a1_gscale = 1 / a13            g1_alphas = a13 * s1
        a2_gscale = (s3 / s1) / a2     g2_alphas = a2 * s2
    """
    s1 = w13_weight_scale_2[:, 0].float()
    s3 = w13_weight_scale_2[:, 1].float()
    s2 = w2_weight_scale_2.float()
    a13 = w13_input_scale.float().max()
    a2 = w2_input_scale.float().max()
    return Sm120NvFp4Scales(
        a1_gscale=(1.0 / a13).expand_as(s1).contiguous(),
        g1_alphas=(a13 * s1).contiguous(),
        a2_gscale=((s3 / s1) / a2).contiguous(),
        g2_alphas=(a2 * s2).contiguous(),
    )


def make_sm120_nvfp4_quant_config(
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, FusedMoEQuantConfig]:
    """Weight-side prep shared by every SM120 NVFP4 engine.

    Returns the swizzled w13 / w2 block scales and the quant config. Only the
    (small) block scales are copied: the packed weights are used as loaded and
    the checkpoint's global scales are left untouched. The caller replaces the
    linear block scales with the returned ones, which frees them.
    """
    hidden = w2_weight_scale.shape[1]
    intermediate = w2_weight_scale.shape[2] * 16
    if hidden % _SHAPE_ALIGNMENT or intermediate % _SHAPE_ALIGNMENT:
        raise NotImplementedError(
            "MuseSpark SM120 NVFP4 MoE needs hidden size and per-rank "
            f"intermediate size divisible by {_SHAPE_ALIGNMENT}; got "
            f"hidden={hidden}, intermediate={intermediate}."
        )
    assert w13_weight_scale.shape[1] == 2 * intermediate
    scales = fold_sm120_nvfp4_scales(
        w13_weight_scale_2, w13_input_scale, w2_weight_scale_2, w2_input_scale
    )
    w13_swizzled = swizzle_blockscale(w13_weight_scale)
    w2_swizzled = swizzle_blockscale(w2_weight_scale)
    quant_config = nvfp4_moe_quant_config(
        g1_alphas=scales.g1_alphas,
        g2_alphas=scales.g2_alphas,
        a1_gscale=scales.a1_gscale,
        a2_gscale=scales.a2_gscale,
        w1_scale=w13_swizzled,
        w2_scale=w2_swizzled,
        is_scale_swizzled=True,
    )
    return w13_swizzled, w2_swizzled, quant_config


def _shuffle_rows_into(
    src: torch.Tensor, dst2src_map: torch.Tensor, out: torch.Tensor
) -> None:
    """``out[i] = src[map[i]]``; out-of-range entries (-1, sentinel) give zeros."""
    torch.ops._moe_C.shuffle_rows(src, dst2src_map, out)


class MuseSparkSm120NvFp4Experts(mk.FusedMoEExpertsModular):
    """NVFP4 W4A4 grouped-GEMM experts that stop after GEMM2.

    Both activations are quantized here, per expert, after the rows are
    permuted. The rows reach the post-GEMM2 hook in one of two layouts:

    * Permuted (EP, or TP of size 1): the GEMM2 output as is. Its row order
      inside an expert is not deterministic, which is fine when no collective
      runs on the rows.
    * Route-major (TP > 1): the partial sums are all-reduced before the norm,
      so every rank needs the same row order. The permutation is undone.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(moe_config, quant_config, max_num_tokens, num_dispatchers)
        if not quant_config.use_nvfp4_w4a4:
            raise NotImplementedError("Requires NVFP4 weights and activations.")
        if moe_config.in_dtype != torch.bfloat16:
            # The SM120 grouped GEMM only writes bfloat16.
            raise NotImplementedError("Requires bfloat16 activations.")
        if moe_config.is_lora_enabled:
            raise NotImplementedError("The post-GEMM2 hook does not support LoRA.")
        self.route_major = needs_pre_norm_all_reduce(moe_config)

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability_family(120)
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, kNvfp4Dynamic)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        # Fused SiLU * up + NVFP4 quantization.
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        # Tokens are replicated across the ranks. EP goes through expert_map and
        # the runner's late all-reduce; the gate-up scales are not parameters,
        # so EPLB could not move them.
        return not (
            moe_parallel_config.use_all2all_kernels
            or moe_parallel_config.use_passthrough_all2all
            or moe_parallel_config.enable_eplb
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        # The hook writes the finalized output.
        return TopKWeightAndReduceNoOP()

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        assert a1.dim() == 2 and topk_ids.size(0) == a1.size(0)
        # N is the intermediate size; w2 is [E, K, N / 2] with packed FP4.
        return w1.size(0), a1.size(0), w2.size(2) * 2, a1.size(1), topk_ids.size(1)

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # The rows the hook reads must not share storage with its output, which
        # is a view of workspace13 unless it aliases the layer output. So all
        # scratch is in workspace2: the shuffled input, GEMM1's output and
        # GEMM2's output reuse one region (each is dead before the next is
        # written), followed by the route-major rows under TP.
        scratch_width = max(2 * N, K) + (K if self.route_major else 0)
        return (0,), (M * topk, scratch_width), (M, K)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        if apply_router_weight_on_input:
            raise NotImplementedError(
                "post_gemm2_hook needs unweighted GEMM2 rows; it does not "
                "support router weights on the input."
            )
        if not self._uses_post_gemm2_hook(hidden_states.shape[0]):
            raise RuntimeError(
                "MuseSparkSm120NvFp4Experts has no top-k reduction of its own; "
                "attach the post-GEMM2 hook."
            )
        assert activation == MoEActivation.SILU
        unfinalized = self._apply_cutlass(
            hidden_states, w1, w2, topk_weights, topk_ids, expert_map, workspace2
        )
        self._run_post_gemm2_hook(unfinalized, output)

    def _apply_cutlass(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        workspace2: torch.Tensor,
    ) -> UnfinalizedMoEOutput:
        """Both grouped GEMMs; returns the unweighted, un-normed GEMM2 rows.

        The only engine-specific step: it reads the swizzled block scales and
        the folded scales from the quant config and nothing else of the weights.
        """
        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.g1_alphas is not None and self.g2_alphas is not None
        assert self.a1_gscale is not None and self.a2_gscale is not None
        num_tokens, hidden = hidden_states.shape
        top_k = topk_ids.size(1)
        num_routes = num_tokens * top_k
        num_experts = w1.size(0)
        intermediate = w2.size(2) * 2
        device = hidden_states.device

        # Non-local routes become -1 and are placed after every local row.
        topk_ids = topk_ids.contiguous()
        if expert_map is not None:
            local_ids = remap_topk_to_local(topk_ids, expert_map, out_dtype=torch.int32)
        else:
            local_ids = topk_ids.to(torch.int32)
        expert_offsets = torch.empty(num_experts + 1, dtype=torch.int32, device=device)
        blockscale_offsets = torch.empty_like(expert_offsets)
        problem_sizes1 = torch.empty((num_experts, 3), dtype=torch.int32, device=device)
        problem_sizes2 = torch.empty_like(problem_sizes1)
        # a_map: permuted row -> token (-1 for the unused tail); c_map: route
        # -> permuted row (num_routes for a non-local route).
        a_map = torch.empty(num_routes, dtype=torch.int32, device=device)
        c_map = torch.empty_like(a_map)
        ops.get_cutlass_moe_mm_data(
            local_ids,
            expert_offsets,
            problem_sizes1,
            problem_sizes2,
            a_map,
            c_map,
            num_experts,
            intermediate,
            hidden,
            blockscale_offsets,
            is_gated=True,
        )

        permuted_in = _resize_cache(workspace2, (num_routes, hidden))
        _shuffle_rows_into(hidden_states.contiguous(), a_map, permuted_in)
        # NOTE: the op allocates a fixed-size scale buffer on every call; it has
        # no preallocated variant in the Python wrapper.
        a_fp4, a_blockscale = ops.scaled_fp4_experts_quant(
            permuted_in, self.a1_gscale, expert_offsets, blockscale_offsets, top_k
        )
        # `permuted_in` is dead from here on.
        gate_up = _resize_cache(workspace2, (num_routes, 2 * intermediate))
        ops.cutlass_fp4_moe_mm(
            gate_up,
            a_fp4,
            w1,
            a_blockscale,
            self.w1_scale,
            self.g1_alphas,
            problem_sizes1,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
        del a_fp4, a_blockscale
        h_fp4, h_blockscale = ops.silu_and_mul_scaled_fp4_experts_quant(
            gate_up, self.a2_gscale, expert_offsets, blockscale_offsets, top_k
        )
        # `gate_up` is dead from here on.
        down = _resize_cache(workspace2, (num_routes, hidden))
        ops.cutlass_fp4_moe_mm(
            down,
            h_fp4,
            w2,
            h_blockscale,
            self.w2_scale,
            self.g2_alphas,
            problem_sizes2,
            expert_offsets[:-1],
            blockscale_offsets[:-1],
        )
        del h_fp4, h_blockscale

        # Rows past the local ones, and the rows of non-local routes, hold stale
        # scratch (possibly NaN); the consumers skip them through the -1 map.
        if self.route_major:
            scratch_width = max(2 * intermediate, hidden)
            rows = workspace2.flatten()[
                num_routes * scratch_width : num_routes * (scratch_width + hidden)
            ].view(num_routes, hidden)
            _shuffle_rows_into(down, c_map, rows)
            return route_major_moe_output(rows, topk_weights, topk_ids, expert_map)
        permuted_idx = (
            torch.where(c_map < num_routes, c_map, -1)
            .view(num_tokens, top_k)
            .to(torch.int32)
        )
        return UnfinalizedMoEOutput(
            gemm2_permuted=down,
            expert_weights=topk_weights,
            expanded_idx_to_permuted_idx=permuted_idx,
        )


def check_sm120_nvfp4_supported(moe_config: FusedMoEConfig) -> None:
    """Raise ``NotImplementedError`` if the experts cannot serve this layer."""
    supported, reason = MuseSparkSm120NvFp4Experts.is_supported_config(
        MuseSparkSm120NvFp4Experts,
        moe_config,
        kNvfp4Static,
        kNvfp4Dynamic,
        mk.FusedMoEActivationFormat.Standard,
    )
    if not supported:
        raise NotImplementedError(f"MuseSpark SM120 NVFP4 MoE: {reason}.")
