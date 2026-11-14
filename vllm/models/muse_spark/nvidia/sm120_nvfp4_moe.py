# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark1x native-FP4 (NVFP4 W4A4) MoE experts for SM120.

Two engines on the FP4 tensor cores: grouped CUTLASS GEMMs, and for a few tokens
inside CUDA graphs the fused b12x decode kernel (``ops/b12x_decode_moe.py``).
The experts stop after GEMM2 and hand the unweighted rows to the post-GEMM2 hook
(the post-expert norm), which also does the weighted top-k sum.

MuseSpark checkpoints carry separate global scales for the gate (w1) and up
(w3) halves of ``w13``, while the grouped GEMM takes one alpha per expert. The
up-side gain is folded into the intermediate quantization scale instead (see
``fold_sm120_nvfp4_scales``), which yields the same FP4 codes as two alphas.

The checkpoint's ``w13`` rows are ``[gate; up]``. A layer that runs the b12x
engine stores them as ``[up; gate]`` instead (see ``swap_gate_up_halves_``), the
order the b12x kernels consume without touching the weights, and its CUTLASS
path quantizes the intermediate with the fused ``_muse_C`` kernel that reads the
halves in that order. Every other layer keeps the checkpoint's order and the
stock kernel, so that the JIT-built extension (nvcc and ninja) is needed only
where b12x runs. ``MuseSparkSm120NvFp4Experts.up_gate_layout`` says which a
layer holds.
"""

from typing import NamedTuple

import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.config import CUDAGraphMode, get_current_vllm_config_or_none
from vllm.logger import init_logger
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
from vllm.models.muse_spark.nvidia.ops import muse_c
from vllm.models.muse_spark.nvidia.ops.b12x_decode_moe import (
    B12X_MAX_TOKENS,
    B12xDecodeMoE,
    b12x_unusable_reason,
    mark_b12x_setup_failed,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)

# The grouped GEMM tiles are 128 wide in N and K, and the block scales are
# swizzled in 128 x 4 tiles.
_SHAPE_ALIGNMENT = 128
# Bounds the temporary copy of one half while the gate / up halves are swapped.
_SWAP_CHUNK_BYTES = 32 << 20


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


def _swap_row_halves_(t: torch.Tensor) -> None:
    """Swap the first and second half of dim 1 of ``t`` [E, 2n, C] in place."""
    assert t.is_contiguous() and t.dim() == 3 and t.shape[1] % 2 == 0
    # A byte view also covers the fp8 block scales.
    rows = t.view(torch.uint8)
    half = rows.shape[1] // 2
    experts_per_chunk = max(1, _SWAP_CHUNK_BYTES // (half * rows.shape[2]))
    for start in range(0, rows.shape[0], experts_per_chunk):
        chunk = rows[start : start + experts_per_chunk]
        first = chunk[:, :half].clone()
        chunk[:, :half] = chunk[:, half:]
        chunk[:, half:] = first


def swap_gate_up_halves_(
    w13_weight: torch.Tensor, w13_weight_scale: torch.Tensor
) -> None:
    """Turn the checkpoint's ``[gate; up]`` w13 rows into ``[up; gate]``.

    Takes the packed weight [E, 2n, K / 2] and its LINEAR block scales
    [E, 2n, K / 16], both in place; the scales must be swapped before they are
    swizzled. The per-expert global scales are indexed by what a half is, not by
    where it sits, so they stay as loaded.
    """
    assert w13_weight.shape[:2] == w13_weight_scale.shape[:2]
    _swap_row_halves_(w13_weight)
    _swap_row_halves_(w13_weight_scale)


def make_sm120_nvfp4_quant_config(
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_weight_scale_2: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_weight_scale_2: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, FusedMoEQuantConfig]:
    """Weight-side prep shared by every SM120 NVFP4 engine.

    Expects ``w13_weight_scale`` in the order of the packed weights the engines
    will read, i.e. swapped by ``swap_gate_up_halves_`` if they are. Returns the
    swizzled w13 / w2 block scales and the quant config. Only the (small) block
    scales are copied: the checkpoint's global scales are left untouched. The
    caller replaces the linear block scales with the returned ones, which frees
    them.
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


def b12x_decode_engine_enabled(moe_backend: str) -> bool:
    """Whether a layer is built for the b12x decode engine, which serves padded
    batches of up to ``B12X_MAX_TOKENS`` tokens inside CUDA graphs.

    The engine needs the ``[up; gate]`` weight layout and with it the JIT-built
    ``_muse_C`` extension, so the choice is made at layer construction, from the
    config in scope. ``--moe-backend cutlass`` forces CUTLASS everywhere.
    Otherwise (``auto``) the engine needs CUDA graphs, since it only runs while
    one is captured, and batch-invariant mode excludes it, since its kernel is
    not bitwise deterministic. A host whose b12x or build toolchain cannot serve
    it keeps the checkpoint's layout and the stock kernels.
    """
    reason = _config_without_b12x(moe_backend)
    if reason is not None:
        logger.info_once(
            "MuseSpark SM120 MoE: the b12x decode engine is off (%s); the CUTLASS "
            "grouped GEMMs run every batch.",
            reason,
        )
        return False
    reason = b12x_unusable_reason() or muse_c.preflight()
    if reason is not None:
        logger.warning_once(
            "MuseSpark SM120 MoE: the b12x decode engine is off (%s); the CUTLASS "
            "grouped GEMMs run every batch.",
            reason,
        )
        return False
    return True


def _config_without_b12x(moe_backend: str) -> str | None:
    if moe_backend != "auto":
        return f"--moe-backend {moe_backend}"
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return "no vLLM config in scope"
    model_config = vllm_config.model_config
    if (
        vllm_config.compilation_config.cudagraph_mode in (None, CUDAGraphMode.NONE)
        or (model_config is not None and model_config.enforce_eager)
    ):
        return "CUDA graphs are off"
    if envs.VLLM_BATCH_INVARIANT:
        return "batch-invariant mode"
    return None


def _capturing() -> bool:
    """Whether the current CUDA stream is being captured into a graph."""
    return torch.cuda.is_current_stream_capturing()


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
        # Set by the owner before process_weights_after_loading: whether w1 is
        # stored [up; gate] (else [gate; up], as in the checkpoint), and whether
        # to run the b12x engine, which needs that layout. The engine is prepared
        # by the first forward pass outside of a capture (see ``_ensure_b12x``),
        # and a layer that cannot prepare it clears ``use_b12x_decode``.
        self.up_gate_layout = False
        self.use_b12x_decode = False
        self.b12x: B12xDecodeMoE | None = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # The layer tensors seen here are not always the ones ``apply`` receives:
        # a weight reload (layerwise reload, refit) runs this on temporaries and
        # then copies the result into the persistent tensors. An engine bound to
        # the temporaries would pin a second copy of all expert weights, so none
        # is prepared here, and one left from before the reload is dropped (its
        # global scales are stale as well). The first forward pass prepares it.
        self.b12x = None

    def _ensure_b12x(self, w1: torch.Tensor, w2: torch.Tensor, capturing: bool) -> None:
        """Keep the b12x engine on the tensors ``apply`` receives.

        An engine that is missing (after a load or a reload) or that reads other
        tensors is prepared now, which takes a few seconds, so it must not happen
        inside a CUDA graph capture, where it raises instead. Startup never gets
        there: vLLM runs eager forward passes through every layer before it
        captures anything,

        * ``Worker.determine_available_memory`` (``v1/worker/gpu_worker.py``)
          runs ``profile_run``, an eager dummy forward pass, with or without
          ``kv_cache_memory_bytes``, before ``profile_cudagraph_memory`` captures;
        * ``Worker.compile_or_warm_up_model`` runs eager dummy passes before
          ``capture_model``;
        * ``GPUModelRunner._warmup_and_capture`` runs
          ``cudagraph_num_of_warmups`` eager dummy passes before each captured
          one, and ``VllmConfig`` sets that to 1 whenever graphs are enabled
          (``config/vllm.py``).

        The global scales are those of the CUTLASS alphas: b12x derives its GEMM
        alphas as ``w_global_scale / a_gscale``. The engine is an optimization:
        if b12x cannot set it up, this layer keeps the CUTLASS grouped GEMMs and
        stops trying (``use_b12x_decode`` is cleared), and so do the layers after
        it.
        """
        assert self.w1_scale is not None and self.w2_scale is not None
        if self.b12x is not None and self.b12x.reads(
            w1, self.w1_scale, w2, self.w2_scale
        ):
            return
        if capturing:
            raise RuntimeError(
                "The b12x decode engine is not prepared on the MoE weights (it is "
                "prepared by the first forward pass after the weights are loaded "
                "or reloaded), and it cannot be prepared inside a CUDA graph "
                "capture. Run a forward pass outside of the capture first."
            )
        assert self.up_gate_layout, "b12x consumes the [up; gate] layout"
        assert self.g1_alphas is not None and self.g2_alphas is not None
        assert self.a1_gscale is not None and self.a2_gscale is not None
        stale = self.b12x is not None
        self.b12x = None
        reason = b12x_unusable_reason()
        if reason is None:
            if stale:
                logger.info_once(
                    "MuseSpark SM120 MoE: preparing the b12x decode engine again "
                    "on the reloaded weights."
                )
            else:
                logger.info_once(
                    "MuseSpark SM120 MoE: preparing the b12x decode engine at the "
                    "first forward pass."
                )
            try:
                self.b12x = B12xDecodeMoE(
                    w1,
                    self.w1_scale,
                    w2,
                    self.w2_scale,
                    w1_global_scale=self.g1_alphas * self.a1_gscale,
                    a1_gscale=self.a1_gscale,
                    w2_global_scale=self.g2_alphas * self.a2_gscale,
                    a2_gscale=self.a2_gscale,
                    top_k=self.moe_config.experts_per_token,
                )
            except Exception as e:
                logger.debug("b12x decode engine setup failed", exc_info=True)
                reason = f"setup failed: {type(e).__name__}: {e}"
                mark_b12x_setup_failed(reason)
        if reason is not None:
            self.use_b12x_decode = False
            logger.warning_once(
                "MuseSpark SM120 MoE: the b12x decode engine is unavailable (%s); "
                "the CUTLASS grouped GEMMs run every batch.",
                reason,
            )

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
        # b12x only runs while a CUDA graph is captured: a replay never executes
        # Python, and an eager call costs more host time than it saves.
        use_b12x = False
        if self.use_b12x_decode:
            capturing = _capturing()
            self._ensure_b12x(w1, w2, capturing)
            use_b12x = (
                self.b12x is not None
                and capturing
                and hidden_states.shape[0] <= B12X_MAX_TOKENS
            )
        if use_b12x:
            unfinalized = self._apply_b12x(
                hidden_states, topk_weights, topk_ids, expert_map
            )
        else:
            unfinalized = self._apply_cutlass(
                hidden_states, w1, w2, topk_weights, topk_ids, expert_map, workspace2
            )
        self._run_post_gemm2_hook(unfinalized, output)

    def _apply_b12x(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> UnfinalizedMoEOutput:
        """The same rows as ``_apply_cutlass``, in route-major order.

        The kernel reads the weights through the b12x engine, which is prepared
        on the ones ``apply`` receives.
        """
        assert self.b12x is not None
        logger.info_once(
            "MuseSpark SM120 MoE: running the b12x decode engine for up to %d "
            "padded tokens.",
            B12X_MAX_TOKENS,
        )
        topk_ids = topk_ids.contiguous()
        rows = self.b12x.run(hidden_states.contiguous(), topk_ids, expert_map)
        return route_major_moe_output(rows, topk_weights, topk_ids, expert_map)

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

        It reads the swizzled block scales and the folded scales from the quant
        config; ``w1`` is stored as ``up_gate_layout`` says.
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
        gemm1_out = _resize_cache(workspace2, (num_routes, 2 * intermediate))
        ops.cutlass_fp4_moe_mm(
            gemm1_out,
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
        if self.up_gate_layout:
            # GEMM1 yields [up; gate]: the fused kernel applies SiLU to the
            # second half, like the stock op does to the first.
            quantize_intermediate = muse_c.mul_and_silu_scaled_fp4_experts_quant
        else:
            quantize_intermediate = ops.silu_and_mul_scaled_fp4_experts_quant
        h_fp4, h_blockscale = quantize_intermediate(
            gemm1_out, self.a2_gscale, expert_offsets, blockscale_offsets, top_k
        )
        # `gemm1_out` is dead from here on.
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
