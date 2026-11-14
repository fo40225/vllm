# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""b12x NVFP4 MoE for a few tokens on SM120, run under CUDA graphs.

b12x fuses routing, both expert GEMMs and the activation into one kernel, which
beats the CUTLASS grouped GEMMs up to a handful of tokens. It runs here as a
top-1 MoE over one row per route: row ``t * top_k + k`` is token ``t`` for the
expert of its k-th route, the router weight is 1, and routes to experts of other
ranks carry the id -1, which the dynamic kernel skips at no cost (their output
rows are zero). The rows come out unweighted and route-major, the layout the
post-GEMM2 hook takes.

Every shape is static: one binding per padded token count, built before any CUDA
graph capture. A binding reads its inputs from static buffers that the caller
refills on each call. The buffers are shared by all layers, which run one after
another on one stream.

The engine is optional: ``b12x_unusable_reason`` says whether the installed b12x
can serve it, and callers that cannot set it up stay on their other engine.
"""

import functools
import importlib
import importlib.util
from dataclasses import dataclass

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Padded token counts 1..B12X_MAX_TOKENS get a binding.
B12X_MAX_TOKENS = 8
# The dynamic kernel routes small row counts directly and larger ones grouped.
# 32 rows are 4 padded tokens at top-8. The cutoff comes from the b12x gate probe
# of 2026-10-20 (b12x 1.5.1, 8x RTX PRO 6000 Blackwell Server, 188 SMs, EP8 decode
# shapes with hidden = intermediate = 4096); it was not re-measured on other SM120
# parts or b12x versions, where it may move.
_DIRECT_ROUTE_MAX_ROWS = 32


# What the engine takes from ``b12x.moe.fused_moe``.
_REQUIRED_SYMBOLS = (
    "plan_weights",
    "prepare_weights",
    "plan",
    "bind",
    "run",
    "Caps",
    "MoeDecodeConfig",
)
# Set once a layer could not set the engine up: the others would fail alike,
# each after a compile attempt.
_setup_failure: str | None = None


@functools.cache
def _probe_b12x() -> str | None:
    try:
        if importlib.util.find_spec("b12x") is None:
            return "b12x is not installed"
        fused_moe = importlib.import_module("b12x.moe.fused_moe")
        missing = [name for name in _REQUIRED_SYMBOLS if not hasattr(fused_moe, name)]
        if missing:
            return f"the installed b12x lacks {', '.join(missing)}"
        # The gate of vLLM's own b12x experts. Older releases have no such query.
        is_supported = getattr(fused_moe, "is_supported", None)
        if is_supported is not None and not is_supported():
            return "the installed b12x does not support this GPU"
    except Exception as e:
        # A version mismatch or a missing nvidia-cutlass-dsl surfaces at the
        # import as any exception, not only ImportError.
        return f"the installed b12x cannot be used ({type(e).__name__}: {e})"
    return None


def b12x_unusable_reason() -> str | None:
    """Why the b12x decode engine cannot run here, or None if it can."""
    return _setup_failure or _probe_b12x()


def mark_b12x_setup_failed(reason: str) -> None:
    global _setup_failure
    _setup_failure = reason


@dataclass
class _Buffers:
    """Static tensors of one binding shape, shared by every layer."""

    scratch: torch.Tensor
    hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    # Token of every row.
    token_index: torch.Tensor


_buffers: dict[tuple, _Buffers] = {}


def _get_buffers(
    key: tuple,
    device: torch.device,
    num_tokens: int,
    top_k: int,
    hidden: int,
    nbytes: int,
) -> _Buffers:
    buffers = _buffers.get(key)
    if buffers is None:
        rows = num_tokens * top_k
        buffers = _buffers[key] = _Buffers(
            scratch=torch.zeros(nbytes, dtype=torch.uint8, device=device),
            hidden_states=torch.zeros(
                rows, hidden, dtype=torch.bfloat16, device=device
            ),
            topk_ids=torch.full((rows, 1), -1, dtype=torch.int32, device=device),
            topk_weights=torch.ones(rows, 1, dtype=torch.float32, device=device),
            output=torch.zeros(rows, hidden, dtype=torch.bfloat16, device=device),
            token_index=torch.arange(num_tokens, device=device).repeat_interleave(
                top_k
            ),
        )
    assert buffers.scratch.numel() >= nbytes
    return buffers


class B12xDecodeMoE:
    """The b12x engine of one layer, over the weights the CUTLASS path reads.

    ``w1`` [E, 2n, K / 2] must be stored ``[up; gate]``, the order b12x calls
    ``w13`` and consumes as is: it neither copies nor modifies the weights, and
    ``w1_scale`` / ``w2_scale`` are the swizzled block scales. The global scales
    are b12x's: its GEMM alphas are ``w_global_scale / a_gscale``. The engine
    reads the weights where they are (see ``reads``); a caller whose weights
    move builds a new one.
    """

    def __init__(
        self,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        w1_global_scale: torch.Tensor,
        a1_gscale: torch.Tensor,
        w2_global_scale: torch.Tensor,
        a2_gscale: torch.Tensor,
        top_k: int,
    ) -> None:
        from b12x.moe import fused_moe
        from b12x.moe.fused_moe import MoeDecodeConfig

        self._fused_moe = fused_moe
        self._decode_config_cls = MoeDecodeConfig
        self.top_k = top_k
        num_experts, self.hidden = w2.shape[:2]
        self.intermediate = w2.shape[2] * 2
        self.device = w1.device
        weight_plan = fused_moe.plan_weights(
            quant_modes="nvfp4",
            source_format="modelopt_nvfp4",
            activation="silu",
            params_dtype=torch.bfloat16,
            num_experts=num_experts,
            hidden_size=self.hidden,
            intermediate_size=self.intermediate,
            w13_layout="w13",
        )
        self.experts = fused_moe.prepare_weights(
            plan=weight_plan,
            w1_fp4=w1,
            w1_blockscale=w1_scale,
            w2_fp4=w2,
            w2_blockscale=w2_scale,
            w1_global_scale=w1_global_scale.float().contiguous(),
            a1_gscale=a1_gscale.float().contiguous(),
            w2_global_scale=w2_global_scale.float().contiguous(),
            a2_gscale=a2_gscale.float().contiguous(),
            params_dtype=torch.bfloat16,
        )
        # A copy would double the w13 / w2 memory; w31 would flip the weights.
        if (
            self.experts.plan.w13_layout != "w13"
            or self.experts.w1_fp4.data_ptr() != w1.data_ptr()
            or self.experts.w1_blockscale.data_ptr() != w1_scale.data_ptr()
            or self.experts.w2_fp4.data_ptr() != w2.data_ptr()
            or self.experts.w2_blockscale.data_ptr() != w2_scale.data_ptr()
        ):
            raise RuntimeError("b12x did not take the MoE weights as they are.")
        self._sources = (
            w1.data_ptr(),
            w1_scale.data_ptr(),
            w2.data_ptr(),
            w2_scale.data_ptr(),
        )
        self._bindings = {n: self._bind(n) for n in range(1, B12X_MAX_TOKENS + 1)}
        self._warm_up()

    def reads(
        self,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> bool:
        """Whether the engine was prepared on exactly these tensors."""
        return self._sources == (
            w1.data_ptr(),
            w1_scale.data_ptr(),
            w2.data_ptr(),
            w2_scale.data_ptr(),
        )

    def _bind(self, num_tokens: int):
        fused_moe = self._fused_moe
        rows = num_tokens * self.top_k
        # The decode config is pinned: the default heuristic picks the micro
        # kernel for a few rows, which costs ~15 us per skipped row (the figure
        # of the same 2026-10-20 probe, see _DIRECT_ROUTE_MAX_ROWS).
        decode_config = self._decode_config_cls(
            backend="dynamic",
            route_planner="internal",
            max_active_clusters=None,
            dynamic_tile_m=16,
            dynamic_route_mode=(
                "direct" if rows <= _DIRECT_ROUTE_MAX_ROWS else "grouped"
            ),
        )
        plan = fused_moe.plan(
            fused_moe.Caps(
                max_tokens=rows,
                num_topk=1,
                device=self.device,
                weight_plan=self.experts.plan,
                core_token_counts=(rows,),
                route_num_experts=0,
                quant_mode="nvfp4",
                apply_router_weight_on_input=False,
                swiglu_limit=None,
                swiglu_alpha=None,
                swiglu_beta=None,
                frozen=True,
                decode_config=decode_config,
            )
        )
        (spec,) = plan.scratch_specs()
        assert spec.dtype == torch.uint8
        key = (
            self.device,
            num_tokens,
            self.top_k,
            self.experts.w1_fp4.shape[0],
            self.hidden,
            self.intermediate,
        )
        buffers = _get_buffers(
            key, self.device, num_tokens, self.top_k, self.hidden, int(spec.shape[0])
        )
        binding = fused_moe.bind(
            plan,
            scratch=buffers.scratch,
            a=buffers.hidden_states,
            experts=self.experts,
            topk_weights=buffers.topk_weights,
            topk_ids=buffers.topk_ids,
            output=buffers.output,
            input_scales_static=True,
            unit_scale_contract=False,
        )
        # The first run compiles the kernels and creates the weight views, which
        # must not happen during a CUDA graph capture.
        buffers.topk_ids.zero_()
        fused_moe.run(binding=binding)
        torch.accelerator.synchronize()
        return binding, buffers

    def _warm_up(self) -> None:
        """Run the glue of ``run`` once per shape and expert map kind.

        The kernels are compiled by ``_bind``; this launches the copies around
        them for the first time, so that no CUDA graph capture does.
        """
        expert_map = torch.zeros(1, dtype=torch.int32, device=self.device)
        for num_tokens in self._bindings:
            hidden_states = torch.zeros(
                num_tokens, self.hidden, dtype=torch.bfloat16, device=self.device
            )
            topk_ids = torch.zeros(
                num_tokens, self.top_k, dtype=torch.int32, device=self.device
            )
            for ids_map in (None, expert_map):
                self.run(hidden_states, topk_ids, ids_map)
        torch.accelerator.synchronize()

    def run(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> torch.Tensor:
        """Unweighted GEMM2 rows [num_tokens * top_k, hidden], route-major.

        Rows of routes to experts that are not local are zero. ``hidden_states``
        and ``topk_ids`` are contiguous; the ids are valid global expert ids and
        ``expert_map`` is the int32 global-to-local map (None: all are local).
        The rows belong to the binding and stay valid until the next call.
        """
        try:
            binding, buffers = self._bindings[hidden_states.shape[0]]
        except KeyError:
            raise RuntimeError(
                f"No b12x binding for {hidden_states.shape[0]} tokens; they are "
                f"built for 1..{B12X_MAX_TOKENS}."
            ) from None
        if expert_map is None:
            buffers.topk_ids.copy_(topk_ids.view(-1, 1))
        else:
            torch.index_select(
                expert_map, 0, topk_ids.view(-1), out=buffers.topk_ids.view(-1)
            )
        torch.index_select(
            hidden_states, 0, buffers.token_index, out=buffers.hidden_states
        )
        self._fused_moe.run(binding=binding)
        return buffers.output
