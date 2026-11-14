# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the MuseSpark1x MoE runner tests."""

from collections.abc import Callable

import torch
import torch.nn.functional as F

from vllm.config import (
    KernelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MuseMoERunner,
    MusePostExpertFinalize,
    MuseSpark1xRouter,
    MuseSparkRoutedExperts,
)
from vllm.models.muse_spark.nvidia.ops.expert_placement import (
    ExpertPlacementTable,
    install_expert_placement,
)

# route_scale must be 1.0 and scoring sigmoid for the routing to classify as
# MiniMax2, which is what lets the monolithic kernel run MuseSpark's router.
ROUTE_SCALE = 1.0
DIVISION_EPS = 1e-15


def random_placement_table(
    num_experts: int, ep_size: int, top_k: int, seed: int = 0
) -> ExpertPlacementTable:
    """A one-layer table that deals the experts to the ranks at random, ``num_experts
    // ep_size`` each."""
    per_rank = num_experts // ep_size
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_experts, generator=g).tolist()
    rank_of = [0] * num_experts
    for position, expert in enumerate(order):
        rank_of[expert] = position // per_rank
    return ExpertPlacementTable(
        snapshot="test",
        num_hidden_layers=1,
        num_experts=num_experts,
        top_k=top_k,
        ep_size=ep_size,
        experts_per_rank=per_rank,
        base_placement="linear",
        layers={0: tuple(rank_of)},
    )


def install_placement(table: ExpertPlacementTable) -> Callable[[MuseMoERunner], None]:
    """A ``make_runner`` ``before_load`` that gives the layer (layer 0) the placement
    of ``table`` ahead of its weights."""

    def install(runner: MuseMoERunner) -> None:
        install_expert_placement(runner.routed_experts, table, layer_idx=0)

    return install


def nvfp4_quant_config() -> ModelOptNvFp4Config:
    return ModelOptNvFp4Config(
        quant_method="NVFP4",
        is_checkpoint_nvfp4_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
        group_size=16,
    )


def make_runner(
    norm: torch.nn.Module,
    expert_bias: torch.Tensor,
    ckpt: dict[str, torch.Tensor],
    quant_config=None,
    *,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    moe_backend: str = "auto",
    parallel_config: ParallelConfig | None = None,
    before_load: Callable[[MuseMoERunner], None] | None = None,
):
    """Build a MuseSpark runner, load ``ckpt`` and finalize its weights.

    ``before_load`` gets the runner once it is built, ahead of the weights (where a
    model installs an expert placement).
    """
    num_experts = expert_bias.numel()
    vllm_config = VllmConfig(
        parallel_config=parallel_config or ParallelConfig(),
        kernel_config=KernelConfig(moe_backend=moe_backend),
    )
    with set_current_vllm_config(vllm_config):
        router = MuseSpark1xRouter(
            top_k=top_k,
            num_experts=num_experts,
            expert_bias=expert_bias,
            route_scale=ROUTE_SCALE,
            router_score_normalization="sigmoid",
            router_score_division_eps=DIVISION_EPS,
        )
        runner = FusedMoEFactory(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            renormalize=True,
            router=router,
            quant_config=quant_config,
            prefix="moe.experts",
            params_dtype=torch.bfloat16,
            scoring_func="sigmoid",
            e_score_correction_bias=expert_bias,
            routed_scaling_factor=ROUTE_SCALE,
            routed_experts_cls=MuseSparkRoutedExperts,
            runner_cls=MuseMoERunner,
            runner_args={"post_expert_norm": norm},
        )
    runner.cuda()
    routed = runner.routed_experts
    if before_load is not None:
        before_load(runner)
    loaded = list(routed.load_weights(list(ckpt.items())))
    assert len(loaded) == len(ckpt)
    routed.quant_method.process_weights_after_loading(routed)
    moe_kernel = routed.quant_method.moe_kernel
    assert moe_kernel is not None
    experts = moe_kernel.fused_experts
    assert experts.post_gemm2_hook is runner.post_expert_finalize
    assert isinstance(runner.post_expert_finalize, MusePostExpertFinalize)
    assert not runner.post_expert_finalize.pre_norm_all_reduce
    return router, runner, vllm_config


def run_moe(runner, vllm_config, x, logits):
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        return runner.forward(hidden_states=x, router_logits=logits)


def assert_cudagraph_replays(
    runner,
    vllm_config,
    x,
    logits,
    next_x,
    next_logits,
    *,
    max_rel_l2: float | None = None,
):
    """A captured forward replayed on new inputs matches an eager forward.

    Experts that are not bitwise deterministic pass ``max_rel_l2`` for a bound on
    the relative L2 error instead of the default elementwise tolerance.
    """
    with set_current_vllm_config(vllm_config), set_forward_context(None, vllm_config):
        # FlashInfer autotuning must happen before capture.
        for _ in range(3):
            runner.forward(hidden_states=x, router_logits=logits)
        torch.accelerator.synchronize()

        x_static, logits_static = x.clone(), logits.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_static = runner.forward(
                hidden_states=x_static, router_logits=logits_static
            )

        expected = runner.forward(
            hidden_states=next_x, router_logits=next_logits
        ).clone()
        x_static.copy_(next_x)
        logits_static.copy_(next_logits)
        graph.replay()
    if max_rel_l2 is None:
        torch.testing.assert_close(out_static, expected)
        return
    error = (out_static.float() - expected.float()).norm() / expected.float().norm()
    assert error <= max_rel_l2, f"replay differs from eager by {error:.4f} (rel L2)"


def swiglu_moe_ref(
    x: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    norm_fn: Callable[[torch.Tensor], torch.Tensor],
    act_quant: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """fp32 SwiGLU MoE with ``norm_fn`` applied to each BF16 expert output."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for e in range(w13.shape[0]):
        token_idx, slot = (topk_ids == e).nonzero(as_tuple=True)
        if token_idx.numel() == 0:
            continue
        gate, up = (x[token_idx].float() @ w13[e].float().t()).chunk(2, dim=-1)
        inter = F.silu(gate) * up
        if act_quant is not None:
            inter = act_quant(inter)
        y = norm_fn((inter @ w2[e].float().t()).to(torch.bfloat16)).float()
        out.index_add_(0, token_idx, y * topk_weights[token_idx, slot, None])
    return out
