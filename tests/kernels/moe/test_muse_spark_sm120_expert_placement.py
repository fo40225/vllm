# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage of the static expert placement of MuseSpark's MoE layers.

A placement replaces the layer's per-rank ``expert_map`` (global expert id -> local
slot) before the weights load; the routing keeps the global ids. These tests run the
real runner and the experts of the SM120 paths on one GPU with a simulated EP layout:
for every simulated rank the layer is built as that rank of EP4, once on the linear
placement and once on a random balanced one, from the same checkpoint. They check that
every per-expert tensor (weights, block scales, global scales and everything derived
from them) lands in the slot the map gives its expert, that every route is computed by
the rank that owns its expert with the very same rows, and that the sum over the ranks
is the unplaced layer's output up to rounding. The 8-GPU variant (real EP, sequence
parallel, packed latent) is in ``test_muse_spark_sm120_sequence_parallel.py``; the muse
decode kernels and their prefetch under a scattered map are in
``test_muse_spark_sm120_decode_moe.py`` and ``test_muse_spark_expert_prefetch.py``.
"""

import dataclasses

import pytest
import torch

from tests.kernels.moe.muse_spark_utils import (
    assert_cudagraph_replays,
    install_placement,
    make_runner,
    nvfp4_quant_config,
    random_placement_table,
    run_moe,
)
from tests.kernels.moe.test_muse_spark_sm120_nvfp4_moe import (  # noqa: F401
    B12X_FEW_ROUTES_REL_L2,
    E,
    K,
    N,
    _Experts,
    _Problem,
    _make_norm,
    _routes,
    engine_calls,
    hook_rows,
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.models.muse_spark.nvidia.sm120_nvfp4_moe import (
    B12X_MAX_TOKENS,
    decode_kernel_params,
)
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

EP = 4
TOP_K = 8
# The sum over the ranks of the partial outputs, placed against linear: the experts a
# rank sums are others, so the fp32 sums group differently and each rank's partial is
# rounded to bf16 on its own.
SUM_REL_L2 = 1e-2


def _linear_partition(ep: int) -> list[torch.Tensor]:
    per_rank = E // ep
    return [
        torch.arange(r * per_rank, (r + 1) * per_rank, device="cuda")
        for r in range(ep)
    ]


def _placed_partition(ep: int, seed: int = 3) -> list[torch.Tensor]:
    table = random_placement_table(E, ep, TOP_K, seed=seed)
    return [
        torch.tensor(table.local_experts(0, r), device="cuda") for r in range(ep)
    ]


def _routes_on(rows, idx, num_tokens: int) -> torch.Tensor:
    """Unweighted GEMM2 rows of the routes of one rank [T, top_k, K], zero for the
    routes it does not hold."""
    valid = idx >= 0
    routes = rows[idx.clamp(min=0).long().view(-1)].float().view(num_tokens, TOP_K, -1)
    return torch.where(valid.unsqueeze(-1), routes, torch.zeros_like(routes))


@pytest.mark.parametrize("exact", [True, False], ids=["exact", "folded"])
@pytest.mark.parametrize("up_gate", [True, False])
@pytest.mark.parametrize("num_tokens", [16, 700])
@torch.inference_mode()
def test_placed_ranks_write_the_rows_of_the_linear_ranks(
    dist_init,
    workspace_init,
    hook_rows,  # noqa: F811
    num_tokens,
    up_gate,
    exact,
) -> None:
    """The CUTLASS grouped GEMMs (local rows, 16 tokens, and 700) on the experts of a
    rank under a random placement: every route is on one rank, and its row is bit for
    bit the row the linear placement's rank for that expert writes."""
    problem = _Problem()
    norm = _make_norm()
    topk_ids, topk_weights = _routes(num_tokens, TOP_K, None, remote_tokens=0)
    x = torch.randn(num_tokens, K, device="cuda").to(torch.bfloat16)
    rows: dict[str, torch.Tensor] = {}
    totals: dict[str, torch.Tensor] = {}
    for name, partition in (
        ("linear", _linear_partition(EP)),
        ("placed", _placed_partition(EP)),
    ):
        rows[name] = torch.zeros(num_tokens, TOP_K, K, device="cuda")
        totals[name] = torch.zeros(num_tokens, K, device="cuda")
        owners = torch.zeros(num_tokens, TOP_K, dtype=torch.int64, device="cuda")
        for local in partition:
            hook_rows.clear()
            experts = _Experts(
                problem, norm, local, TOP_K, up_gate=up_gate, exact=exact
            )
            out = experts.apply(x, topk_ids, topk_weights)
            rows[name] += _routes_on(*hook_rows[-1], num_tokens)
            totals[name] += out.float()
            owners += (experts.expert_map[topk_ids.long()] >= 0).long()
        assert (owners == 1).all()
    assert torch.equal(rows["placed"], rows["linear"])
    assert rows["placed"].abs().max() > 0
    diff = totals["placed"] - totals["linear"]
    assert diff.norm() / totals["linear"].norm() < SUM_REL_L2
    assert totals["linear"].abs().max() > 0
    # The placement really is another split of the experts.
    assert any(
        not torch.equal(a, b)
        for a, b in zip(_linear_partition(EP), _placed_partition(EP))
    )


# ------------------------------------------------------------------------------
# The real runner: every per-expert tensor lands in its slot
# ------------------------------------------------------------------------------
@pytest.fixture
def simulated_ep(monkeypatch):
    """Layers built while ``state['ep_size']`` is set are rank ``state['ep_rank']`` of
    that EP size (one process, one device: the final all-reduce is over one rank, so
    the output is the rank's partial sum)."""
    state: dict[str, int] = {}
    make = FusedMoEParallelConfig.make

    def simulated(*args, **kwargs):
        config = make(*args, **kwargs)
        if "ep_size" not in state:
            return config
        return dataclasses.replace(
            config, ep_size=state["ep_size"], ep_rank=state["ep_rank"], use_ep=True
        )

    monkeypatch.setattr(FusedMoEParallelConfig, "make", staticmethod(simulated))
    return state


def _checkpoint(kind: str):
    if kind == "bf16_triton":
        g = torch.Generator(device="cuda").manual_seed(1)
        ckpt = {
            "w13_weight": torch.randn(
                E, 2 * N, K, dtype=torch.bfloat16, device="cuda", generator=g
            )
            / 10,
            "w2_weight": torch.randn(
                E, K, N, dtype=torch.bfloat16, device="cuda", generator=g
            )
            / 10,
        }
        return ckpt, None, "auto"
    backend = "auto" if kind == "nvfp4_auto" else "cutlass"
    return _Problem().checkpoint(), nvfp4_quant_config(), backend


def _layer(kind: str, ckpt_spec, state, rank: int | None, table=None):
    """(runner, vllm_config) of ``kind``: the unplaced layer with every expert
    (``rank`` None), or rank ``rank`` of EP4, linear or placed by ``table``."""
    ckpt, quant_config, backend = ckpt_spec
    state.pop("ep_size", None)
    if rank is not None:
        state.update(ep_size=EP, ep_rank=rank)
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    _, runner, vllm_config = make_runner(
        _make_norm(),
        expert_bias,
        ckpt,
        quant_config,
        hidden_size=K,
        intermediate_size=N,
        top_k=TOP_K,
        moe_backend=backend,
        before_load=None if table is None else install_placement(table),
    )
    state.pop("ep_size", None)
    return runner, vllm_config


def _fused(runner):
    return runner.routed_experts.quant_method.moe_kernel.fused_experts


@pytest.mark.parametrize("kind", ["nvfp4_cutlass", "nvfp4_auto", "bf16_triton"])
@torch.inference_mode()
def test_layer_holds_every_expert_tensor_in_the_slot_of_its_expert(
    dist_init, workspace_init, simulated_ep, kind
) -> None:
    """Built from the checkpoint under the placement, rank r holds the experts of the
    table in ascending order: the packed weights, the swizzled block scales, the
    global scales and every tensor derived from them (the folded and exact alphas, the
    decode kernel's parameters) are those of the unplaced layer's experts of the rank,
    bit for bit. The input scales, indexed by global id, are untouched."""
    spec = _checkpoint(kind)
    table = random_placement_table(E, EP, TOP_K, seed=5)
    full, _ = _layer(kind, spec, simulated_ep, None)
    full_routed = full.routed_experts
    assert full_routed.expert_map_manager.expert_map is None
    seen: list[int] = []
    for rank in range(EP):
        runner, _ = _layer(kind, spec, simulated_ep, rank, table)
        routed = runner.routed_experts
        manager = routed.expert_map_manager
        assert manager.placement_strategy == "custom" and manager.ep_rank == rank
        local = table.local_experts(0, rank)
        seen += local
        assert manager.get_local_expert_ids() == list(local)
        idx = torch.tensor(local, device="cuda")
        assert torch.equal(routed.expert_map, table.expert_map(0, rank, "cuda"))
        # Weights (and for NVFP4 the swizzled block scales and the global scales).
        names = ["w13_weight", "w2_weight"]
        if kind != "bf16_triton":
            names += [
                "w13_weight_scale",
                "w2_weight_scale",
                "w13_weight_scale_2",
                "w2_weight_scale_2",
            ]
            for name in ("w13_input_scale", "w2_input_scale"):
                # Indexed by global id on every rank, then collapsed to their max.
                assert torch.equal(
                    getattr(routed, name).data, getattr(full_routed, name).data
                ), name
        for name in names:
            got, want = getattr(routed, name).data, getattr(full_routed, name).data
            assert torch.equal(
                got.view(torch.uint8), want[idx].view(torch.uint8)
            ), (kind, rank, name)
        if kind == "bf16_triton":
            continue
        experts, full_experts = _fused(runner), _fused(full)
        for name in ("g1_alphas", "g2_alphas", "a1_gscale", "a2_gscale"):
            assert torch.equal(
                getattr(experts, name), getattr(full_experts, name)[idx]
            ), (kind, rank, name)
        scales, full_scales = experts.decode_scales, full_experts.decode_scales
        for name in ("alpha_g", "alpha_u", "alpha2"):
            assert torch.equal(
                getattr(scales, name), getattr(full_scales, name)[idx]
            ), (kind, rank, name)
        assert torch.equal(scales.gs, full_scales.gs)
        prm, gs = decode_kernel_params(scales)
        full_prm, full_gs = decode_kernel_params(full_scales)
        assert torch.equal(prm, full_prm[idx]) and torch.equal(gs, full_gs)
        if experts.exact_scales is not None:
            alphas1 = full_experts.exact_scales.alphas1.view(E, 2)[idx].flatten()
            assert torch.equal(experts.exact_scales.alphas1, alphas1)
            assert torch.equal(
                experts.exact_scales.a2_gscale, full_experts.exact_scales.a2_gscale[idx]
            )
    # Every expert is held by exactly one rank.
    assert sorted(seen) == list(range(E))


@pytest.mark.parametrize("num_tokens", [1, 3, 37])
@pytest.mark.parametrize("kind", ["nvfp4_cutlass", "nvfp4_auto", "bf16_triton"])
@torch.inference_mode()
def test_placed_layer_sums_to_the_unplaced_layer(
    dist_init,
    workspace_init,
    simulated_ep,
    engine_calls,  # noqa: F811
    kind,
    num_tokens,
) -> None:
    """The runner of every simulated rank (router logits, selection by global id,
    experts, post-expert norm): the sum of the partial outputs over the ranks is the
    unplaced layer's output up to rounding, on the linear placement as well as on the
    placed one, and the placed layer's graph replays like its eager forward."""
    spec = _checkpoint(kind)
    table = random_placement_table(E, EP, TOP_K, seed=7)
    full, full_config = _layer(kind, spec, simulated_ep, None)
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    x = torch.randn(num_tokens, K, device="cuda", generator=g).to(torch.bfloat16)
    logits = torch.randn(num_tokens, E, dtype=torch.float32, device="cuda", generator=g)
    expected = run_moe(full, full_config, x, logits).float()
    assert expected.abs().max() > 0
    sums = {}
    for name, placement in (("linear", None), ("placed", table)):
        total = torch.zeros_like(expected)
        for rank in range(EP):
            runner, config = _layer(kind, spec, simulated_ep, rank, placement)
            out = run_moe(runner, config, x, logits)
            assert torch.isfinite(out).all()
            total += out.float()
            if name == "placed" and rank in (0, EP - 1):
                next_x = torch.randn_like(x)
                next_logits = torch.randn_like(logits)
                # b12x serves a few tokens inside the graph (it is not bitwise
                # repeatable); the other experts replay exactly.
                assert_cudagraph_replays(
                    runner,
                    config,
                    x,
                    logits,
                    next_x,
                    next_logits,
                    max_rel_l2=B12X_FEW_ROUTES_REL_L2 if kind == "nvfp4_auto" else None,
                )
        if name == "placed" and kind == "nvfp4_auto" and num_tokens <= B12X_MAX_TOKENS:
            # The graph of the placed layer ran the b12x engine, not only CUTLASS.
            assert "b12x" in engine_calls
        sums[name] = total
        error = (total - expected).norm() / expected.norm()
        assert error < SUM_REL_L2, (name, float(error))
    diff = (sums["placed"] - sums["linear"]).norm() / sums["linear"].norm()
    assert diff < SUM_REL_L2, float(diff)
