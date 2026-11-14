# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage for MuseSpark's native-FP4 (NVFP4 W4A4) MoE experts."""

import math

import pytest
import torch
import torch.nn.functional as F

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.muse_spark_utils import (
    assert_cudagraph_replays,
    make_runner,
    nvfp4_quant_config,
    run_moe,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale
from vllm.models.muse_spark.nvidia.muse_moe_runner import MusePostExpertFinalize
from vllm.models.muse_spark.nvidia.sm120_nvfp4_moe import (
    MuseSparkSm120NvFp4Experts,
    fold_sm120_nvfp4_scales,
    make_sm120_nvfp4_quant_config,
)
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

# Both sizes are multiples of 128, the grouped GEMM's tile.
E, N, K = 16, 256, 512
FP4_MAX = 6.0
FP8_MAX = 448.0
# RMS of a uniformly random e2m1 code.
NIB_RMS = 2.9269
# Activation scales are calibrated with headroom factors that are deliberately
# not round: with a round one, bf16 inputs land on exact round-to-nearest-even
# ties of the e2m1 grid, where the kernel's approximate reciprocal and an exact
# emulation legitimately disagree.
HEADROOM_A13 = 1.0937
HEADROOM_A2 = 1.2371
# Against the reference that quantizes at the same points as the kernel...
STAGE_FAITHFUL_REL_L2 = 1e-2
# ...and against bf16 activations, which only bounds the FP4 quantization error.
BF16_ACTIVATION_REL_L2 = 0.25

_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@pytest.fixture(autouse=True)
def poisoned_rows(monkeypatch):
    """Make every GEMM2 row that no route refers to NaN, so that a stale row
    leaking into the output cannot go unnoticed."""
    apply_cutlass = MuseSparkSm120NvFp4Experts._apply_cutlass

    def poisoned(self, *args):
        unfinalized = apply_cutlass(self, *args)
        rows = unfinalized.gemm2_permuted
        idx = unfinalized.expanded_idx_to_permuted_idx.flatten().long()
        num_rows = rows.shape[0]
        # Routes without a row (-1) are sent to a spare slot.
        used = torch.zeros(num_rows + 1, dtype=torch.bool, device=rows.device)
        used.scatter_(0, torch.where(idx >= 0, idx, num_rows), True)
        rows.masked_fill_(~used[:num_rows].unsqueeze(1), float("nan"))
        return unfinalized

    monkeypatch.setattr(MuseSparkSm120NvFp4Experts, "_apply_cutlass", poisoned)


@pytest.fixture
def row_layouts(monkeypatch) -> list[bool]:
    """``route_major`` of every ``UnfinalizedMoEOutput`` the finalizer receives;
    the torch fallback of the finalizer is not allowed."""
    seen: list[bool] = []
    finalize = MusePostExpertFinalize.__call__

    def spy(self, unfinalized, out):
        seen.append(unfinalized.route_major)
        finalize(self, unfinalized, out)

    def no_fallback(self, unfinalized, out):
        raise AssertionError("fell back to the native finalize")

    monkeypatch.setattr(MusePostExpertFinalize, "__call__", spy)
    monkeypatch.setattr(MusePostExpertFinalize, "_finalize_native", no_fallback)
    return seen


# ------------------------------------------------------------------------------
# NVFP4 helpers: dequantization of packed weights, exact fake-quant of activations
# ------------------------------------------------------------------------------
def _e2m1_lut(device) -> torch.Tensor:
    values = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=device)
    return torch.cat([values, -values])


def _dequant_fp4(packed, block_scale, scale_2: float) -> torch.Tensor:
    """packed [R, C/2] uint8 (low nibble = even column), block_scale [R, C/16]
    e4m3, per-tensor scale_2 -> fp32 [R, C]."""
    rows, packed_cols = packed.shape
    lut = _e2m1_lut(packed.device)
    low = lut[(packed & 0xF).long()]
    high = lut[(packed >> 4).long()]
    w = torch.stack([low, high], dim=-1).view(rows, packed_cols * 2 // 16, 16)
    w = w * (block_scale.float().unsqueeze(-1) * scale_2)
    return w.view(rows, packed_cols * 2)


def _round_to_e2m1(q: torch.Tensor) -> torch.Tensor:
    """Round to the nearest e2m1 value, ties to even (as cvt.rn does)."""
    magnitude = q.abs()
    code = (
        (magnitude > 0.25).to(torch.int8)
        + (magnitude >= 0.75).to(torch.int8)
        + (magnitude > 1.25).to(torch.int8)
        + (magnitude >= 1.75).to(torch.int8)
        + (magnitude > 2.5).to(torch.int8)
        + (magnitude >= 3.5).to(torch.int8)
        + (magnitude > 5.0).to(torch.int8)
    )
    rounded = _e2m1_lut(q.device)[code.long()]
    return torch.where(q < 0, -rounded, rounded)


def _fake_quant_nvfp4(x: torch.Tensor, global_scale: float) -> torch.Tensor:
    """Quantize and dequantize fp32 ``x`` [R, C] like the NVFP4 kernels do.

    The block scale is ``e4m3(amax16 * global_scale / 6)``, saturating at 448.
    """
    rows, cols = x.shape
    blocks = x.view(rows, cols // 16, 16)
    amax = blocks.abs().amax(dim=-1)
    block_scale = (amax * (global_scale / FP4_MAX)).clamp(max=FP8_MAX)
    block_scale = block_scale.to(torch.float8_e4m3fn).float()
    inv = torch.where(
        block_scale > 0,
        global_scale / block_scale.clamp_min(1e-30),
        torch.zeros_like(block_scale),
    )
    q = _round_to_e2m1(blocks * inv.unsqueeze(-1))
    return (q * (block_scale / global_scale).unsqueeze(-1)).view(rows, cols)


# ------------------------------------------------------------------------------
# A random NVFP4 MoE layer, calibrated like a checkpoint
# ------------------------------------------------------------------------------
class _Problem:
    """Checkpoint-layout tensors; ``s13[:, 0]`` / ``s13[:, 1]`` are the gate / up
    global weight scales and differ."""

    def __init__(self, seed: int = 0):
        device = "cuda"
        g = torch.Generator(device=device).manual_seed(seed)

        def packed(*shape):
            return torch.randint(
                0, 256, shape, dtype=torch.uint8, device=device, generator=g
            )

        def block_scales(*shape):
            noise = torch.randn(*shape, device=device, generator=g)
            ratio = torch.exp(-noise.abs() * 0.6)
            return (FP8_MAX * ratio).clamp(min=1.0).to(torch.float8_e4m3fn)

        def rms(t):
            return t.float().pow(2).mean(dim=(1, 2)).sqrt()

        def spread():
            return torch.rand(E, device=device, generator=g) * 0.7 + 0.7

        self.w13 = packed(E, 2 * N, K // 2)
        self.w13_sf = block_scales(E, 2 * N, K // 16)
        self.w2 = packed(E, K, N // 2)
        self.w2_sf = block_scales(E, K, N // 16)
        # Global scales that make the pre-activations O(1) for unit inputs.
        s1 = spread() / (K**0.5 * NIB_RMS * rms(self.w13_sf[:, :N]))
        s3_over_s1 = torch.rand(E, device=device, generator=g) * 3.5 + 0.5
        self.s13 = torch.stack([s1, s1 * s3_over_s1], dim=1)
        self.s2 = spread() / (N**0.5 * NIB_RMS * rms(self.w2_sf))
        self._dense: dict[int, tuple[torch.Tensor, ...]] = {}

        # Input scales, calibrated on activations of the bf16 flow.
        u = torch.randn(512, K, device=device, generator=g).to(torch.bfloat16)
        a13 = HEADROOM_A13 * u.float().abs().max().item() / (FP8_MAX * FP4_MAX)
        self.in13 = a13 * (0.35 + 0.65 * torch.rand(E, 2, device=device, generator=g))
        self.in13[0, 0] = a13
        amax_h = torch.stack(
            [self._hidden(e, u.float(), folded=False).abs().max() for e in range(E)]
        )
        self.in2 = HEADROOM_A2 * amax_h / (FP8_MAX * FP4_MAX)

    def checkpoint(self) -> dict[str, torch.Tensor]:
        return {
            "w13_weight": self.w13,
            "w13_weight_scale": self.w13_sf,
            "w13_weight_scale_2": self.s13,
            "w13_input_scale": self.in13,
            "w2_weight": self.w2,
            "w2_weight_scale": self.w2_sf,
            "w2_weight_scale_2": self.s2,
            "w2_input_scale": self.in2,
        }

    def dense(self, e: int) -> tuple[torch.Tensor, ...]:
        """fp32 (gate, up with s1, up with s3, down) of expert ``e``."""
        if e not in self._dense:
            s1, s3 = self.s13[e].tolist()
            gate_rows, up_rows = slice(0, N), slice(N, 2 * N)
            self._dense[e] = (
                _dequant_fp4(self.w13[e, gate_rows], self.w13_sf[e, gate_rows], s1),
                _dequant_fp4(self.w13[e, up_rows], self.w13_sf[e, up_rows], s1),
                _dequant_fp4(self.w13[e, up_rows], self.w13_sf[e, up_rows], s3),
                _dequant_fp4(self.w2[e], self.w2_sf[e], self.s2[e].item()),
            )
        return self._dense[e]

    def _hidden(self, e: int, x: torch.Tensor, folded: bool) -> torch.Tensor:
        """bf16-rounded ``silu(gate) * up`` of expert ``e``; with ``folded`` the
        up half carries s1 instead of s3, as the grouped GEMM computes it."""
        gate_w, up_w_s1, up_w_s3, _ = self.dense(e)
        gate = (x @ gate_w.T).bfloat16().float()
        up = (x @ (up_w_s1 if folded else up_w_s3).T).bfloat16().float()
        return (F.silu(gate) * up).bfloat16().float()


def _reference_rows(
    problem: _Problem,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    valid: torch.Tensor,
    *,
    a1_gscale: float | None,
    a2_gscale: torch.Tensor | None,
) -> torch.Tensor:
    """Unweighted GEMM2 rows [T, top_k, K] in fp32 (rounded to bf16 like the
    kernel's), zero for the routes that are not local.

    Without scales the activations stay bf16. With them the activations are
    quantized where the kernel quantizes them, in the folded domain: GEMM1 uses
    the gate scale for both halves, so ``up`` carries s1 instead of s3, and the
    intermediate is quantized with ``a2_gscale = (s3 / s1) / a2`` per expert.
    GEMM2 then needs the s3 / s1 back.
    """
    folded = a1_gscale is not None
    assert folded == (a2_gscale is not None)
    num_tokens, top_k = topk_ids.shape
    x = x.float()
    if folded:
        x = _fake_quant_nvfp4(x, a1_gscale)
    rows = torch.zeros(num_tokens, top_k, K, dtype=torch.float32, device=x.device)
    for e in torch.unique(topk_ids[valid]).tolist():
        tokens, slots = ((topk_ids == e) & valid).nonzero(as_tuple=True)
        h = problem._hidden(e, x[tokens], folded)
        if folded:
            s1, s3 = problem.s13[e].tolist()
            h = _fake_quant_nvfp4(h, a2_gscale[e].item()) * (s3 / s1)
        rows[tokens, slots] = (h @ problem.dense(e)[3].T).bfloat16().float()
    return rows


def _finalize_reference(rows, valid, topk_weights, norm) -> torch.Tensor:
    """Post-expert norm per route, then the fp32 weighted sum over local routes."""
    num_tokens, top_k, _ = rows.shape
    y = norm.forward_native(rows.bfloat16().view(-1, K), out_dtype=torch.float32)
    y = y.view(num_tokens, top_k, K) * valid.unsqueeze(-1)
    return (y * topk_weights.unsqueeze(-1)).sum(dim=1)


def _rel_l2(out: torch.Tensor, ref: torch.Tensor) -> float:
    return ((out.float() - ref).norm() / ref.norm().clamp_min(1e-30)).item()


def _check_against_references(
    out, problem, x, topk_ids, topk_weights, norm, valid, a1_gscale, a2_gscale
) -> None:
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all(), "non-finite values leaked into the output"
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for scales, bound in (
            ((a1_gscale, a2_gscale), STAGE_FAITHFUL_REL_L2),
            ((None, None), BF16_ACTIVATION_REL_L2),
        ):
            rows = _reference_rows(
                problem,
                x,
                topk_ids,
                valid,
                a1_gscale=scales[0],
                a2_gscale=scales[1],
            )
            ref = _finalize_reference(rows, valid, topk_weights, norm)
            assert ref.abs().max() > 0
            error = _rel_l2(out, ref)
            assert error <= bound, (
                f"relative L2 error {error:.4f} > {bound} against the "
                f"{'stage-faithful' if scales[0] is not None else 'bf16'} reference"
            )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def _make_norm() -> MuseSpark1xRMSNormWithInputScale:
    norm = MuseSpark1xRMSNormWithInputScale(
        K,
        eps=1e-8,
        has_weight=True,
        is_post_norm=True,
        zero_centered_gamma=True,
    ).cuda()
    norm.weight.copy_(torch.linspace(-0.2, 0.2, K, device="cuda"))
    return norm


# ------------------------------------------------------------------------------
# Scales
# ------------------------------------------------------------------------------
def test_fold_keeps_the_gate_and_up_scales_apart() -> None:
    g = torch.Generator().manual_seed(0)
    num_experts = 5
    w13_scale_2 = torch.rand(num_experts, 2, generator=g, dtype=torch.float32) * 0.02
    w13_scale_2 += 0.005
    w13_input_scale = torch.rand(num_experts, 2, generator=g, dtype=torch.float32) + 0.5
    w2_scale_2 = torch.rand(num_experts, generator=g, dtype=torch.float32) * 0.02
    w2_input_scale = torch.rand(num_experts, generator=g, dtype=torch.float32) + 0.5
    scales = fold_sm120_nvfp4_scales(
        w13_scale_2, w13_input_scale, w2_scale_2, w2_input_scale
    )
    a13 = w13_input_scale.max().item()
    a2 = w2_input_scale.max().item()
    for tensor in scales:
        assert tensor.shape == (num_experts,) and tensor.dtype == torch.float32
        assert tensor.is_contiguous()
    for e in range(num_experts):
        s1, s3 = w13_scale_2[e].tolist()
        assert s1 != s3
        assert math.isclose(scales.a1_gscale[e].item(), 1 / a13, rel_tol=1e-6)
        assert math.isclose(scales.g1_alphas[e].item(), a13 * s1, rel_tol=1e-6)
        assert math.isclose(scales.a2_gscale[e].item(), s3 / s1 / a2, rel_tol=1e-6)
        assert math.isclose(
            scales.g2_alphas[e].item(), a2 * w2_scale_2[e].item(), rel_tol=1e-6
        )


# ------------------------------------------------------------------------------
# The layer: runner -> quant method -> experts -> finalizer
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("top_k", [4, 8])
@torch.inference_mode()
def test_sm120_layer_matches_the_references(
    dist_init, workspace_init, row_layouts, num_tokens, top_k
) -> None:
    problem = _Problem()
    checkpoint = problem.checkpoint()
    assert (problem.s13[:, 0] != problem.s13[:, 1]).all()
    norm = _make_norm()
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    router, runner, vllm_config = make_runner(
        norm,
        expert_bias,
        checkpoint,
        nvfp4_quant_config(),
        hidden_size=K,
        intermediate_size=N,
        top_k=top_k,
    )
    routed = runner.routed_experts
    experts = routed.quant_method.moe_kernel.fused_experts
    assert type(experts) is MuseSparkSm120NvFp4Experts

    # The checkpoint scales and the packed weights are not touched; only the
    # block scales are swizzled, and the kernel reads the swizzled parameters.
    torch.testing.assert_close(routed.w13_weight_scale_2.data, problem.s13)
    torch.testing.assert_close(routed.w2_weight_scale_2.data, problem.s2)
    assert torch.equal(routed.w13_weight.data, problem.w13)
    assert torch.equal(routed.w2_weight.data, problem.w2)
    assert experts.w1_scale.data_ptr() == routed.w13_weight_scale.data_ptr()
    assert experts.w2_scale.data_ptr() == routed.w2_weight_scale.data_ptr()
    a13 = problem.in13.max().item()
    a2 = problem.in2.max().item()
    s1, s3 = problem.s13[:, 0], problem.s13[:, 1]
    torch.testing.assert_close(
        experts.a1_gscale, torch.full_like(s1, 1 / a13), rtol=1e-6, atol=0
    )
    torch.testing.assert_close(experts.g1_alphas, a13 * s1, rtol=1e-6, atol=0)
    torch.testing.assert_close(experts.a2_gscale, s3 / s1 / a2, rtol=1e-6, atol=0)
    torch.testing.assert_close(experts.g2_alphas, a2 * problem.s2, rtol=1e-6, atol=0)

    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(num_tokens, K, device="cuda", generator=g).to(torch.bfloat16)
    router_logits = torch.randn(
        num_tokens, E, dtype=torch.float32, device="cuda", generator=g
    )
    out = run_moe(runner, vllm_config, x, router_logits)
    topk_weights, topk_ids = router._compute_routing(x, router_logits, None)
    _check_against_references(
        out,
        problem,
        x,
        topk_ids.long(),
        topk_weights,
        norm,
        torch.ones_like(topk_ids, dtype=torch.bool),
        experts.a1_gscale[0].item(),
        experts.a2_gscale,
    )
    # One device, one token layout: the permuted rows go straight to the norm.
    assert row_layouts == [False]

    assert_cudagraph_replays(
        runner,
        vllm_config,
        x,
        router_logits,
        torch.randn_like(x),
        torch.randn_like(router_logits),
    )


# ------------------------------------------------------------------------------
# The experts on their own, with an explicit parallel layout
# ------------------------------------------------------------------------------
def _moe_config(top_k: int, num_local_experts: int, tp_size: int = 1) -> FusedMoEConfig:
    ep_size = E // num_local_experts
    return FusedMoEConfig(
        num_experts=E,
        experts_per_token=top_k,
        hidden_dim=K,
        intermediate_size=N * tp_size,
        num_local_experts=num_local_experts,
        num_logical_experts=E,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=tp_size,
            pcp_size=1,
            dp_size=1,
            ep_size=ep_size,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=1,
            use_ep=ep_size > 1,
            all2all_backend="allgather_reducescatter",
            enable_eplb=False,
        ),
        activation=MoEActivation.SILU,
        device=torch.device("cuda"),
        routing_method=RoutingMethodType.TopK,
        in_dtype=torch.bfloat16,
        max_num_tokens=64,
    )


def _routes(
    num_tokens: int, top_k: int, remote: torch.Tensor | None, *, remote_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random top-k routes. The first ``remote_tokens`` tokens route only to the
    experts in ``remote`` (a global-id tensor), if given."""
    g = torch.Generator(device="cuda").manual_seed(2)
    scores = torch.rand(num_tokens, E, device="cuda", generator=g)
    if remote is not None:
        mask = torch.ones(E, dtype=torch.bool, device="cuda")
        mask[remote] = False
        scores[:remote_tokens, mask] = -1.0
    topk_ids = scores.topk(top_k, dim=1).indices.to(torch.int32)
    weights = torch.sigmoid(
        torch.randn(num_tokens, top_k, device="cuda", generator=g) * 1.5
    )
    return topk_ids.contiguous(), (weights / weights.sum(1, keepdim=True)).contiguous()


def _run_experts(
    problem: _Problem,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    norm: torch.nn.Module,
    local_experts: torch.Tensor,
    *,
    tp_size: int = 1,
):
    """Runs the experts of one rank holding ``local_experts`` (global ids).

    Returns the output, the expert map (None when every expert is local) and the
    kernel's per-expert intermediate scales indexed by global expert id.
    """
    num_local = local_experts.numel()
    top_k = topk_ids.shape[1]
    # The block scales are taken per local expert; the input scales are those
    # of all experts, which is what the layer loads.
    w13_scale, w2_scale, quant_config = make_sm120_nvfp4_quant_config(
        problem.w13_sf[local_experts].contiguous(),
        problem.w2_sf[local_experts].contiguous(),
        problem.s13[local_experts],
        problem.in13,
        problem.s2[local_experts],
        problem.in2,
    )
    moe_config = _moe_config(top_k, num_local, tp_size)
    kernel = mk.FusedMoEKernel(
        make_moe_prepare_and_finalize_no_dp_ep(use_monolithic=False),
        MuseSparkSm120NvFp4Experts(moe_config=moe_config, quant_config=quant_config),
    )
    kernel.fused_experts.post_gemm2_hook = MusePostExpertFinalize(norm, moe_config)
    expert_map = None
    if num_local < E:
        expert_map = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        expert_map[local_experts] = torch.arange(
            num_local, dtype=torch.int32, device="cuda"
        )
    out = kernel.apply(
        hidden_states=x,
        w1=problem.w13[local_experts].contiguous(),
        w2=problem.w2[local_experts].contiguous(),
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=E,
        expert_map=expert_map,
        apply_router_weight_on_input=False,
    )
    a2_gscale = torch.zeros(E, dtype=torch.float32, device="cuda")
    a2_gscale[local_experts] = quant_config.a2_gscale
    return out, expert_map, quant_config.a1_gscale[0].item(), a2_gscale


@pytest.mark.parametrize("top_k", [4, 8])
@torch.inference_mode()
def test_sm120_expert_parallel_skips_the_remote_routes(
    dist_init, workspace_init, row_layouts, top_k
) -> None:
    problem = _Problem()
    norm = _make_norm()
    # Half of the experts, not a contiguous range.
    local_experts = torch.tensor(
        [1, 2, 4, 7, 8, 11, 13, 14], dtype=torch.long, device="cuda"
    )
    remote_experts = torch.tensor(
        [0, 3, 5, 6, 9, 10, 12, 15], dtype=torch.long, device="cuda"
    )
    num_tokens = 16
    # Token 0 has no local route at all.
    topk_ids, topk_weights = _routes(
        num_tokens, top_k, remote_experts, remote_tokens=1
    )
    x = torch.randn(num_tokens, K, device="cuda").to(torch.bfloat16)
    out, expert_map, a1_gscale, a2_gscale = _run_experts(
        problem, x, topk_ids, topk_weights, norm, local_experts
    )
    valid = expert_map[topk_ids.long()] >= 0
    assert not valid[0].any() and valid[1:].any() and not valid.all()
    _check_against_references(
        out,
        problem,
        x,
        topk_ids.long(),
        topk_weights,
        norm,
        valid,
        a1_gscale,
        a2_gscale,
    )
    assert torch.count_nonzero(out[0]) == 0
    assert row_layouts == [False]


@torch.inference_mode()
def test_sm120_expert_parallel_without_local_routes(
    dist_init, workspace_init, row_layouts
) -> None:
    problem = _Problem()
    norm = _make_norm()
    local_experts = torch.arange(E // 2, dtype=torch.long, device="cuda")
    remote_experts = torch.arange(E // 2, E, dtype=torch.long, device="cuda")
    num_tokens, top_k = 8, 8
    topk_ids, topk_weights = _routes(
        num_tokens, top_k, remote_experts, remote_tokens=num_tokens
    )
    x = torch.randn(num_tokens, K, device="cuda").to(torch.bfloat16)
    out, expert_map, _, _ = _run_experts(
        problem, x, topk_ids, topk_weights, norm, local_experts
    )
    assert (expert_map[topk_ids.long()] < 0).all()
    assert out.shape == x.shape
    assert torch.count_nonzero(out) == 0 and torch.isfinite(out).all()
    assert row_layouts == [False]


@pytest.mark.parametrize("top_k", [4, 8])
@torch.inference_mode()
def test_sm120_tensor_parallel_hands_over_route_major_rows(
    dist_init, workspace_init, row_layouts, top_k
) -> None:
    """With TP the rows are all-reduced before the norm, so they must come in the
    same order on every rank. One process stands in for the ranks: the TP group
    has a single member and the all-reduce is the identity."""
    problem = _Problem()
    norm = _make_norm()
    all_experts = torch.arange(E, dtype=torch.long, device="cuda")
    num_tokens = 16
    topk_ids, topk_weights = _routes(num_tokens, top_k, None, remote_tokens=0)
    x = torch.randn(num_tokens, K, device="cuda").to(torch.bfloat16)
    out, expert_map, a1_gscale, a2_gscale = _run_experts(
        problem, x, topk_ids, topk_weights, norm, all_experts, tp_size=2
    )
    assert expert_map is None
    _check_against_references(
        out,
        problem,
        x,
        topk_ids.long(),
        topk_weights,
        norm,
        torch.ones_like(topk_ids, dtype=torch.bool),
        a1_gscale,
        a2_gscale,
    )
    assert row_layouts == [True]
