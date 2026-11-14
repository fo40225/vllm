# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage for MuseSpark's native-FP4 (NVFP4 W4A4) MoE experts."""

import fcntl
import gc
import math
import os
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils import cpp_extension

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.muse_spark_utils import (
    assert_cudagraph_replays,
    make_runner,
    nvfp4_quant_config,
    run_moe,
)
from vllm import _custom_ops as ops
from vllm.config import CUDAGraphMode
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.model_executor.layers.fused_moe.utils import remap_topk_to_local
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.models.muse_spark.common.layers import MuseSpark1xRMSNormWithInputScale
from vllm.models.muse_spark.nvidia import sm120_nvfp4_moe
from vllm.models.muse_spark.nvidia.muse_moe_runner import MusePostExpertFinalize
from vllm.models.muse_spark.nvidia.ops import b12x_decode_moe, muse_c
from vllm.models.muse_spark.nvidia.sm120_nvfp4_moe import (
    B12X_MAX_TOKENS,
    MuseSparkSm120NvFp4Experts,
    b12x_decode_engine_enabled,
    decode_kernel_params,
    fold_sm120_nvfp4_scales,
    make_sm120_nvfp4_quant_config,
    swap_gate_up_halves_,
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
# The b12x kernel rounds the intermediate differently from the reference, so some
# of its FP4 codes differ. A handful of routes average that out less, and the
# rows before the norm are dominated by the largest of them.
B12X_STAGE_FAITHFUL_REL_L2 = 0.05
B12X_FEW_ROUTES_REL_L2 = 0.08
B12X_FEW_ROUTES = 8
B12X_ROWS_REL_L2 = 0.08
# The b12x kernel is not bitwise deterministic: runs may differ by about 1% at
# larger shapes.
B12X_RERUN_REL_L2 = 0.03

_E2M1_VALUES = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


@pytest.fixture(autouse=True)
def poisoned_rows(monkeypatch):
    """Make every GEMM2 row that no route refers to NaN, so that a stale row
    leaking into the output cannot go unnoticed."""

    def poison(apply_engine):
        def poisoned(self, *args):
            unfinalized = apply_engine(self, *args)
            rows = unfinalized.gemm2_permuted
            idx = unfinalized.expanded_idx_to_permuted_idx.flatten().long()
            num_rows = rows.shape[0]
            # Routes without a row (-1) are sent to a spare slot.
            used = torch.zeros(num_rows + 1, dtype=torch.bool, device=rows.device)
            used.scatter_(0, torch.where(idx >= 0, idx, num_rows), True)
            rows.masked_fill_(~used[:num_rows].unsqueeze(1), float("nan"))
            return unfinalized

        return poisoned

    for engine in ("_apply_cutlass", "_apply_b12x"):
        monkeypatch.setattr(
            MuseSparkSm120NvFp4Experts,
            engine,
            poison(getattr(MuseSparkSm120NvFp4Experts, engine)),
        )


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


@pytest.fixture
def engine_calls(monkeypatch) -> list[str]:
    """The engine (``cutlass`` or ``b12x``) of every ``apply``, in order."""
    seen: list[str] = []

    def spy(label, apply_engine):
        def spied(self, *args):
            seen.append(label)
            return apply_engine(self, *args)

        return spied

    for label in ("cutlass", "b12x"):
        name = f"_apply_{label}"
        monkeypatch.setattr(
            MuseSparkSm120NvFp4Experts,
            name,
            spy(label, getattr(MuseSparkSm120NvFp4Experts, name)),
        )
    return seen


@pytest.fixture
def quant_kernels(monkeypatch) -> list[str]:
    """The kernel that quantizes the intermediate of every CUTLASS ``apply``, in
    order: ``stock`` for [gate; up] weights, ``muse`` for [up; gate]."""
    seen: list[str] = []
    for label, owner, name in (
        ("stock", ops, "silu_and_mul_scaled_fp4_experts_quant"),
        ("muse", muse_c, "mul_and_silu_scaled_fp4_experts_quant"),
    ):

        def spy(*args, _kernel=getattr(owner, name), _label=label):
            seen.append(_label)
            return _kernel(*args)

        monkeypatch.setattr(owner, name, spy)
    return seen


@pytest.fixture
def hook_rows(monkeypatch) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Copies of the GEMM2 rows and the permuted index of every
    ``UnfinalizedMoEOutput`` the finalizer receives. The rows are checked before
    the norm, which hides any per-row scale error of the GEMMs."""
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []
    finalize = MusePostExpertFinalize.__call__

    def spy(self, unfinalized, out):
        seen.append(
            (
                unfinalized.gemm2_permuted.clone(),
                unfinalized.expanded_idx_to_permuted_idx.clone(),
            )
        )
        finalize(self, unfinalized, out)

    monkeypatch.setattr(MusePostExpertFinalize, "__call__", spy)
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

    def up_gate(self, experts: torch.Tensor | None = None):
        """``w13`` and its linear block scales of ``experts`` (all by default),
        as copies in the stored order: up rows first."""
        index = slice(None) if experts is None else experts
        return (
            torch.cat([self.w13[index, N:], self.w13[index, :N]], dim=1),
            torch.cat([self.w13_sf[index, N:], self.w13_sf[index, :N]], dim=1),
        )

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
    out,
    problem,
    x,
    topk_ids,
    topk_weights,
    norm,
    valid,
    a1_gscale,
    a2_gscale,
    stage_bound: float = STAGE_FAITHFUL_REL_L2,
) -> None:
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all(), "non-finite values leaked into the output"
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for scales, bound in (
            ((a1_gscale, a2_gscale), stage_bound),
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


def _b12x_bound(valid: torch.Tensor) -> float:
    """Bound on the relative L2 error of b12x' final output for ``valid`` routes,
    against the stage-faithful reference or the CUTLASS output."""
    few = valid.sum().item() < B12X_FEW_ROUTES
    return B12X_FEW_ROUTES_REL_L2 if few else B12X_STAGE_FAITHFUL_REL_L2


def _check_rows_against_reference(
    seen: tuple[torch.Tensor, torch.Tensor],
    problem,
    x,
    topk_ids,
    valid,
    a1_gscale,
    a2_gscale,
    bound: float,
) -> torch.Tensor:
    """The unweighted GEMM2 rows of the routes, zero for the remote ones, are
    those of the stage-faithful reference. Returns them [T, top_k, K]."""
    rows, permuted_idx = seen
    assert torch.equal(permuted_idx >= 0, valid)
    routes = rows[permuted_idx.long().clamp(min=0).view(-1)].float()
    routes = routes.view(*permuted_idx.shape, -1)
    routes = torch.where(valid.unsqueeze(-1), routes, torch.zeros_like(routes))
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref = _reference_rows(
            problem,
            x,
            topk_ids.long(),
            valid,
            a1_gscale=a1_gscale,
            a2_gscale=a2_gscale,
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    assert torch.isfinite(routes).all() and ref.abs().max() > 0
    error = _rel_l2(routes, ref)
    assert error <= bound, f"GEMM2 rows: relative L2 error {error:.4f} > {bound}"
    return routes


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
    for name in scales._fields:
        tensor = getattr(scales, name)
        expected = (2,) if name == "gs" else (num_experts,)
        assert tensor.shape == expected and tensor.dtype == torch.float32
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


def test_decode_scales_are_exact_and_consistent_with_the_fold() -> None:
    """The decode kernel's scales: the fold's gate / down alphas, plus the up
    alpha the fold loses and the two global activation scales."""
    g = torch.Generator().manual_seed(1)
    num_experts = 7
    w13_scale_2 = torch.rand(num_experts, 2, generator=g, dtype=torch.float32) * 0.02
    w13_scale_2 += 0.005
    w13_input_scale = torch.rand(num_experts, 2, generator=g, dtype=torch.float32) + 0.5
    w2_scale_2 = torch.rand(num_experts, generator=g, dtype=torch.float32) * 0.02
    w2_input_scale = torch.rand(num_experts, generator=g, dtype=torch.float32) + 0.5
    scales = fold_sm120_nvfp4_scales(
        w13_scale_2, w13_input_scale, w2_scale_2, w2_input_scale
    )
    a13 = w13_input_scale.float().max()
    # Bitwise: the same products as the fold's.
    assert torch.equal(scales.alpha_g, scales.g1_alphas)
    assert torch.equal(scales.alpha2, scales.g2_alphas)
    assert scales.gs[0].item() == scales.a1_gscale[0].item()
    assert scales.gs[0].item() == (1.0 / a13).item()
    assert scales.gs[1].item() == (1.0 / w2_input_scale.float().max()).item()
    # s3 is recoverable from the up alpha, and from the fold's a2_gscale.
    torch.testing.assert_close(
        scales.alpha_u / a13, w13_scale_2[:, 1], rtol=1e-6, atol=0
    )
    s3_over_s1 = scales.alpha_u / scales.alpha_g
    torch.testing.assert_close(
        scales.a2_gscale * w2_input_scale.float().max(), s3_over_s1, rtol=1e-6, atol=0
    )
    prm, gs = decode_kernel_params(scales)
    assert prm.shape == (num_experts, 4) and prm.dtype == torch.float32
    assert torch.equal(prm[:, 0], scales.alpha_g)
    assert torch.equal(prm[:, 1], scales.alpha_u)
    assert torch.equal(prm[:, 2], scales.alpha2)
    assert torch.all(prm[:, 3] == 0)
    assert torch.equal(gs, scales.gs)


@pytest.mark.parametrize("num_experts", [1, 5, 16])
@torch.inference_mode()
def test_swap_gate_up_halves_in_chunks(monkeypatch, num_experts: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(3)
    w13 = torch.randint(
        0, 256, (num_experts, 2 * N, K // 2), dtype=torch.uint8, device="cuda",
        generator=g,
    )  # fmt: skip
    sf = torch.randint(
        0, 120, (num_experts, 2 * N, K // 16), dtype=torch.uint8, device="cuda",
        generator=g,
    ).view(torch.float8_e4m3fn)  # fmt: skip
    expected = (
        torch.cat([w13[:, N:], w13[:, :N]], dim=1),
        torch.cat([sf[:, N:], sf[:, :N]], dim=1),
    )
    # A chunk of three experts, so that the last chunk is partial.
    monkeypatch.setattr(sm120_nvfp4_moe, "_SWAP_CHUNK_BYTES", 3 * N * K // 2)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    w13_ptr, sf_ptr = w13.data_ptr(), sf.data_ptr()
    swap_gate_up_halves_(w13, sf)
    torch.cuda.synchronize()
    # One half of a few experts at a time, not a copy of the tensor.
    assert torch.cuda.max_memory_allocated() - before <= 3 * N * K // 2 + (1 << 18)
    assert w13.data_ptr() == w13_ptr and sf.data_ptr() == sf_ptr
    assert torch.equal(w13, expected[0])
    assert torch.equal(sf.view(torch.uint8), expected[1].view(torch.uint8))


# ------------------------------------------------------------------------------
# The layer: runner -> quant method -> experts -> finalizer
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("moe_backend", ["auto", "cutlass"])
@pytest.mark.parametrize("num_tokens", [1, 16])
@pytest.mark.parametrize("top_k", [4, 8])
@torch.inference_mode()
def test_sm120_layer_matches_the_references(
    dist_init,
    workspace_init,
    row_layouts,
    hook_rows,
    engine_calls,
    quant_kernels,
    moe_backend,
    num_tokens,
    top_k,
) -> None:
    problem = _Problem()
    checkpoint = problem.checkpoint()
    w13_loaded = problem.w13.clone()
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
        moe_backend=moe_backend,
    )
    routed = runner.routed_experts
    experts = routed.quant_method.moe_kernel.fused_experts
    assert type(experts) is MuseSparkSm120NvFp4Experts
    # --moe-backend auto runs b12x for a few tokens inside CUDA graphs: the
    # weights are stored [up; gate] and the intermediate is quantized by the
    # _muse_C kernel. --moe-backend cutlass never runs b12x and needs neither:
    # the weights stay [gate; up], as in the checkpoint, under the stock kernel.
    up_gate = moe_backend == "auto"
    assert experts.up_gate_layout == up_gate
    # The engine is prepared by the first eager forward pass, not at load.
    assert experts.use_b12x_decode == up_gate
    assert experts.b12x is None
    uses_b12x_in_graphs = up_gate and num_tokens <= B12X_MAX_TOKENS

    # The checkpoint scales are not touched, and the kernel reads the swizzled
    # block scales, swapped together with the weights.
    torch.testing.assert_close(routed.w13_weight_scale_2.data, problem.s13)
    torch.testing.assert_close(routed.w2_weight_scale_2.data, problem.s2)
    assert torch.equal(problem.w13, w13_loaded)
    stored_w13, stored_sf = (
        problem.up_gate() if up_gate else (problem.w13, problem.w13_sf)
    )
    assert torch.equal(routed.w13_weight.data, stored_w13)
    assert torch.equal(routed.w2_weight.data, problem.w2)
    assert torch.equal(
        routed.w13_weight_scale.data.view(torch.uint8),
        swizzle_blockscale(stored_sf).view(torch.uint8),
    )
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
    # An eager forward pass is CUTLASS, whatever the batch size.
    assert engine_calls == ["cutlass"]
    assert quant_kernels == ["muse" if up_gate else "stock"]
    # It prepared the engine on the stored weights, if the layer runs it.
    assert (experts.b12x is not None) == up_gate
    if up_gate:
        assert experts.b12x.reads(
            routed.w13_weight.data,
            experts.w1_scale,
            routed.w2_weight.data,
            experts.w2_scale,
        )
    topk_weights, topk_ids = router._compute_routing(x, router_logits, None)
    valid = torch.ones_like(topk_ids, dtype=torch.bool)
    _check_against_references(
        out,
        problem,
        x,
        topk_ids.long(),
        topk_weights,
        norm,
        valid,
        experts.a1_gscale[0].item(),
        experts.a2_gscale,
    )
    _check_rows_against_reference(
        hook_rows[0],
        problem,
        x,
        topk_ids,
        valid,
        experts.a1_gscale[0].item(),
        experts.a2_gscale,
        STAGE_FAITHFUL_REL_L2,
    )
    # One device, one token layout: CUTLASS' permuted rows go straight to the norm.
    assert row_layouts == [False]

    # The captured graph runs b12x for a few tokens (it replays no Python, so
    # this is where b12x pays) and is checked against the eager CUTLASS output.
    assert_cudagraph_replays(
        runner,
        vllm_config,
        x,
        router_logits,
        torch.randn_like(x),
        torch.randn_like(router_logits),
        max_rel_l2=_b12x_bound(valid) if uses_b12x_in_graphs else None,
    )
    assert engine_calls.count("b12x") == int(uses_b12x_in_graphs)
    assert engine_calls[0] == "cutlass"


@torch.inference_mode()
def test_sm120_layer_keeps_the_checkpoint_layout_without_the_muse_c_extension(
    dist_init, workspace_init, engine_calls, quant_kernels, monkeypatch
) -> None:
    """A host that passes the preflight but cannot build the extension: the
    layer neither swaps the weights nor runs b12x, and uses the stock kernel."""

    def cannot_build():
        raise RuntimeError("nvcc: out of luck")

    monkeypatch.setattr(muse_c, "load", cannot_build)
    problem = _Problem()
    norm = _make_norm()
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    router, runner, vllm_config = make_runner(
        norm,
        expert_bias,
        problem.checkpoint(),
        nvfp4_quant_config(),
        hidden_size=K,
        intermediate_size=N,
        top_k=8,
    )
    routed = runner.routed_experts
    experts = routed.quant_method.moe_kernel.fused_experts
    assert not routed.quant_method.use_b12x_decode
    assert not experts.use_b12x_decode
    assert not experts.up_gate_layout and experts.b12x is None
    assert torch.equal(routed.w13_weight.data, problem.w13)
    x = _activations(4, seed=1)
    router_logits = torch.randn(4, E, dtype=torch.float32, device="cuda")
    out = run_moe(runner, vllm_config, x, router_logits)
    assert engine_calls == ["cutlass"] and quant_kernels == ["stock"]
    assert experts.b12x is None
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


@torch.inference_mode()
def test_sm120_layer_without_a_usable_b12x_runs_cutlass_only(
    dist_init, workspace_init, engine_calls, quant_kernels, monkeypatch
) -> None:
    """b12x is importable but unusable (a version mismatch, no CUTLASS DSL): the
    default ``auto`` leaves the layer on CUTLASS and the stock kernel, and the
    model still loads."""
    monkeypatch.setattr(
        sm120_nvfp4_moe, "b12x_unusable_reason", lambda: "b12x lacks plan_weights"
    )
    monkeypatch.setattr(
        muse_c, "load", lambda: pytest.fail("the extension has no use here")
    )
    problem = _Problem()
    norm = _make_norm()
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    router, runner, vllm_config = make_runner(
        norm,
        expert_bias,
        problem.checkpoint(),
        nvfp4_quant_config(),
        hidden_size=K,
        intermediate_size=N,
        top_k=8,
    )
    routed = runner.routed_experts
    experts = routed.quant_method.moe_kernel.fused_experts
    assert not routed.quant_method.use_b12x_decode
    assert not experts.use_b12x_decode
    assert not experts.up_gate_layout and experts.b12x is None
    assert torch.equal(routed.w13_weight.data, problem.w13)
    x = _activations(4, seed=1)
    out = run_moe(runner, vllm_config, x, torch.randn(4, E, device="cuda"))
    assert engine_calls == ["cutlass"] and quant_kernels == ["stock"]
    assert experts.b12x is None
    assert torch.isfinite(out).all()


@torch.inference_mode()
def test_sm120_b12x_setup_failure_leaves_the_layer_on_cutlass(
    dist_init, workspace_init, engine_calls, quant_kernels, monkeypatch
) -> None:
    """b12x passes the probe but fails to set the engine up: ``auto`` never
    raises. The layer already holds [up; gate] (the extension is built), so it
    keeps using the _muse_C kernel. The first eager forward pass makes the one
    attempt, which turns the engine off for this layer and for all later ones."""

    class Failing:
        attempts = 0

        def __init__(self, *args, **kwargs):
            Failing.attempts += 1
            raise TypeError("plan_weights() got an unexpected keyword argument")

    monkeypatch.setattr(sm120_nvfp4_moe, "B12xDecodeMoE", Failing)
    monkeypatch.setattr(b12x_decode_moe, "_setup_failure", None)
    problem = _Problem()
    norm = _make_norm()
    expert_bias = torch.linspace(-0.5, 0.5, E, dtype=torch.float32, device="cuda")
    router, runner, vllm_config = make_runner(
        norm,
        expert_bias,
        problem.checkpoint(),
        nvfp4_quant_config(),
        hidden_size=K,
        intermediate_size=N,
        top_k=8,
    )
    routed = runner.routed_experts
    experts = routed.quant_method.moe_kernel.fused_experts
    # Nothing is attempted at load.
    assert Failing.attempts == 0
    assert experts.up_gate_layout and experts.use_b12x_decode
    assert experts.b12x is None
    assert torch.equal(routed.w13_weight.data, problem.up_gate()[0])
    assert b12x_decode_moe.b12x_unusable_reason() is None

    x = _activations(4, seed=1)
    router_logits = torch.randn(4, E, device="cuda")
    # The first eager forward pass tries once, without raising, and gives up.
    out = run_moe(runner, vllm_config, x, router_logits)
    assert torch.isfinite(out).all()
    assert Failing.attempts == 1
    assert experts.b12x is None and not experts.use_b12x_decode
    assert b12x_decode_moe.b12x_unusable_reason() is not None
    # The layer is not retried on later calls, eager or captured.
    run_moe(runner, vllm_config, x, router_logits)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sm120_nvfp4_moe, "_capturing", lambda: True)
        run_moe(runner, vllm_config, x, router_logits)
    assert Failing.attempts == 1 and experts.b12x is None
    assert engine_calls == ["cutlass"] * 3 and quant_kernels == ["muse"] * 3

    # The next layer does not try again either: it sees the failure and stops.
    other = _Experts(
        problem,
        norm,
        torch.arange(E, dtype=torch.long, device="cuda"),
        8,
        b12x=True,
        prepare=False,
    )
    assert other.experts.use_b12x_decode
    ids, weights = _routes(4, 8, None, remote_tokens=0)
    other.apply(x, ids, weights)
    assert Failing.attempts == 1
    assert other.b12x is None and not other.experts.use_b12x_decode


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
    num_tokens: int,
    top_k: int,
    remote: torch.Tensor | None,
    *,
    remote_tokens: int,
    seed: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random top-k routes. The first ``remote_tokens`` tokens route only to the
    experts in ``remote`` (a global-id tensor), if given."""
    g = torch.Generator(device="cuda").manual_seed(seed)
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


class _Experts:
    """The experts of one rank holding ``local_experts`` (global ids), as the
    layer stores them: w13 as [up; gate] (``up_gate``) or, as in the checkpoint,
    [gate; up]. With ``b12x`` (which needs [up; gate]) the b12x engine is on and
    prepared on the stored weights, as the first eager forward pass would do;
    ``prepare=False`` leaves it for that pass.

    ``expert_map`` is None when every expert is local, and ``a2_gscale`` is the
    kernel's per-expert intermediate scale indexed by global expert id.
    """

    def __init__(
        self,
        problem: _Problem,
        norm: torch.nn.Module,
        local_experts: torch.Tensor,
        top_k: int,
        *,
        tp_size: int = 1,
        b12x: bool = False,
        prepare: bool = True,
        up_gate: bool = True,
    ):
        num_local = local_experts.numel()
        # The block scales are taken per local expert; the input scales are those
        # of all experts, which is what the layer loads.
        if up_gate:
            self.w1, w13_sf = problem.up_gate(local_experts)
        else:
            self.w1 = problem.w13[local_experts].contiguous()
            w13_sf = problem.w13_sf[local_experts].contiguous()
        self.w2 = problem.w2[local_experts].contiguous()
        _, _, quant_config = make_sm120_nvfp4_quant_config(
            w13_sf,
            problem.w2_sf[local_experts].contiguous(),
            problem.s13[local_experts],
            problem.in13,
            problem.s2[local_experts],
            problem.in2,
        )
        moe_config = _moe_config(top_k, num_local, tp_size)
        self.kernel = mk.FusedMoEKernel(
            make_moe_prepare_and_finalize_no_dp_ep(use_monolithic=False),
            MuseSparkSm120NvFp4Experts(
                moe_config=moe_config, quant_config=quant_config
            ),
        )
        self.experts = self.kernel.fused_experts
        self.experts.post_gemm2_hook = MusePostExpertFinalize(norm, moe_config)
        self.experts.up_gate_layout = up_gate
        if b12x:
            self.experts.use_b12x_decode = True
            if prepare:
                self.experts._ensure_b12x(self.w1, self.w2, capturing=False)
                assert self.b12x is not None
        self.expert_map = None
        if num_local < E:
            self.expert_map = torch.full((E,), -1, dtype=torch.int32, device="cuda")
            self.expert_map[local_experts] = torch.arange(
                num_local, dtype=torch.int32, device="cuda"
            )
        self.a1_gscale = quant_config.a1_gscale[0].item()
        self.a2_gscale = torch.zeros(E, dtype=torch.float32, device="cuda")
        self.a2_gscale[local_experts] = quant_config.a2_gscale

    @property
    def b12x(self) -> b12x_decode_moe.B12xDecodeMoE | None:
        return self.experts.b12x

    def apply(
        self,
        x: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        use_b12x: bool = False,
    ) -> torch.Tensor:
        """The finalized output of the experts. They pick b12x only while a CUDA
        graph is captured; ``use_b12x`` makes an eager call take that path, to
        compare the engines outside a capture."""
        assert not use_b12x or self.experts.use_b12x_decode
        with pytest.MonkeyPatch.context() as patch:
            if use_b12x:
                patch.setattr(sm120_nvfp4_moe, "_capturing", lambda: True)
            return self.kernel.apply(
                hidden_states=x,
                w1=self.w1,
                w2=self.w2,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=MoEActivation.SILU,
                global_num_experts=E,
                expert_map=self.expert_map,
                apply_router_weight_on_input=False,
            )


def _run_experts(
    problem: _Problem,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    norm: torch.nn.Module,
    local_experts: torch.Tensor,
    *,
    tp_size: int = 1,
    up_gate: bool = True,
):
    """Runs the CUTLASS experts of one rank holding ``local_experts``, with w13
    stored [up; gate] or [gate; up].

    Returns the output, the expert map, the activation scale and the per-expert
    intermediate scales of ``_Experts``.
    """
    experts = _Experts(
        problem,
        norm,
        local_experts,
        topk_ids.shape[1],
        tp_size=tp_size,
        up_gate=up_gate,
    )
    out = experts.apply(x, topk_ids, topk_weights)
    return out, experts.expert_map, experts.a1_gscale, experts.a2_gscale


@pytest.mark.parametrize("up_gate", [True, False])
@pytest.mark.parametrize("top_k", [4, 8])
@torch.inference_mode()
def test_sm120_expert_parallel_skips_the_remote_routes(
    dist_init, workspace_init, row_layouts, top_k, up_gate
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
        problem, x, topk_ids, topk_weights, norm, local_experts, up_gate=up_gate
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


@pytest.mark.parametrize("up_gate", [True, False])
@pytest.mark.parametrize("top_k", [4, 8])
@torch.inference_mode()
def test_sm120_tensor_parallel_hands_over_route_major_rows(
    dist_init, workspace_init, row_layouts, top_k, up_gate
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
        problem,
        x,
        topk_ids,
        topk_weights,
        norm,
        all_experts,
        tp_size=2,
        up_gate=up_gate,
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


# ------------------------------------------------------------------------------
# The b12x decode engine
# ------------------------------------------------------------------------------
# The experts of the rank are not a contiguous range, so that local ids differ
# from the global ones.
_LOCAL_EXPERTS = [1, 2, 4, 7, 8, 11, 13, 14]
_REMOTE_EXPERTS = [0, 3, 5, 6, 9, 10, 12, 15]
_ROUTINGS = ["mixed", "zero_local", "all_local", "no_expert_map"]


def _routing_case(routing: str, num_tokens: int, top_k: int, seed: int = 2):
    """The local experts, top-k ids and weights of one routing pattern: ``mixed``
    (the first half of the tokens has no local route), ``zero_local``,
    ``all_local`` (EP, but every route is local) and ``no_expert_map`` (every
    expert is local)."""
    local = torch.tensor(_LOCAL_EXPERTS, dtype=torch.long, device="cuda")
    remote = torch.tensor(_REMOTE_EXPERTS, dtype=torch.long, device="cuda")
    if routing == "mixed":
        ids, weights = _routes(
            num_tokens, top_k, remote, remote_tokens=num_tokens // 2, seed=seed
        )
    elif routing == "zero_local":
        ids, weights = _routes(
            num_tokens, top_k, remote, remote_tokens=num_tokens, seed=seed
        )
    elif routing == "all_local":
        ids, weights = _routes(
            num_tokens, top_k, local, remote_tokens=num_tokens, seed=seed
        )
    else:
        local = torch.arange(E, dtype=torch.long, device="cuda")
        ids, weights = _routes(num_tokens, top_k, None, remote_tokens=0, seed=seed)
    return local, ids, weights


def _valid_routes(experts: _Experts, topk_ids: torch.Tensor) -> torch.Tensor:
    if experts.expert_map is None:
        return torch.ones_like(topk_ids, dtype=torch.bool)
    return experts.expert_map[topk_ids.long()] >= 0


def _activations(num_tokens: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(num_tokens, K, device="cuda", generator=g).to(torch.bfloat16)


@pytest.mark.parametrize("top_k", [4, 8])
@pytest.mark.parametrize("routing", _ROUTINGS)
@pytest.mark.parametrize("num_tokens", range(1, B12X_MAX_TOKENS + 1))
@torch.inference_mode()
def test_sm120_b12x_matches_the_references_and_cutlass(
    dist_init, workspace_init, row_layouts, hook_rows, num_tokens, routing, top_k
) -> None:
    problem = _Problem()
    norm = _make_norm()
    local_experts, topk_ids, topk_weights = _routing_case(routing, num_tokens, top_k)
    experts = _Experts(problem, norm, local_experts, top_k, b12x=True)
    x = _activations(num_tokens, seed=1)
    out_b12x = experts.apply(x, topk_ids, topk_weights, use_b12x=True)
    out_cutlass = experts.apply(x, topk_ids, topk_weights)
    # b12x hands over one row per route, CUTLASS the permuted rows.
    assert row_layouts == [True, False]
    valid = _valid_routes(experts, topk_ids)
    if routing == "zero_local":
        assert not valid.any()
        for out in (out_b12x, out_cutlass):
            assert torch.count_nonzero(out) == 0 and torch.isfinite(out).all()
        return
    assert valid.any()
    assert valid.all() == (routing in ("all_local", "no_expert_map"))
    bound = _b12x_bound(valid)
    for out, stage_bound in (
        (out_b12x, bound),
        (out_cutlass, STAGE_FAITHFUL_REL_L2),
    ):
        _check_against_references(
            out,
            problem,
            x,
            topk_ids.long(),
            topk_weights,
            norm,
            valid,
            experts.a1_gscale,
            experts.a2_gscale,
            stage_bound,
        )
    assert _rel_l2(out_b12x, out_cutlass.float()) <= bound
    rows_b12x, rows_cutlass = (
        _check_rows_against_reference(
            seen,
            problem,
            x,
            topk_ids,
            valid,
            experts.a1_gscale,
            experts.a2_gscale,
            rows_bound,
        )
        for seen, rows_bound in zip(
            hook_rows, (B12X_ROWS_REL_L2, STAGE_FAITHFUL_REL_L2)
        )
    )
    assert _rel_l2(rows_b12x, rows_cutlass) <= B12X_ROWS_REL_L2
    # The tokens without a local route get nothing from either engine.
    no_route = ~valid.any(dim=1)
    assert no_route.any() or routing != "mixed" or num_tokens == 1
    assert torch.count_nonzero(out_b12x[no_route]) == 0
    assert torch.count_nonzero(out_cutlass[no_route]) == 0


@pytest.mark.parametrize("num_tokens", [1, 3, 8])
@torch.inference_mode()
def test_sm120_b12x_tensor_parallel_rows_are_route_major(
    dist_init, workspace_init, row_layouts, num_tokens
) -> None:
    """Under TP the CUTLASS engine un-permutes its rows to the order b12x has."""
    problem = _Problem()
    norm = _make_norm()
    top_k = 8
    all_experts = torch.arange(E, dtype=torch.long, device="cuda")
    topk_ids, topk_weights = _routes(num_tokens, top_k, None, remote_tokens=0)
    experts = _Experts(problem, norm, all_experts, top_k, tp_size=2, b12x=True)
    x = _activations(num_tokens, seed=1)
    out_b12x = experts.apply(x, topk_ids, topk_weights, use_b12x=True)
    out_cutlass = experts.apply(x, topk_ids, topk_weights)
    assert row_layouts == [True, True]
    valid = torch.ones_like(topk_ids, dtype=torch.bool)
    _check_against_references(
        out_b12x,
        problem,
        x,
        topk_ids.long(),
        topk_weights,
        norm,
        valid,
        experts.a1_gscale,
        experts.a2_gscale,
        _b12x_bound(valid),
    )
    assert _rel_l2(out_b12x, out_cutlass.float()) <= _b12x_bound(valid)


@torch.inference_mode()
def test_sm120_b12x_leaves_the_stored_weights_alone(dist_init, workspace_init) -> None:
    """b12x consumes the [up; gate] weights and swizzled scales as stored: no
    second copy, and no in-place flip at its first run (which a gate-first
    declaration would do)."""
    problem = _Problem()
    norm = _make_norm()
    top_k = 8
    local_experts, topk_ids, topk_weights = _routing_case("all_local", 8, top_k)
    experts = _Experts(problem, norm, local_experts, top_k, b12x=True)
    assert experts.b12x is not None
    prepared = experts.b12x.experts
    assert prepared.plan.w13_layout == "w13"
    assert prepared.w1_fp4.data_ptr() == experts.w1.data_ptr()
    assert prepared.w2_fp4.data_ptr() == experts.w2.data_ptr()
    assert prepared.w1_blockscale.data_ptr() == experts.experts.w1_scale.data_ptr()
    assert prepared.w2_blockscale.data_ptr() == experts.experts.w2_scale.data_ptr()

    def stored() -> list[torch.Tensor]:
        return [
            experts.w1.clone(),
            experts.w2.clone(),
            experts.experts.w1_scale.view(torch.uint8).clone(),
            experts.experts.w2_scale.view(torch.uint8).clone(),
        ]

    before = stored()
    for num_tokens in (1, 8, 5, 1):
        experts.apply(
            _activations(num_tokens, seed=num_tokens),
            topk_ids[:num_tokens],
            topk_weights[:num_tokens],
            use_b12x=True,
        )
    torch.accelerator.synchronize()
    for kept, now in zip(before, stored()):
        assert torch.equal(kept, now)


@pytest.mark.parametrize("num_tokens", range(1, B12X_MAX_TOKENS + 1))
@torch.inference_mode()
def test_sm120_b12x_cuda_graph_replays_with_changing_routes(
    dist_init, workspace_init, row_layouts, num_tokens
) -> None:
    problem = _Problem()
    norm = _make_norm()
    top_k = 8
    local_experts, ids, weights = _routing_case("mixed", num_tokens, top_k)
    experts = _Experts(problem, norm, local_experts, top_k, b12x=True)
    x_static = _activations(num_tokens, seed=1)
    ids_static, weights_static = ids.clone(), weights.clone()
    # The finalizer compiles at its first call; b12x was prepared with the experts.
    for _ in range(2):
        experts.apply(x_static, ids_static, weights_static, use_b12x=True)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # The experts choose b12x on their own, since a capture is running.
        out_static = experts.apply(x_static, ids_static, weights_static)
    assert row_layouts[-1], "the capture did not run b12x"

    for seed, routing in enumerate(["all_local", "zero_local", "mixed", "mixed"]):
        _, ids, weights = _routing_case(routing, num_tokens, top_k, seed=10 + seed)
        x = _activations(num_tokens, seed=10 + seed)
        x_static.copy_(x)
        ids_static.copy_(ids)
        weights_static.copy_(weights)
        graph.replay()
        replayed = out_static.clone()
        valid = _valid_routes(experts, ids)
        if routing == "zero_local":
            assert torch.count_nonzero(replayed) == 0
            continue
        eager = experts.apply(x, ids, weights, use_b12x=True)
        assert _rel_l2(replayed, eager.float()) <= B12X_RERUN_REL_L2
        _check_against_references(
            replayed,
            problem,
            x,
            ids.long(),
            weights,
            norm,
            valid,
            experts.a1_gscale,
            experts.a2_gscale,
            _b12x_bound(valid),
        )


@torch.inference_mode()
def test_sm120_b12x_runs_only_while_a_graph_is_captured(
    dist_init, workspace_init, row_layouts
) -> None:
    """An eager call is CUTLASS for any batch, a capture is b12x for up to
    ``B12X_MAX_TOKENS`` tokens and CUTLASS above, as a replay runs no Python."""
    problem = _Problem()
    norm = _make_norm()
    top_k = 8
    all_experts = torch.arange(E, dtype=torch.long, device="cuda")
    experts = _Experts(problem, norm, all_experts, top_k, b12x=True)
    assert experts.b12x is not None
    bindings = dict(experts.b12x._bindings)
    assert sorted(bindings) == list(range(1, B12X_MAX_TOKENS + 1))
    max_tokens = B12X_MAX_TOKENS + 1
    ids, weights = _routes(max_tokens, top_k, None, remote_tokens=0)
    x = _activations(max_tokens, seed=1)
    for num_tokens in range(1, max_tokens + 1):
        experts.apply(x[:num_tokens], ids[:num_tokens], weights[:num_tokens])
    assert row_layouts == [False] * max_tokens

    # The route-major hook compiles at its first call, so before the captures.
    capture_sizes = (1, 5, B12X_MAX_TOKENS, max_tokens)
    for num_tokens in capture_sizes:
        experts.apply(
            x[:num_tokens], ids[:num_tokens], weights[:num_tokens], use_b12x=True
        )
    row_layouts.clear()
    for num_tokens in capture_sizes:
        x_static = x[:num_tokens].clone()
        ids_static, weights_static = ids[:num_tokens], weights[:num_tokens]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out_static = experts.apply(x_static, ids_static, weights_static)
        graph.replay()
        assert torch.isfinite(out_static).all()
    assert row_layouts == [True, True, True, False]
    # Nothing was bound inside a capture.
    assert all(experts.b12x._bindings[n] is bindings[n] for n in bindings)


@torch.inference_mode()
def test_sm120_b12x_is_prepared_by_the_first_eager_forward(
    dist_init, workspace_init, row_layouts
) -> None:
    """Nothing is prepared at load. A capture before any eager forward pass
    raises and prepares nothing; the first eager pass prepares the engine on the
    stored weights, and a capture after it runs b12x."""
    problem = _Problem()
    norm = _make_norm()
    top_k = 8
    all_experts = torch.arange(E, dtype=torch.long, device="cuda")
    experts = _Experts(problem, norm, all_experts, top_k, b12x=True, prepare=False)
    assert experts.experts.use_b12x_decode and experts.b12x is None
    num_tokens = 3
    ids, weights = _routes(num_tokens, top_k, None, remote_tokens=0)
    x = _activations(num_tokens, seed=1)

    # The patched capture raises before anything runs, so no stream is captured.
    with pytest.raises(RuntimeError, match="prepared on"):
        experts.apply(x, ids, weights, use_b12x=True)
    assert experts.b12x is None and experts.experts.use_b12x_decode

    experts.apply(x, ids, weights)
    engine = experts.b12x
    assert engine is not None
    assert engine.reads(
        experts.w1, experts.experts.w1_scale, experts.w2, experts.experts.w2_scale
    )
    assert engine.experts.w1_fp4.data_ptr() == experts.w1.data_ptr()
    assert engine.experts.w2_fp4.data_ptr() == experts.w2.data_ptr()
    # The next eager pass keeps it, and a capture runs it.
    experts.apply(x, ids, weights)
    assert experts.b12x is engine
    x_static = x.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        experts.apply(x_static, ids, weights)
    assert row_layouts[-1], "the capture did not run b12x"
    assert experts.b12x is engine


@pytest.mark.parametrize("num_tokens", [1, 3, B12X_MAX_TOKENS])
@torch.inference_mode()
def test_sm120_b12x_follows_the_weights_that_apply_receives(
    dist_init, workspace_init, row_layouts, num_tokens
) -> None:
    """If ``apply`` gets other tensors than the ones the engine was prepared on,
    an eager call prepares it again and a capture raises."""
    problem = _Problem()
    norm = _make_norm()
    top_k = 8
    local_experts, topk_ids, topk_weights = _routing_case("mixed", num_tokens, top_k)
    experts = _Experts(problem, norm, local_experts, top_k, b12x=True)
    x = _activations(num_tokens, seed=1)
    before = experts.apply(x, topk_ids, topk_weights, use_b12x=True)
    engine = experts.b12x
    assert engine.reads(
        experts.w1, experts.experts.w1_scale, experts.w2, experts.experts.w2_scale
    )
    weights_before = [experts.w1.clone(), experts.w2.clone()]

    # The reload copies the processed weights into the persistent tensors, which
    # then replace the ones the engine was prepared on.
    experts.w1 = experts.w1.clone()
    experts.w2 = experts.w2.clone()
    assert not engine.reads(
        experts.w1, experts.experts.w1_scale, experts.w2, experts.experts.w2_scale
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(sm120_nvfp4_moe, "_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="prepared on"):
            experts.apply(x, topk_ids, topk_weights)
    assert experts.b12x is engine

    # An eager call, of any engine, prepares it again on the new tensors.
    after = experts.apply(x, topk_ids, topk_weights)
    rebound = experts.b12x
    assert rebound is not None and rebound is not engine
    assert rebound.reads(
        experts.w1, experts.experts.w1_scale, experts.w2, experts.experts.w2_scale
    )
    assert rebound.experts.w1_fp4.data_ptr() == experts.w1.data_ptr()
    assert rebound.experts.w2_fp4.data_ptr() == experts.w2.data_ptr()
    assert torch.equal(experts.w1, weights_before[0])
    assert torch.equal(experts.w2, weights_before[1])
    # Calls on the same tensors do not prepare it once more.
    out = experts.apply(x, topk_ids, topk_weights, use_b12x=True)
    assert experts.b12x is rebound
    assert _rel_l2(out, before.float()) <= B12X_RERUN_REL_L2
    assert _rel_l2(out, after.float()) <= _b12x_bound(
        _valid_routes(experts, topk_ids)
    )


@torch.inference_mode()
def test_sm120_layer_weight_reload_keeps_the_layout_and_prepares_b12x_late(
    dist_init, workspace_init, engine_calls, monkeypatch
) -> None:
    """The reload path (layerwise reload, refit) runs ``process_weights_after_
    loading`` on temporary tensors holding the checkpoint again, then copies the
    result into the persistent ones and puts them back on the layer. No b12x
    engine is bound to the temporaries at any point: the next eager forward pass
    prepares it on the persistent tensors."""
    problem = _Problem()
    checkpoint = problem.checkpoint()
    norm = _make_norm()
    top_k = 8
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
    quant_method = routed.quant_method
    first_experts = quant_method.moe_kernel.fused_experts
    assert first_experts.use_b12x_decode and first_experts.b12x is None
    x = _activations(3, seed=1)
    router_logits = torch.randn(3, E, dtype=torch.float32, device="cuda")
    before = run_moe(runner, vllm_config, x, router_logits)
    assert first_experts.b12x is not None
    assert first_experts.b12x.reads(
        routed.w13_weight.data,
        first_experts.w1_scale,
        routed.w2_weight.data,
        first_experts.w2_scale,
    )

    # Every engine prepared from here on.
    prepared: list[b12x_decode_moe.B12xDecodeMoE] = []
    engine_cls = sm120_nvfp4_moe.B12xDecodeMoE

    def spy(*args, **kwargs):
        prepared.append(engine_cls(*args, **kwargs))
        return prepared[-1]

    monkeypatch.setattr(sm120_nvfp4_moe, "B12xDecodeMoE", spy)

    names = ("w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale")
    persistent = {name: getattr(routed, name) for name in names}
    pointers = {name: t.data_ptr() for name, t in persistent.items()}
    temporaries = {
        "w13_weight": problem.w13.clone(),
        "w13_weight_scale": problem.w13_sf.clone(),
        "w2_weight": problem.w2.clone(),
        "w2_weight_scale": problem.w2_sf.clone(),
    }
    weight_bytes = sum(temporaries[name].nbytes for name in ("w13_weight", "w2_weight"))
    for name, tensor in temporaries.items():
        setattr(routed, name, torch.nn.Parameter(tensor, requires_grad=False))
    quant_method.process_weights_after_loading(routed)
    experts = quant_method.moe_kernel.fused_experts
    assert experts is not first_experts and experts.up_gate_layout
    # The checkpoint order was swapped again, on the temporaries.
    assert torch.equal(routed.w13_weight.data, problem.up_gate()[0])
    # Nothing is bound to the temporaries: the layer is ready to run b12x, but no
    # engine exists yet.
    assert experts.use_b12x_decode
    assert experts.b12x is None and prepared == []
    for name, tensor in persistent.items():
        tensor.data.copy_(getattr(routed, name))
        setattr(routed, name, tensor)
    for name, tensor in persistent.items():
        assert tensor.data_ptr() == pointers[name]
    assert experts.b12x is None and prepared == []

    # Dropping the test's own references frees the temporary weights: nothing
    # else (an engine, say) holds them.
    torch.accelerator.synchronize()
    gc.collect()
    allocated = torch.cuda.memory_allocated()
    del temporaries
    gc.collect()
    assert allocated - torch.cuda.memory_allocated() >= weight_bytes

    after = run_moe(runner, vllm_config, x, router_logits)
    torch.testing.assert_close(after, before)
    assert len(prepared) == 1 and experts.b12x is prepared[0]
    assert experts.b12x.reads(
        routed.w13_weight.data,
        experts.w1_scale,
        routed.w2_weight.data,
        experts.w2_scale,
    )
    assert experts.b12x.experts.w1_fp4.data_ptr() == pointers["w13_weight"]
    assert experts.b12x.experts.w2_fp4.data_ptr() == pointers["w2_weight"]
    assert engine_calls == ["cutlass", "cutlass"]
    # Later passes keep it.
    run_moe(runner, vllm_config, x, router_logits)
    assert len(prepared) == 1


def test_b12x_probe_checks_that_the_installed_b12x_is_usable(monkeypatch) -> None:
    probe = b12x_decode_moe._probe_b12x
    symbols = {name: object() for name in b12x_decode_moe._REQUIRED_SYMBOLS}

    def reason_with(module=None, *, error=None, installed=True) -> str | None:
        """The probe's verdict on a b12x whose fused_moe module is ``module``, or
        whose import raises ``error``."""

        def import_module(name):
            assert name == "b12x.moe.fused_moe"
            if error is not None:
                raise error
            return module

        probe.cache_clear()
        try:
            with monkeypatch.context() as patch:
                patch.setattr(
                    b12x_decode_moe.importlib.util,
                    "find_spec",
                    lambda name: object() if installed else None,
                )
                patch.setattr(b12x_decode_moe.importlib, "import_module", import_module)
                return probe()
        finally:
            probe.cache_clear()

    assert reason_with(SimpleNamespace(**symbols, is_supported=lambda: True)) is None
    # A b12x that has no support query is taken as it comes.
    assert reason_with(SimpleNamespace(**symbols)) is None
    reason = reason_with(SimpleNamespace(**symbols, is_supported=lambda: False))
    assert reason is not None and "support" in reason
    for missing in b12x_decode_moe._REQUIRED_SYMBOLS:
        partial = {k: v for k, v in symbols.items() if k != missing}
        reason = reason_with(SimpleNamespace(**partial))
        assert reason is not None and missing in reason
    # nvidia-cutlass-dsl missing, or a version that fails at the import.
    for error in (ModuleNotFoundError("cutlass"), AttributeError("api drift")):
        reason = reason_with(error=error)
        assert reason is not None and type(error).__name__ in reason
    reason = reason_with(SimpleNamespace(**symbols, is_supported=lambda: 1 / 0))
    assert reason is not None and "ZeroDivisionError" in reason
    assert reason_with(installed=False) == "b12x is not installed"


def test_b12x_decode_engine_selection(monkeypatch) -> None:
    enabled = b12x_decode_engine_enabled
    # The host has a usable b12x and a build toolchain, unless a case says not.
    monkeypatch.setattr(sm120_nvfp4_moe, "b12x_unusable_reason", lambda: None)
    monkeypatch.setattr(muse_c, "preflight", lambda: None)

    def set_config(cudagraph_mode, enforce_eager=False):
        config = SimpleNamespace(
            compilation_config=SimpleNamespace(cudagraph_mode=cudagraph_mode),
            model_config=SimpleNamespace(enforce_eager=enforce_eager),
        )
        monkeypatch.setattr(
            sm120_nvfp4_moe, "get_current_vllm_config_or_none", lambda: config
        )

    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    for mode in (
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.FULL_DECODE_ONLY,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    ):
        set_config(mode)
        assert enabled("auto")
        assert not enabled("cutlass")
        assert not enabled("marlin")
    set_config(CUDAGraphMode.NONE)
    assert not enabled("auto")
    # --enforce-eager resets the mode too, but the engine does not rely on that.
    set_config(CUDAGraphMode.FULL_AND_PIECEWISE, enforce_eager=True)
    assert not enabled("auto")
    set_config(CUDAGraphMode.FULL_AND_PIECEWISE)
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    assert not enabled("auto")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT")
    assert enabled("auto")
    # b12x that cannot be used, or no way to build the extension.
    monkeypatch.setattr(
        sm120_nvfp4_moe, "b12x_unusable_reason", lambda: "b12x is not installed"
    )
    assert not enabled("auto")
    monkeypatch.setattr(sm120_nvfp4_moe, "b12x_unusable_reason", lambda: None)
    monkeypatch.setattr(muse_c, "preflight", lambda: "nvcc was not found")
    assert not enabled("auto")
    monkeypatch.setattr(muse_c, "preflight", lambda: None)
    assert enabled("auto")
    monkeypatch.setattr(
        sm120_nvfp4_moe, "get_current_vllm_config_or_none", lambda: None
    )
    assert not enabled("auto")


# ------------------------------------------------------------------------------
# The _muse_C extension
# ------------------------------------------------------------------------------
@pytest.fixture
def muse_c_build(monkeypatch, tmp_path):
    """``muse_c.load`` against a fresh cache root, with a fake compiler: the
    build directory, and the calls ``cpp_extension.load`` gets."""
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    nvcc = "/usr/local/cuda/bin/nvcc"
    monkeypatch.setattr(muse_c, "_nvcc_path", lambda: nvcc)
    monkeypatch.setattr(muse_c, "_nvcc_version", lambda path: (13, 0))
    monkeypatch.setattr(muse_c, "_loaded", False)
    monkeypatch.setattr(muse_c, "_load_error", None)
    muse_c.preflight.cache_clear()
    loaded: list[str] = []
    monkeypatch.setattr(torch.ops, "load_library", loaded.append)
    cflags, cuda_cflags = muse_c._build_flags((13, 0))
    build_dir = muse_c._build_directory(nvcc, cflags, cuda_cflags)
    os.makedirs(build_dir)
    yield SimpleNamespace(build_dir=build_dir, loaded=loaded)
    muse_c.preflight.cache_clear()


def _fake_build(build_dir: str, calls: list[str]):
    def build(**kwargs):
        assert kwargs["build_directory"] == build_dir
        calls.append("build")
        # No other process may build while this one does.
        with open(os.path.join(build_dir, muse_c._BUILD_FLOCK), "a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # The lock of torch's FileBaton would make it wait for ever.
        assert not os.path.exists(os.path.join(build_dir, muse_c._TORCH_BUILD_LOCK))
        library = os.path.join(build_dir, "_muse_C.so")
        with open(library, "w") as f:
            f.write("library")
        return library

    return build


def test_muse_c_load_survives_a_build_lock_of_a_killed_builder(
    monkeypatch, muse_c_build
) -> None:
    build_dir = muse_c_build.build_dir
    calls: list[str] = []
    monkeypatch.setattr(cpp_extension, "load", _fake_build(build_dir, calls))
    # What a SIGKILLed builder leaves behind.
    stale_lock = os.path.join(build_dir, muse_c._TORCH_BUILD_LOCK)
    with open(stale_lock, "w"):
        pass
    muse_c.load()
    assert calls == ["build"]
    assert not os.path.exists(stale_lock)
    library = os.path.join(build_dir, "_muse_C.so")
    assert muse_c._built_library(build_dir) == library
    # The library is loaded by the build, and by nobody else.
    assert muse_c_build.loaded == []
    # The next process finds the finished build: no flock, no compiler, no ninja.
    monkeypatch.setattr(muse_c, "_loaded", False)
    monkeypatch.setattr(cpp_extension, "load", lambda **kw: pytest.fail("rebuilt"))
    monkeypatch.setattr(cpp_extension, "is_ninja_available", lambda: False)
    assert muse_c.preflight() is None
    muse_c.load()
    assert muse_c_build.loaded == [library]


def test_muse_c_load_waits_for_a_live_builder(monkeypatch, muse_c_build) -> None:
    build_dir = muse_c_build.build_dir
    calls: list[str] = []
    monkeypatch.setattr(cpp_extension, "load", _fake_build(build_dir, calls))
    done = threading.Event()
    errors: list[BaseException] = []

    def load():
        try:
            muse_c.load()
        except BaseException as e:
            errors.append(e)
        finally:
            done.set()

    with open(os.path.join(build_dir, muse_c._BUILD_FLOCK), "a") as builder:
        fcntl.flock(builder, fcntl.LOCK_EX)
        thread = threading.Thread(target=load)
        thread.start()
        assert not done.wait(1.0), "load did not wait for the builder"
        # The builder finishes and publishes its library.
        library = os.path.join(build_dir, "_muse_C.so")
        with open(library, "w") as f:
            f.write("library")
        with open(os.path.join(build_dir, muse_c._BUILT_MARKER), "w") as f:
            f.write("_muse_C.so")
        fcntl.flock(builder, fcntl.LOCK_UN)
    thread.join(30)
    assert done.is_set() and not errors
    # It loaded what the other process built, and built nothing itself.
    assert calls == [] and muse_c_build.loaded == [library]


def test_muse_c_load_failure_is_final(monkeypatch, muse_c_build) -> None:
    calls: list[str] = []

    def fail(**kwargs):
        calls.append("build")
        raise RuntimeError("nvcc fatal")

    monkeypatch.setattr(cpp_extension, "load", fail)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="nvcc fatal|failed to load earlier"):
            muse_c.load()
    assert calls == ["build"]
    assert muse_c._built_library(muse_c_build.build_dir) is None


def test_muse_c_preflight_names_what_is_missing(monkeypatch, muse_c_build) -> None:
    build_dir = muse_c_build.build_dir
    monkeypatch.setattr(cpp_extension, "is_ninja_available", lambda: True)
    muse_c.preflight.cache_clear()
    assert muse_c.preflight() is None

    monkeypatch.setattr(cpp_extension, "is_ninja_available", lambda: False)
    muse_c.preflight.cache_clear()
    assert "ninja" in muse_c.preflight()
    # A finished build needs no ninja.
    with open(os.path.join(build_dir, "_muse_C.so"), "w") as f:
        f.write("library")
    with open(os.path.join(build_dir, muse_c._BUILT_MARKER), "w") as f:
        f.write("_muse_C.so")
    muse_c.preflight.cache_clear()
    assert muse_c.preflight() is None
    # ...but a half-written library, without the marker of a finished build, is
    # not a build.
    os.remove(os.path.join(build_dir, muse_c._BUILT_MARKER))
    muse_c.preflight.cache_clear()
    assert "ninja" in muse_c.preflight()

    def no_nvcc():
        raise RuntimeError("nvcc was not found")

    monkeypatch.setattr(muse_c, "_nvcc_path", no_nvcc)
    muse_c.preflight.cache_clear()
    assert "nvcc was not found" in muse_c.preflight()


@pytest.mark.parametrize("num_local_experts", [8, 32])
@pytest.mark.parametrize("num_tokens", [8, 4096])
@torch.inference_mode()
def test_muse_c_quantization_is_bit_identical_to_the_stock_kernel(
    num_tokens: int, num_local_experts: int
) -> None:
    """The [up | gate] kernel on the exchanged halves gives the bytes of the
    stock kernel, for the offsets of one rank of an 8-way EP deployment: a few
    rows per expert take the low-latency path, thousands the grid-stride one."""
    muse_c.load()
    num_experts, top_k, hidden, intermediate = num_local_experts * 8, 8, 512, 1024
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    topk_ids = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device="cuda",
        generator=g,
    )  # fmt: skip
    # Rank 3 holds a contiguous range.
    expert_map = torch.full((num_experts,), -1, dtype=torch.int32, device="cuda")
    expert_map[3 * num_local_experts : 4 * num_local_experts] = torch.arange(
        num_local_experts, dtype=torch.int32, device="cuda"
    )
    local_ids = remap_topk_to_local(topk_ids, expert_map, out_dtype=torch.int32)
    num_routes = num_tokens * top_k
    expert_offsets = torch.empty(
        num_local_experts + 1, dtype=torch.int32, device="cuda"
    )
    blockscale_offsets = torch.empty_like(expert_offsets)
    problem_sizes1 = torch.empty(
        (num_local_experts, 3), dtype=torch.int32, device="cuda"
    )
    problem_sizes2 = torch.empty_like(problem_sizes1)
    a_map = torch.empty(num_routes, dtype=torch.int32, device="cuda")
    c_map = torch.empty_like(a_map)
    ops.get_cutlass_moe_mm_data(
        local_ids,
        expert_offsets,
        problem_sizes1,
        problem_sizes2,
        a_map,
        c_map,
        num_local_experts,
        intermediate,
        hidden,
        blockscale_offsets,
        is_gated=True,
    )
    local_rows = expert_offsets[-1].item()
    assert 0 < local_rows < num_routes
    gate_up = torch.randn(
        num_routes, 2 * intermediate, device="cuda", generator=g, dtype=torch.float32
    ).to(torch.bfloat16)
    up_gate = torch.cat([gate_up[:, intermediate:], gate_up[:, :intermediate]], dim=1)
    gscale = 300 + 600 * torch.rand(
        num_local_experts, device="cuda", generator=g, dtype=torch.float32
    )

    def run(op, activations):
        # Zero-filled, so that the padding no kernel writes compares equal too.
        fp4 = torch.zeros(
            num_routes, intermediate // 2, dtype=torch.uint8, device="cuda"
        )
        scale_rows = num_routes + 128 * num_local_experts
        sf = torch.zeros(
            scale_rows, intermediate // 64, dtype=torch.int32, device="cuda"
        )
        op(fp4, sf, activations, gscale, expert_offsets, blockscale_offsets)
        return fp4, sf

    stock_fp4, stock_sf = run(
        torch.ops._C.silu_and_mul_scaled_fp4_experts_quant, gate_up
    )
    muse_fp4, muse_sf = run(
        torch.ops._muse_C.mul_and_silu_scaled_fp4_experts_quant, up_gate
    )
    assert stock_fp4[:local_rows].any() and stock_sf.any()
    assert torch.equal(muse_fp4, stock_fp4)
    assert torch.equal(muse_sf, stock_sf)
