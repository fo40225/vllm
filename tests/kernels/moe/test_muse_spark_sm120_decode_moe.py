# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage for the NVFP4 decode MoE kernel (``_muse_C::nvfp4_decode_moe``).

The kernel has fixed hidden = intermediate = 4096 and top-k 8 (any number of local
experts; the EP8 case of 32 and the EP4 case of 64 are used here), so the tests
build one random layer of that size. The reference is an
exact fake-quantization in torch (activations and the SiLU * up intermediate
quantized to NVFP4 at the points the CUTLASS composition quantizes them).
"""

import pytest
import torch
import torch.nn.functional as F

from tests.kernels.muse_spark_pdl_utils import slow_copy, synchronize_or_exit
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.utils import remap_topk_to_local
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    swizzle_blockscale,
)
from vllm.models.muse_spark.nvidia.ops import muse_c
from vllm.models.muse_spark.nvidia.sm120_nvfp4_moe import (
    decode_kernel_params,
    fold_sm120_nvfp4_scales,
)
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

_unusable = muse_c.preflight()
if _unusable is not None:
    pytest.skip(
        f"The _muse_C extension cannot be built: {_unusable}", allow_module_level=True
    )

# Local experts per rank: 32 is this model's EP8 layout (the engine test imports
# it); the layer fixture also runs EP4 with 64, since the kernel's plan and
# counters scale with the number of local experts.
E_LOCAL = 32
E_LOCAL_CASES = (E_LOCAL, 64)
E_GLOBAL = 256
H = 4096
TOP_K = 8
FP8_MAX = 448.0
FP4_MAX = 6.0
NIB_RMS = 2.9269  # RMS of a uniformly random e2m1 code
# Round-ish headroom factors put bf16 inputs on exact ties of the e2m1 grid,
# where the kernel's approximate reciprocal and an exact emulation may disagree.
HEADROOM_A13 = 1.0937
HEADROOM_A2 = 1.2371
# Rel L2 of the final output against the reference that quantizes at the same
# points as the kernel (the tie flips of a few intermediate codes dominate).
FQ_REL_L2 = 3e-3
# The reference rounds along a different path than the kernel (fp32 matmul of
# dequantized values, scales applied after the sum), so an intermediate value
# can land on the other side of an e2m1 tie. The raw tensor core sums are exact
# and the stages equal CUTLASS's bit for bit (see the stage test); such a flip
# is worth about 0.5% when the output has only T independent rows, as with every
# route on the same expert.
FQ_REL_L2_FEW_ROWS = 1e-2
BF16_REL_L2 = 0.25
FP4_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_LUT = FP4_LUT + [-v for v in FP4_LUT]


def _lut(device: torch.device) -> torch.Tensor:
    return torch.tensor(FP4_LUT, dtype=torch.float32, device=device)


def _dequant(packed: torch.Tensor, sf: torch.Tensor, scale2: float) -> torch.Tensor:
    """packed [R, C / 2] u8 (low nibble = even column), linear block scales
    [R, C / 16] e4m3 -> fp32 [R, C]."""
    lut = _lut(packed.device)
    both = torch.stack([lut[(packed & 0xF).long()], lut[(packed >> 4).long()]], -1)
    w = both.view(packed.shape[0], -1, 16) * (sf.float().unsqueeze(-1) * scale2)
    return w.view(packed.shape[0], -1)


def _rne_fp4(q: torch.Tensor) -> torch.Tensor:
    ax = q.abs()
    code = (
        (ax > 0.25).to(torch.int8)
        + (ax >= 0.75).to(torch.int8)
        + (ax > 1.25).to(torch.int8)
        + (ax >= 1.75).to(torch.int8)
        + (ax > 2.5).to(torch.int8)
        + (ax >= 3.5).to(torch.int8)
        + (ax > 5.0).to(torch.int8)
    )
    v = _lut(q.device)[code.long()]
    return torch.where(q < 0, -v, v)


def _fake_quant(x: torch.Tensor, gscale: float) -> torch.Tensor:
    """NVFP4 quantize -> dequantize of x [R, C] with global scale ``gscale``."""
    rows, cols = x.shape
    xb = x.float().view(rows, cols // 16, 16)
    amax = xb.abs().amax(-1)
    s8 = (amax * (gscale / FP4_MAX)).clamp(max=FP8_MAX).to(torch.float8_e4m3fn).float()
    inv = torch.where(s8 > 0, gscale / s8.clamp_min(1e-30), torch.zeros_like(s8))
    q = _rne_fp4(xb * inv.unsqueeze(-1))
    return (q * (s8 / gscale).unsqueeze(-1)).view(rows, cols)


def _finalize(rows, weights, gamma, valid):
    """Post-expert RMSNorm (weight gamma + 1) and router-weighted sum of the
    valid routes. rows [T, 8, H] fp32."""
    z = rows.float() * (gamma.float() + 1.0)
    y = z * torch.rsqrt(z.pow(2).mean(-1, keepdim=True) + 1e-8)
    y = torch.where(valid.unsqueeze(-1), y, torch.zeros_like(y))
    return (weights.float().unsqueeze(-1) * y).sum(1)


class _Layer:
    """A random expert-parallel rank: [gate; up] weights, scales and the reference."""

    def __init__(self, diff_s: bool, seed: int = 0, e_local: int = E_LOCAL):
        dev = torch.device("cuda")
        self.e_local = e_local
        g = torch.Generator(device=dev).manual_seed(1000 + seed)
        self.diff_s = diff_s
        self.references = {}
        self.w13 = torch.randint(
            0, 256, (e_local, 2 * H, H // 2), dtype=torch.uint8, device=dev, generator=g
        )
        self.w2 = torch.randint(
            0, 256, (e_local, H, H // 2), dtype=torch.uint8, device=dev, generator=g
        )

        def block_scales(shape):
            ratio = torch.exp(-torch.randn(shape, device=dev, generator=g).abs() * 0.6)
            return (FP8_MAX * ratio).clamp(min=1.0).to(torch.float8_e4m3fn)

        self.w13_sf = block_scales((e_local, 2 * H, H // 16))
        self.w2_sf = block_scales((e_local, H, H // 16))

        def rms(t):
            return t.float().pow(2).mean(dim=(1, 2)).sqrt()

        def spread():
            return torch.rand(e_local, device=dev, generator=g) * 0.7 + 0.7

        self.s1 = (spread() / H**0.5 / (NIB_RMS * rms(self.w13_sf[:, :H]))).float()
        ratio = torch.rand(e_local, device=dev, generator=g) * 3.5 + 0.5
        self.s3 = self.s1 * (ratio if diff_s else 1.0)
        self.s2 = (spread() / H**0.5 / (NIB_RMS * rms(self.w2_sf))).float()
        # Global input scales calibrated on random activations.
        calib = torch.randn(256, H, device=dev, generator=g).to(torch.bfloat16)
        self.a13 = HEADROOM_A13 * calib.float().abs().max() / (FP8_MAX * FP4_MAX)
        self.a2 = torch.ones((), device=dev)
        hmax = 0.0
        x = _fake_quant(calib[:64], float(1 / self.a13))
        for e in range(4):
            wg, wu, _ = self.dequant_expert(e)
            gate = (x @ wg.T).bfloat16().float()
            up = (x @ wu.T).bfloat16().float()
            hmax = max(hmax, (F.silu(gate) * up).abs().max().item())
        self.a2 = torch.tensor(HEADROOM_A2 * hmax / (FP8_MAX * FP4_MAX), device=dev)
        self.gamma = (torch.randn(H, device=dev, generator=g) * 0.1).to(torch.bfloat16)
        w13_scale_2 = torch.stack([self.s1, self.s3], 1)
        in13 = self.a13 * (0.35 + 0.65 * torch.rand(e_local, device=dev, generator=g))
        in13[0] = self.a13
        in2 = self.a2 * (0.35 + 0.65 * torch.rand(e_local, device=dev, generator=g))
        in2[0] = self.a2
        self.scales = fold_sm120_nvfp4_scales(
            w13_scale_2, in13.unsqueeze(1).expand(-1, 2), self.s2, in2
        )
        self.prm, self.gs = decode_kernel_params(self.scales)
        self.expert_map = torch.full((E_GLOBAL,), -1, dtype=torch.int32, device=dev)
        self.expert_map[:e_local] = torch.arange(e_local, dtype=torch.int32, device=dev)
        self.sf13 = swizzle_blockscale(self.w13_sf)
        self.sf2 = swizzle_blockscale(self.w2_sf)

    def dequant_expert(self, e: int):
        wg = _dequant(self.w13[e, :H], self.w13_sf[e, :H], float(self.s1[e]))
        wu = _dequant(self.w13[e, H:], self.w13_sf[e, H:], float(self.s3[e]))
        w2 = _dequant(self.w2[e], self.w2_sf[e], float(self.s2[e]))
        return wg, wu, w2

    def reference_rows(self, u: torch.Tensor, ids: torch.Tensor, mode: str = "fq"):
        """Unweighted pre-norm rows [T, 8, H] fp32 (zero for non-local routes)."""
        lids = self.expert_map[ids.long()]
        x = _fake_quant(u, float(1 / self.a13)) if mode == "fq" else u.float()
        rows = torch.zeros(u.shape[0], TOP_K, H, device=u.device)
        for e in torch.unique(lids[lids >= 0]).tolist():
            tok, slot = (lids == e).nonzero(as_tuple=True)
            wg, wu, w2 = self.dequant_expert(e)
            gate = (x[tok] @ wg.T).bfloat16().float()
            up = (x[tok] @ wu.T).bfloat16().float()
            h = (F.silu(gate) * up).bfloat16().float()
            if mode == "fq":
                h = _fake_quant(h, float(1 / self.a2))
            rows[tok, slot] = (h @ w2.T).bfloat16().float()
        return rows, lids >= 0


class _Engine:
    """Static buffers and the op call, as a layer of the model would hold them."""

    def __init__(self, layer: _Layer, up_first: bool = False):
        dev = torch.device("cuda")
        self.layer = layer
        self.up_first = up_first
        if up_first:
            self.w13 = torch.cat([layer.w13[:, H:], layer.w13[:, :H]], 1).contiguous()
            sf = torch.cat([layer.w13_sf[:, H:], layer.w13_sf[:, :H]], 1).contiguous()
            self.sf13 = swizzle_blockscale(sf)
        else:
            self.w13, self.sf13 = layer.w13, layer.sf13
        self.hq_data = torch.zeros(64, H // 2, dtype=torch.uint8, device=dev)
        self.hq_sf = torch.zeros(64, H // 16, dtype=torch.uint8, device=dev)
        self.plan = torch.zeros(1024, dtype=torch.int32, device=dev)
        # Counters of the persistent kernel: zeroed once, reset by every launch.
        self.ctl = torch.zeros(128, dtype=torch.int32, device=dev)
        self.buffers = {}

    def out_buffers(self, t_pad: int):
        if t_pad not in self.buffers:
            self.buffers[t_pad] = (
                torch.zeros(t_pad * 8, H, dtype=torch.bfloat16, device="cuda"),
                torch.zeros(t_pad * 8, dtype=torch.int32, device="cuda"),
            )
        return self.buffers[t_pad]

    def run(self, u, ids, dbg=None, t_pad=None, flags=0, xq=None, expert_map="map"):
        t_pad = t_pad or u.shape[0]
        rows, perm = self.out_buffers(t_pad)
        lay = self.layer
        torch.ops._muse_C.nvfp4_decode_moe(
            u, ids, lay.expert_map if expert_map == "map" else expert_map,
            self.w13, self.sf13, lay.w2, lay.sf2, lay.prm, lay.gs, self.up_first,
            rows, perm, self.hq_data, self.hq_sf, self.plan, self.ctl, dbg, flags,
            None if xq is None else xq[0], None if xq is None else xq[1],
        )  # fmt: skip
        return rows, perm

    def counters_are_zero(self) -> bool:
        return not bool(self.ctl.any())

    def plan_tables(self):
        p = self.plan[:708].cpu()
        n = int(p[0])
        return {
            "n_entries": n,
            "n_cols": int(p[1]),
            "expert": p[4:68][:n].tolist(),
            "cnt": p[68:132][:n].tolist(),
            "col0": p[132:196][:n].tolist(),
            "route": p[196:708].view(64, 8)[:n].tolist(),
        }

    def final_output(self, u, ids, weights):
        rows, perm = self.run(u, ids)
        t = u.shape[0]
        idx = perm.view(t, TOP_K)
        valid = idx >= 0
        r = rows[idx.clamp_min(0).long().view(-1)].view(t, TOP_K, H)
        return _finalize(r, weights, self.layer.gamma, valid)


@pytest.fixture(scope="module", autouse=True)
def _extension() -> None:
    muse_c.load()


@pytest.fixture(
    scope="module",
    params=[(diff_s, e) for e in E_LOCAL_CASES for diff_s in (True, False)],
    ids=lambda p: f"{'diff_s' if p[0] else 'same_s'}-e{p[1]}",
)
def layer(request) -> _Layer:
    diff_s, e_local = request.param
    with torch.inference_mode():
        yield _Layer(diff_s=diff_s, e_local=e_local)


def _routing(t: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(10_000 + seed)
    u = torch.randn(t, H, device="cuda", generator=g).to(torch.bfloat16)
    ids = (
        torch.rand(t, E_GLOBAL, device="cuda", generator=g)
        .topk(TOP_K, dim=1)
        .indices.to(torch.int32)
        .contiguous()
    )
    s = torch.sigmoid(torch.randn(t, TOP_K, device="cuda", generator=g) * 1.5)
    return u, ids, (s / s.sum(1, keepdim=True)).float()


def _local_ids(
    t: int, n_local: int | list[int], seed: int, e_local: int = E_LOCAL
) -> torch.Tensor:
    """Each token routes to exactly ``n_local`` local and 8 - n_local non-local
    experts (distinct), in random slot order; the rank owns ``e_local`` experts."""
    g = torch.Generator().manual_seed(1234 + seed)
    nl = [n_local] * t if isinstance(n_local, int) else n_local
    out = torch.empty(t, TOP_K, dtype=torch.int32)
    for i in range(t):
        loc = torch.randperm(e_local, generator=g)[: nl[i]]
        non = e_local + torch.randperm(E_GLOBAL - e_local, generator=g)[: TOP_K - nl[i]]
        row = torch.cat([loc, non])[torch.randperm(TOP_K, generator=g)]
        out[i] = row.to(torch.int32)
    return out.cuda()


def _ids_cases(t: int, seed: int, e_local: int = E_LOCAL):
    cases = {
        "random": _routing(t, seed)[1],
        "no_local": _local_ids(t, 0, seed, e_local),
        "one_local": _local_ids(t, 1, seed, e_local),
        "three_local": _local_ids(t, 3, seed, e_local),
        "all_local": _local_ids(t, 8, seed, e_local),
        # The first and last local expert of the rank.
        "first_last": torch.tensor(
            [
                [0, e_local - 1] + list(range(e_local + 6 * i, e_local + 6 + 6 * i))
                for i in range(t)
            ],
            dtype=torch.int32,
            device="cuda",
        ),
        # More routes than one entry holds: the expert is split over entries.
        "same_expert": torch.full((t, TOP_K), 5, dtype=torch.int32, device="cuda"),
    }
    return cases


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


# ------------------------------------------------------------------------------
# Activation quantization
# ------------------------------------------------------------------------------
def _stock_quant(u: torch.Tensor, gscale: float):
    """ops.scaled_fp4_experts_quant of T rows as the rows of expert 0; scales
    de-swizzled to linear [T, 256]."""
    t = u.shape[0]
    offsets = torch.zeros(E_LOCAL + 1, dtype=torch.int32, device="cuda")
    offsets[1:] = t
    block_offsets = torch.zeros_like(offsets)
    block_offsets[1:] = 128
    gs = torch.full((E_LOCAL,), gscale, dtype=torch.float32, device="cuda")
    q, s = ops.scaled_fp4_experts_quant(u, gs, offsets, block_offsets, TOP_K)
    flat = s.view(torch.uint8).flatten()
    r = torch.arange(t, device="cuda").unsqueeze(1)
    k = torch.arange(H // 16, device="cuda").unsqueeze(0)
    off = ((k >> 2) << 9) + ((r & 31) << 4) + (((r >> 5) & 3) << 2) + (k & 3)
    return q, flat[off]


@pytest.mark.parametrize("t", [1, 2, 4, 8])
@torch.inference_mode()
def test_activation_quant_is_byte_identical_to_the_stock_op(t: int) -> None:
    g = torch.Generator(device="cuda").manual_seed(5)
    base = torch.randn(t, H, device="cuda", generator=g)
    mixed = base.clone()
    mixed[:, ::3] = 0
    mixed[0, :16] = 0
    grid = torch.randint(-12, 13, (t, H), device="cuda", generator=g).float() * 0.25
    sign = torch.where(torch.arange(H, device="cuda") % 2 == 0, 1.0, -1.0)
    block_head = torch.arange(H, device="cuda") % 16 == 0
    ties = (sign * (0.75 + 0.5 * block_head)).expand(t, H)
    cases = {
        "randn": (base, 0.9),
        "large": (base * 1e3, 0.9),
        "small": (base * 1e-3, 0.9),
        "zeros": (mixed, 0.9),
        "all_zero": (torch.zeros_like(base), 0.9),
        # The block scale saturates at e4m3 max.
        "saturating": ((base * 50 / 0.9 * 6 * 448 / 3).clamp(-3e38, 3e38), 0.9),
        "subnormal": (base * 1e-39, 0.9),
        "grid_ties": (grid, 0.9),
        "sign_ties": (ties, 0.9),
        "other_scale": (base, 0.37),
    }
    for name, (x, gscale) in cases.items():
        x = x.to(torch.bfloat16).contiguous()
        q_ref, sf_ref = _stock_quant(x, gscale)
        data = torch.empty(t, H // 2, dtype=torch.uint8, device="cuda")
        sf = torch.empty(t, H // 16, dtype=torch.uint8, device="cuda")
        torch.ops._muse_C.nvfp4_decode_act_quant(
            x, torch.tensor([gscale], dtype=torch.float32, device="cuda"), data, sf
        )
        assert torch.equal(data, q_ref), name
        assert torch.equal(sf, sf_ref), name


# ------------------------------------------------------------------------------
# Stages against the CUTLASS composition: with the exact GEMM1 (the gate and up
# halves as groups of their own), whatever the scales; with the fold of s3 / s1
# into the intermediate scale only where it is the identity (s3 == s1)
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("exact", [True, False], ids=["exact", "folded"])
@pytest.mark.parametrize("up_first", [False, True])
@pytest.mark.parametrize("t", [1, 4, 8])
@torch.inference_mode()
def test_stages_match_the_cutlass_composition(
    layer: _Layer, t: int, up_first: bool, exact: bool
) -> None:
    if layer.diff_s and not exact:
        pytest.skip("the fold rounds the up half to bf16 in the gate's scale")
    u, ids, _ = _routing(t, 3)
    engine = _Engine(layer, up_first)
    rows, perm = engine.run(u, ids)
    plan = engine.plan_tables()

    e = layer.e_local
    local_ids = remap_topk_to_local(ids, layer.expert_map, out_dtype=torch.int32)
    eo = torch.empty(e + 1, dtype=torch.int32, device="cuda")
    bo = torch.empty_like(eo)
    ps1 = torch.empty(e, 3, dtype=torch.int32, device="cuda")
    ps2 = torch.empty_like(ps1)
    a_map = torch.empty(t * TOP_K, dtype=torch.int32, device="cuda")
    c_map = torch.empty_like(a_map)
    ops.get_cutlass_moe_mm_data(
        local_ids, eo, ps1, ps2, a_map, c_map, e, H, H, bo, is_gated=True
    )
    a_gs = torch.full((e,), float(1 / layer.a13), device="cuda")
    aq, asf = ops.scaled_fp4_experts_quant(
        ops.shuffle_rows(u, a_map), a_gs, eo, bo, TOP_K
    )
    a2_gs = torch.full((e,), float(1 / layer.a2), device="cuda")
    if exact:
        aq2, eo2, bo2, ps = muse_c.split_gemm1_prepare(aq, eo, bo, ps1)
        alphas = torch.stack((layer.a13 * layer.s1, layer.a13 * layer.s3), 1)
        c1 = torch.empty(2 * t * TOP_K, H, dtype=torch.bfloat16, device="cuda")
        ops.cutlass_fp4_moe_mm(
            c1,
            aq2,
            layer.w13.view(2 * e, H, H // 2),
            asf,
            layer.sf13.view(2 * e, H, H // 16),
            alphas.flatten().float().contiguous(),
            ps,
            eo2,
            bo2,
        )
        iq, isf = muse_c.silu_and_mul_split_scaled_fp4_experts_quant(
            c1, a2_gs, eo, bo, TOP_K, True
        )
    else:
        c1 = torch.empty(t * TOP_K, 2 * H, dtype=torch.bfloat16, device="cuda")
        g1 = (layer.a13 * layer.s1).float().contiguous()
        ops.cutlass_fp4_moe_mm(
            c1, aq, layer.w13, asf, layer.sf13, g1, ps1, eo[:-1], bo[:-1]
        )
        iq, isf = ops.silu_and_mul_scaled_fp4_experts_quant(c1, a2_gs, eo, bo, TOP_K)
    c3 = torch.empty(t * TOP_K, H, dtype=torch.bfloat16, device="cuda")
    g2 = (layer.a2 * layer.s2).float().contiguous()
    ops.cutlass_fp4_moe_mm(c3, iq, layer.w2, isf, layer.sf2, g2, ps2, eo[:-1], bo[:-1])

    isf_bytes = isf.view(torch.uint8).flatten()
    lids = local_ids.view(-1)
    n_routes = 0
    for g_idx in range(plan["n_entries"]):
        expert = plan["expert"][g_idx]
        for n in range(plan["cnt"][g_idx]):
            route = plan["route"][g_idx][n]
            assert int(lids[route]) == expert
            crow = int(c_map[route])
            row_in_expert = crow - int(eo[expert])
            col = plan["col0"][g_idx] + n
            # Intermediate codes and scales (swizzled in the CUTLASS tensor).
            assert torch.equal(engine.hq_data[col], iq[crow])
            k = torch.arange(H // 16, device="cuda")
            r = row_in_expert
            off = (
                (int(bo[expert]) * (H // 16))
                + (((r >> 7) * 64 + (k >> 2)) << 9)
                + ((r & 31) << 4)
                + (((r >> 5) & 3) << 2)
                + (k & 3)
            )
            assert torch.equal(engine.hq_sf[col], isf_bytes[off])
            # GEMM2 rows.
            got, want = rows[route].view(torch.int16), c3[crow].view(torch.int16)
            assert torch.equal(got, want)
            n_routes += 1
    assert n_routes == int((perm >= 0).sum())


# ------------------------------------------------------------------------------
# Accuracy against the exact fake-quantization reference
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("up_first", [False, True])
@pytest.mark.parametrize("t", [1, 2, 4, 8])
@torch.inference_mode()
def test_output_matches_the_fake_quant_reference(
    layer: _Layer, t: int, up_first: bool
) -> None:
    engine = _Engine(layer, up_first)
    u, _, weights = _routing(t, 7)
    for name, ids in _ids_cases(t, 11, layer.e_local).items():
        out = engine.final_output(u, ids, weights)
        key = (t, name)  # the references do not depend on the weight layout
        if key not in layer.references:
            layer.references[key] = layer.reference_rows(u, ids, "fq")
        rows_ref, valid = layer.references[key]
        ref = _finalize(rows_ref, weights, layer.gamma, valid)
        assert not torch.isnan(out).any(), name
        if valid.any():
            bound = FQ_REL_L2_FEW_ROWS if name == "same_expert" else FQ_REL_L2
            assert _rel_l2(out, ref) <= bound, (name, _rel_l2(out, ref))
            if name in ("random", "three_local"):
                bf16_rows, _ = layer.reference_rows(u, ids, "bf16")
                bf16 = _finalize(bf16_rows, weights, layer.gamma, valid)
                assert _rel_l2(out, bf16) <= BF16_REL_L2, name
        else:
            assert out.abs().max() == 0, name


@torch.inference_mode()
def test_plan_and_route_map(layer: _Layer) -> None:
    engine = _Engine(layer)
    t = 8
    u, ids, _ = _routing(t, 21)
    _, perm = engine.run(u, ids)
    local = (layer.expert_map[ids.long()] >= 0).view(-1)
    expected = torch.where(local, torch.arange(t * TOP_K, device="cuda"), -1)
    assert torch.equal(perm.long(), expected)
    plan = engine.plan_tables()
    assert plan["n_cols"] == int(local.sum())
    seen = sorted(
        r
        for g in range(plan["n_entries"])
        for r in plan["route"][g][: plan["cnt"][g]]
    )
    assert seen == local.nonzero().flatten().tolist()
    ids_flat = ids.view(-1).long()
    for g in range(plan["n_entries"]):
        for n in range(plan["cnt"][g]):
            route = plan["route"][g][n]
            assert int(layer.expert_map[ids_flat[route]]) == plan["expert"][g]


# ------------------------------------------------------------------------------
# Determinism, CUDA graphs, safety
# ------------------------------------------------------------------------------
def _valid(rows: torch.Tensor, perm: torch.Tensor):
    return rows[perm >= 0].clone().view(torch.int16), perm.clone()


@pytest.mark.parametrize("t", [1, 2, 4, 8])
@torch.inference_mode()
def test_bitwise_deterministic_eager_and_graph(layer: _Layer, t: int) -> None:
    engine = _Engine(layer)
    inputs = []
    for i in range(6):
        u, ids, _ = _routing(t, 40 + i)
        if i % 2:
            ids = _local_ids(
                t, [1 + (i + k) % 5 for k in range(t)], 60 + i, layer.e_local
            )
        inputs.append((u, ids))
    ref = [_valid(*engine.run(u, ids)) for u, ids in inputs]
    for i in range(50):
        got = _valid(*engine.run(*inputs[i % 2]))
        assert torch.equal(got[0], ref[i % 2][0]) and torch.equal(got[1], ref[i % 2][1])

    u_s, ids_s = inputs[0][0].clone(), inputs[0][1].clone()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        engine.run(u_s, ids_s)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rows, perm = engine.run(u_s, ids_s)
    for i in range(60):
        k = i % len(inputs)
        u_s.copy_(inputs[k][0])
        ids_s.copy_(inputs[k][1])
        graph.replay()
        got = _valid(rows, perm)
        assert torch.equal(got[0], ref[k][0]) and torch.equal(got[1], ref[k][1])


@torch.inference_mode()
def test_poisoned_scratch_and_unused_experts_do_not_matter(layer: _Layer) -> None:
    engine = _Engine(layer)
    t = 4
    u, _, _ = _routing(t, 80)
    ids = _local_ids(t, [2, 1, 3, 0], 81, layer.e_local)
    base = _valid(*engine.run(u, ids))
    rows, perm = engine.out_buffers(t)
    rows.fill_(float("nan"))
    engine.hq_data.fill_(0xFF)
    engine.hq_sf.fill_(0xFF)
    engine.run(u, ids)
    nonlocal_rows = rows[perm < 0]
    assert torch.isnan(nonlocal_rows.float()).all()  # untouched
    got = _valid(rows, perm)
    assert torch.equal(got[0], base[0]) and torch.equal(got[1], base[1])
    # Unused experts' weights are never read.
    # The weights and the block scales of unused experts are never read either.
    used = set(engine.plan_tables()["expert"])
    saved = {}
    for e in range(layer.e_local):
        if e not in used:
            saved[e] = (
                layer.w13[e].clone(),
                layer.w2[e].clone(),
                engine.sf13[e].clone(),
                layer.sf2[e].clone(),
            )
            layer.w13[e].fill_(0xFF)
            layer.w2[e].fill_(0xFF)
            engine.sf13[e].view(torch.uint8).fill_(0xFF)  # e4m3 NaN
            layer.sf2[e].view(torch.uint8).fill_(0xFF)
    try:
        got = _valid(*engine.run(u, ids))
    finally:
        for e, (a, b, c, d) in saved.items():
            layer.w13[e].copy_(a)
            layer.w2[e].copy_(b)
            engine.sf13[e].copy_(c)
            layer.sf2[e].copy_(d)
    assert torch.equal(got[0], base[0])
    assert engine.counters_are_zero()


@torch.inference_mode()
def test_token_rows_do_not_depend_on_the_other_tokens(layer: _Layer) -> None:
    engine = _Engine(layer)
    u, _, _ = _routing(8, 90)
    ids = _local_ids(8, [3, 2, 4, 5, 1, 6, 2, 3], 91, layer.e_local)
    alone = engine.run(u[:1].contiguous(), ids[:1].contiguous())
    one = alone[0][:8][alone[1][:8] >= 0].clone().view(torch.int16)
    for t in (2, 4, 8):
        rows, perm = engine.run(u[:t].contiguous(), ids[:t].contiguous())
        got = rows[:8][perm[:8] >= 0].view(torch.int16)
        assert torch.equal(got, one), t


@pytest.mark.parametrize(("t", "t_pad"), [(1, 2), (3, 4), (5, 8), (7, 8)])
@torch.inference_mode()
def test_fewer_tokens_than_the_padded_count(layer: _Layer, t: int, t_pad: int) -> None:
    engine = _Engine(layer)
    u, ids, weights = _routing(t, 110)
    rows, perm = engine.run(u, ids, t_pad=t_pad)
    assert rows.shape[0] == t_pad * TOP_K
    local = (layer.expert_map[ids.long()] >= 0).view(-1)
    expected = torch.full((t_pad * TOP_K,), -1, dtype=torch.long, device="cuda")
    routes = torch.arange(t * TOP_K, device="cuda")
    expected[: t * TOP_K] = torch.where(local, routes, -1)
    assert torch.equal(perm.long(), expected)
    idx = perm[: t * TOP_K].view(t, TOP_K)
    out = _finalize(
        rows[idx.clamp_min(0).long().view(-1)].view(t, TOP_K, H),
        weights, layer.gamma, idx >= 0,
    )  # fmt: skip
    rows_ref, valid = layer.reference_rows(u, ids, "fq")
    ref = _finalize(rows_ref, weights, layer.gamma, valid)
    assert _rel_l2(out, ref) <= FQ_REL_L2


@pytest.mark.parametrize("t", [1, 4])
@torch.inference_mode()
def test_identity_expert_map(layer: _Layer, t: int) -> None:
    """Without an expert map (TP) the global ids are the local ones."""
    engine = _Engine(layer)
    u, _, _ = _routing(t, 120)
    ids = _local_ids(t, 8, 121, layer.e_local)
    e_local = layer.e_local
    ids = torch.where(ids < e_local, ids, ids % e_local).to(torch.int32).contiguous()
    # layer.expert_map is the identity on the local experts.
    with_map = _valid(*engine.run(u, ids))
    rows, perm = engine.out_buffers(t)
    torch.ops._muse_C.nvfp4_decode_moe(
        u, ids, None, engine.w13, engine.sf13, layer.w2, layer.sf2, layer.prm,
        layer.gs, False, rows, perm, engine.hq_data, engine.hq_sf, engine.plan,
        engine.ctl, None,
    )  # fmt: skip
    without_map = _valid(rows, perm)
    assert torch.equal(with_map[0], without_map[0])
    assert torch.equal(with_map[1], without_map[1])


@torch.inference_mode()
def test_no_host_synchronization(layer: _Layer) -> None:
    engine = _Engine(layer)
    u, ids, _ = _routing(2, 100)
    engine.run(u, ids)
    torch.cuda.set_sync_debug_mode("error")
    try:
        engine.run(u, ids)
    finally:
        torch.cuda.set_sync_debug_mode("default")


# ------------------------------------------------------------------------------
# Routes without work, -1 ids, counters, launch variants
# ------------------------------------------------------------------------------
FLAG_TWO_LAUNCH = 1
FLAG_CTL_MEMSET = 2
FLAG_ZERO_NON_LOCAL = 4
FLAG_NO_PDL = 8


@pytest.mark.parametrize("t", [1, 2, 4, 8])
@torch.inference_mode()
def test_no_local_route_leaves_everything_untouched(layer: _Layer, t: int) -> None:
    """n_entries = 0: the kernel exits early; perm_idx is all -1, no row is
    written and the counters of the persistent kernel stay zero."""
    engine = _Engine(layer)
    u, _, _ = _routing(t, 130)
    ids = _local_ids(t, 0, 131, layer.e_local)
    rows, perm = engine.out_buffers(t)
    rows.fill_(float("nan"))
    perm.fill_(7)
    for flags in (0, FLAG_TWO_LAUNCH):
        engine.run(u, ids, flags=flags)
        assert bool((perm == -1).all())
        assert bool(torch.isnan(rows.float()).all())
        assert engine.plan_tables()["n_entries"] == 0
        assert engine.counters_are_zero()
    # The debug flag zero-fills the rows of non-local routes.
    engine.run(u, ids, flags=FLAG_ZERO_NON_LOCAL)
    assert bool((rows.view(torch.int16) == 0).all())


@pytest.mark.parametrize("t", [1, 2, 4, 8])
@torch.inference_mode()
def test_negative_and_unmapped_ids_are_non_local_routes(layer: _Layer, t: int) -> None:
    """Padding routes (-1), ids beyond the map and ids the map sends to -1 give
    no work, perm_idx = -1 and leave the other rows unchanged."""
    engine = _Engine(layer)
    u, _, _ = _routing(t, 140)
    ids = _local_ids(t, [3] * t, 141, layer.e_local)
    base = _valid(*engine.run(u, ids))
    bad = ids.clone()
    non_local = (layer.expert_map[ids.long()] < 0).view(-1).nonzero().flatten()
    assert non_local.numel() >= 3
    flat = bad.view(-1)
    flat[non_local[0]] = -1
    flat[non_local[1]] = E_GLOBAL + 5
    flat[non_local[2]] = 2**31 - 1
    rows, perm = engine.out_buffers(t)
    rows[perm < 0] = float("nan")
    got = _valid(*engine.run(u, bad))
    assert torch.equal(got[0], base[0]) and torch.equal(got[1], base[1])
    assert bool(torch.isnan(rows[perm < 0].float()).all())
    assert engine.counters_are_zero()
    # Every route -1 (a fully padded batch).
    engine.run(u, torch.full_like(ids, -1))
    assert bool((perm == -1).all())
    assert engine.counters_are_zero()


@pytest.mark.parametrize("t", [1, 2, 4, 8])
@torch.inference_mode()
def test_counters_are_zero_after_every_launch(layer: _Layer, t: int) -> None:
    engine = _Engine(layer)
    u, _, _ = _routing(t, 150)
    for name, ids in _ids_cases(t, 151, layer.e_local).items():
        for _ in range(2):
            engine.run(u, ids)
            assert engine.counters_are_zero(), name


@pytest.mark.parametrize("t", [1, 4, 8])
@torch.inference_mode()
def test_launch_variants_are_bitwise_equal(layer: _Layer, t: int) -> None:
    """Plan B (two launches), the memset node, any grid size (so fewer CTAs than
    the work, or than the SMs) and the stock activation quantization all give
    the persistent kernel's rows."""
    engine = _Engine(layer)
    u, _, _ = _routing(t, 160)
    gs0 = layer.gs[:1].contiguous()
    xq_data, xq_sf = ops.scaled_fp4_quant(u, gs0, is_sf_swizzled_layout=False)
    xq = (xq_data.view(torch.uint8).contiguous(), xq_sf.view(torch.uint8).contiguous())
    for name, ids in _ids_cases(t, 161, layer.e_local).items():
        ref = _valid(*engine.run(u, ids))
        variants = {
            "two_launch": dict(flags=FLAG_TWO_LAUNCH),
            "two_launch_no_pdl": dict(flags=FLAG_TWO_LAUNCH | FLAG_NO_PDL),
            "no_pdl": dict(flags=FLAG_NO_PDL),
            "memset": dict(flags=FLAG_CTL_MEMSET),
            "memset_no_pdl": dict(flags=FLAG_CTL_MEMSET | FLAG_NO_PDL),
            "grid_1": dict(flags=1 << 16),
            "grid_7": dict(flags=7 << 16),
            "grid_400": dict(flags=400 << 16),
            "stock_quant": dict(xq=xq),
            "stock_quant_two_launch": dict(xq=xq, flags=FLAG_TWO_LAUNCH),
        }
        for vname, kw in variants.items():
            got = _valid(*engine.run(u, ids, **kw))
            assert torch.equal(got[0], ref[0]), (name, vname)
            assert torch.equal(got[1], ref[1]), (name, vname)
            assert engine.counters_are_zero(), (name, vname)
        rows, perm = engine.out_buffers(t)
        rows.fill_(float("nan"))
        engine.run(u, ids, flags=FLAG_ZERO_NON_LOCAL)
        assert torch.equal(rows[perm >= 0].view(torch.int16), ref[0])
        assert bool((rows[perm < 0].view(torch.int16) == 0).all())
        for grid in (1, 5):  # fewer CTAs than routes
            rows.fill_(float("nan"))
            engine.run(u, ids, flags=FLAG_ZERO_NON_LOCAL | (grid << 16))
            assert torch.equal(rows[perm >= 0].view(torch.int16), ref[0])
            assert bool((rows[perm < 0].view(torch.int16) == 0).all())


@pytest.mark.parametrize("t", [1, 2, 4, 8])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_persistent_kernel_waits_for_the_routing_and_the_norm(
    layer: _Layer, t: int, use_graph: bool, capfd
) -> None:
    """The kernel is a programmatic dependent of the routing kernel: it may start
    while the top-k ids and the activations are still being written, and must wait
    before it reads either (``ids`` and ``u`` start as other valid inputs and are
    both written late, by one producer). Without the wait the CTAs build different
    plans and the kernel hangs: ``synchronize_or_exit`` ends the process then."""
    engine = _Engine(layer)
    u, _, _ = _routing(t, 170)
    ids = _ids_cases(t, 171)["three_local"]
    ref = _valid(*engine.run(u, ids, flags=FLAG_NO_PDL))
    stale_u, _, _ = _routing(t, 172)
    stale_ids = _ids_cases(t, 173)["all_local"]

    u_in, ids_in = stale_u.clone(), stale_ids.clone()

    def step():
        slow_copy((u_in, u), (ids_in, ids))
        return engine.run(u_in, ids_in)

    def restore():
        u_in.copy_(stale_u)
        ids_in.copy_(stale_ids)

    restore()
    step()  # compiles the Triton producer
    synchronize_or_exit("the first launch", capfd)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rows, perm = step()
        for _ in range(4):
            restore()
            graph.replay()
            synchronize_or_exit("a graph replay", capfd)
            got = _valid(rows, perm)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    else:
        for _ in range(4):
            restore()
            rows, perm = step()
            synchronize_or_exit("a launch", capfd)
            got = _valid(rows, perm)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    assert engine.counters_are_zero()
