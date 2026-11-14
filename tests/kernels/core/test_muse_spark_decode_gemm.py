# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage for the skinny decode GEMM (``_muse_C::decode_gemm``).

The kernel computes ``x @ w.T`` for 1 .. 128 rows with fp32 accumulation, K split over a
thread block cluster, and adds the partial tiles in rank order
(``muse_spark_decode_gemm.cu``). The order of the additions depends on the K split of
the config only. The tests check that

* the product is as close to the exact one as the cuBLAS product (in fp64 ulps of bf16),
* a row gets the same bits at every row count and for every tile shape of one split, and
  in every run, eager and in a CUDA graph (the claim of the file header),
* strided operands, a launch as a programmatic dependent and the Python glue work,
* the arguments are validated.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.models.muse_spark.nvidia.ops import decode_gemm as dg
from vllm.models.muse_spark.nvidia.ops import muse_c
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

_unusable = muse_c.preflight()
if _unusable is not None:
    pytest.skip(
        f"The _muse_C extension cannot be built: {_unusable}", allow_module_level=True
    )

DEVICE = torch.device("cuda")
# (N, K) of the TP8 pooler, unpooler and QKV projection of a rank.
SHAPES = {"pooler": (512, 8192), "unpooler": (1024, 4096), "qkv": (2304, 8192)}


@pytest.fixture(scope="module")
def op():
    muse_c.load()
    return torch.ops._muse_C.decode_gemm


@pytest.fixture(scope="module")
def configs(op):
    n = torch.ops._muse_C.decode_gemm_num_configs()
    return [tuple(torch.ops._muse_C.decode_gemm_config(i)) for i in range(n)]


@pytest.fixture(scope="module")
def scratch():
    return torch.empty(4 << 20, dtype=torch.float32, device=DEVICE)


def cfg_index(configs, mt, bn, wm, wn, stages, split):
    return configs.index((mt, bn, wm, wn, stages, split))


def data(rows, n, k, seed=0, wide=False):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, k, device=DEVICE, generator=g)
    if wide:  # activations over several binades, like the output of a norm and a gate
        x = x * torch.exp2(torch.randint(-3, 4, (k,), device=DEVICE, generator=g))
    w = torch.randn(n, k, device=DEVICE, generator=g) * 0.02
    return x.bfloat16(), w.bfloat16()


def run(op, scratch, x, w, cfg=-1, early_w=False, pdl=False):
    out = torch.full((x.shape[0], w.shape[0]), float("nan"), device=DEVICE)
    out = out.bfloat16()
    op(x, w, out, scratch, cfg, early_w, pdl)
    return out


def bf16_ulp(ref):
    """The spacing of bf16 at each exact value."""
    exp = torch.floor(torch.log2(ref.abs().clamp_min(2.0**-100)))
    return torch.exp2(exp - 7)


def error_in_ulps(got, x, w):
    ref = x.double() @ w.double().t()
    return (got.double() - ref).abs() / bf16_ulp(ref), ref


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("rows", [1, 4, 8, 9, 16, 24, 33, 48, 64, 100, 128])
@pytest.mark.parametrize("wide", [False, True])
def test_product_is_correctly_rounded_up_to_the_fp32_accumulation(
    op, scratch, name, rows, wide
):
    """|got - exact| <= half a bf16 step + the error bound of an fp32 accumulation."""
    n, k = SHAPES[name]
    x, w = data(rows, n, k, seed=rows, wide=wide)
    got = run(op, scratch, x, w)
    assert not torch.isnan(got.float()).any()
    err, ref = error_in_ulps(got, x, w)
    abs_sum = x.double().abs() @ w.double().abs().t()
    slack = 4e-6 * abs_sum / bf16_ulp(ref)
    assert bool((err <= 0.5 + slack).all()), float((err - slack).max())
    # The bulk is rounded correctly (the rest sits within the slack of a midpoint).
    assert float((err < 0.5).float().mean()) >= 0.98


@pytest.mark.parametrize("name", list(SHAPES))
@pytest.mark.parametrize("rows", [1, 16, 33, 64])
def test_run_to_run_and_graph_bits_are_the_same(op, scratch, name, rows):
    n, k = SHAPES[name]
    x, w = data(rows, n, k, seed=3)
    first = run(op, scratch, x, w)
    for _ in range(5):
        assert torch.equal(run(op, scratch, x, w), first)
    out = torch.empty_like(first)
    # An unrelated tensor in the scratch between the runs: nothing is carried over.
    scratch.fill_(float("nan"))
    graph = torch.cuda.CUDAGraph()
    op(x, w, out, scratch, -1, False, False)  # warm-up
    with torch.cuda.graph(graph):
        op(x, w, out, scratch, -1, False, False)
    out.zero_()
    graph.replay()
    assert torch.equal(out, first)


def test_bits_do_not_depend_on_rows_or_tile_shape(op, configs, scratch):
    """One K split, any tile: a row is the same bits whatever the batch around it."""
    cases = {
        "pooler": [
            (1, 32, 1, 4, 4, 8),
            (1, 32, 1, 2, 5, 8),
            (2, 32, 2, 2, 4, 8),
            (3, 32, 3, 2, 4, 8),
            (4, 32, 4, 1, 4, 8),
            (4, 32, 2, 2, 5, 8),
            (6, 32, 2, 2, 4, 8),
            (8, 32, 4, 1, 4, 8),
        ],
        "unpooler": [
            (1, 32, 1, 4, 4, 4),
            (2, 32, 2, 2, 4, 4),
            (3, 32, 1, 4, 4, 4),
            (4, 32, 4, 1, 4, 4),
            (6, 32, 3, 2, 4, 4),
            (8, 32, 4, 1, 4, 4),
            (8, 32, 2, 2, 4, 4),
        ],
        "qkv": [
            (1, 64, 1, 4, 4, 4),
            (2, 64, 2, 4, 4, 4),
            (2, 32, 2, 2, 4, 4),
            (4, 64, 2, 4, 4, 4),
            (4, 64, 4, 2, 4, 4),
            (8, 32, 4, 1, 4, 4),
        ],
        # The QKV projection of 33 .. 64 rows has a split of 8.
        "qkv ": [
            (2, 128, 2, 4, 3, 8),
            (3, 128, 3, 4, 3, 8),
            (4, 128, 2, 4, 3, 8),
            (4, 128, 4, 2, 3, 8),
            (6, 32, 2, 2, 4, 8),
            (8, 32, 4, 1, 4, 8),
        ],
    }
    for name, cfgs in cases.items():
        n, k = SHAPES[name.strip()]
        x, w = data(128, n, k, seed=11, wide=True)
        reference = None
        for cfg in cfgs:
            mt = cfg[0]
            rows = min(128, mt * 16)
            for r in {rows, max(1, rows - 5), max(1, rows // 2 + 1)}:
                if r > mt * 16:
                    continue
                got = run(op, scratch, x[:r].contiguous(), w, cfg_index(configs, *cfg))
                if reference is None:
                    reference = run(op, scratch, x, w, cfg_index(configs, *cfgs[-1]))
                assert torch.equal(got, reference[:r]), (name, cfg, r)


def test_the_k_split_changes_the_rounding_only(op, configs, scratch):
    n, k = SHAPES["pooler"]
    x, w = data(16, n, k, seed=5, wide=True)
    outs = {}
    for split, cfg in {
        1: (2, 32, 2, 2, 4, 1),
        2: (1, 32, 1, 4, 4, 2),
        4: (4, 32, 4, 1, 4, 4),
        8: (2, 32, 2, 2, 4, 8),
    }.items():
        outs[split] = run(op, scratch, x, w, cfg_index(configs, *cfg))
        err, ref = error_in_ulps(outs[split], x, w)
        abs_sum = x.double().abs() @ w.double().abs().t()
        assert bool((err <= 1.0 + 4e-6 * abs_sum / bf16_ulp(ref)).all()), split
    # Not all the same bits (the order of the additions is a function of the split) ...
    assert not torch.equal(outs[1], outs[8])
    # ... and within two bf16 steps of each other where the values are not cancelled.
    big = (x.double() @ w.double().t()).abs() > 0.05
    diff = (outs[1].double() - outs[8].double()).abs()
    assert bool((diff <= 2 * bf16_ulp(outs[1].double()))[big].all())


def test_tuned_choice_is_the_explicit_one(op, configs, scratch):
    """cfg = -1 picks the config of the table; its bits are those of that config."""
    for name, (n, k) in SHAPES.items():
        for rows in (1, 16, 24, 40, 64):
            x, w = data(rows, n, k, seed=rows)
            auto = run(op, scratch, x, w)
            mt = (rows + 15) // 16
            split = 8 if name == "pooler" or (name == "qkv" and rows > 32) else 4
            same = [
                c
                for c in configs
                if c[5] == split and c[0] >= mt and n % c[1] == 0
            ]
            assert same
            for c in same[:3]:
                assert torch.equal(run(op, scratch, x, w, configs.index(c)), auto), c


def test_strided_operands_and_programmatic_launch(op, scratch):
    n, k = SHAPES["pooler"]
    rows = 40
    x, w = data(rows, n, k, seed=2)
    expected = run(op, scratch, x, w)
    # x is a view of a wider row (ldx > K), out a view of a wider row.
    wide_x = torch.zeros(rows, k + 64, dtype=torch.bfloat16, device=DEVICE)
    wide_x[:, :k] = x
    wide_out = torch.zeros(rows, n + 32, dtype=torch.bfloat16, device=DEVICE)
    op(wide_x[:, :k], w, wide_out[:, :n], scratch, -1, False, False)
    assert torch.equal(wide_out[:, :n], expected)
    assert not wide_out[:, n:].any()
    # A programmatic dependent behind a kernel that writes x, with the early weights.
    for early_w in (False, True):
        for use_graph in (False, True):
            src = x.clone()
            buf = torch.zeros_like(x)
            out = torch.zeros_like(expected)

            def step():
                buf.copy_(src)  # the producer of x
                op(buf, w, out, scratch, -1, early_w, True)

            step()
            if use_graph:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    step()
                out.zero_()
                buf.zero_()
                graph.replay()
            assert torch.equal(out, expected), (early_w, use_graph)


def test_rows_beyond_the_valid_ones_are_not_written(op, scratch):
    n, k = SHAPES["unpooler"]
    x, w = data(24, n, k, seed=8)
    out = torch.full((40, n), 7.0, dtype=torch.bfloat16, device=DEVICE)
    op(x, w, out[:24], scratch, -1, False, False)
    assert bool((out[24:] == 7.0).all())


def test_rejects_bad_arguments(op, configs, scratch):
    n, k = SHAPES["pooler"]
    x, w = data(16, n, k)
    out = torch.empty(16, n, dtype=torch.bfloat16, device=DEVICE)

    def call(x=x, w=w, out=out, scratch=scratch, cfg=-1):
        op(x, w, out, scratch, cfg, False, False)

    call()
    with pytest.raises(RuntimeError, match="bf16"):
        call(x=x.float())
    with pytest.raises(RuntimeError, match="1 <= T <= 128"):
        d, _ = data(129, n, k)
        call(x=d, out=torch.empty(129, n, dtype=torch.bfloat16, device=DEVICE))
    with pytest.raises(RuntimeError, match="aligned"):
        flat = torch.zeros(16 * k + 8, dtype=torch.bfloat16, device=DEVICE)
        call(x=flat[1 : 1 + 16 * k].view(16, k))
    with pytest.raises(RuntimeError, match="scratch"):
        call(scratch=torch.empty(10, dtype=torch.float32, device=DEVICE))
    with pytest.raises(RuntimeError, match="scratch"):
        call(scratch=torch.empty(4 << 20, dtype=torch.bfloat16, device=DEVICE))
    with pytest.raises(RuntimeError, match="rows"):
        # 16 rows per tile, 33 asked for.
        d, _ = data(33, n, k)
        call(
            x=d,
            out=torch.empty(33, n, dtype=torch.bfloat16, device=DEVICE),
            cfg=cfg_index(configs, 1, 32, 1, 4, 4, 8),
        )
    with pytest.raises(RuntimeError, match="do not fit"):
        # N = 520 is not a multiple of 32.
        call(
            w=torch.empty(520, k, dtype=torch.bfloat16, device=DEVICE),
            out=torch.empty(16, 520, dtype=torch.bfloat16, device=DEVICE),
            cfg=cfg_index(configs, 1, 32, 1, 4, 4, 8),
        )
    with pytest.raises(RuntimeError, match="config"):
        call(cfg=len(configs))
    with pytest.raises(RuntimeError, match="dense"):
        call(w=torch.empty(n, k + 8, dtype=torch.bfloat16, device=DEVICE)[:, :k])


# The Python glue


def test_usable_and_supports_weight():
    muse_c.load()
    dg.reserve_scratch(DEVICE)
    n, k = SHAPES["qkv"]
    x, w = data(16, n, k)
    assert dg.usable(x) and dg.supports_weight(w)
    assert not dg.usable(torch.cat([x, x, x, x, x])[:65].contiguous())  # 65 rows
    assert dg.usable(x[:1])
    assert not dg.usable(x.float())
    assert not dg.supports_weight(w.t())  # not [N, K] row-major
    assert not dg.supports_weight(w.float())


def test_decode_gemm_matches_the_op_and_wraps_the_projection():
    muse_c.load()
    dg.reserve_scratch(DEVICE)
    calls = []

    def fallback(layer, x, weight, bias=None):
        calls.append(x.shape[0])
        return F.linear(x, weight, bias)

    gemm = dg.wrap_projection(fallback)
    n, k = SHAPES["pooler"]
    _, w = data(1, n, k)
    for rows in (1, 16, 64):
        x, _ = data(rows, n, k, seed=rows)
        got = gemm(None, x, w, None)
        assert torch.equal(got, dg.decode_gemm(x, w))
    assert calls == []
    x, _ = data(65, n, k)
    gemm(None, x, w, None)
    bias = torch.zeros(n, dtype=torch.bfloat16, device=DEVICE)
    gemm(None, x[:16].contiguous(), w, bias)
    assert calls == [65, 16]
