# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The plan of the large-batch muse decode kernel on the router's aux stream
(``VLLM_MUSE_SPARK_SM120_AUX_PLAN``), and its activations quantized by another launch,
in particular by the one of the pre-expert RMSNorm
(``VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT``).

The preparation launch of ``nvfp4_decode_moe_large`` (``k_prep``) has one CTA that
builds the plan and the route -> row map from the top-k ids, and one CTA per 32 rows
that quantizes the activations. Both halves can be left out of it: the plan by
``nvfp4_decode_moe_plan`` (flag 32 of the large-batch op), the quantization by
``nvfp4_decode_act_quant`` or any producer that writes the same bytes (flag 64). The
kernels of the tests below are required to give the same bytes in every combination, for
every token count of the large-batch kernel, and to wait for what they read.
"""

import contextlib
import math
from types import SimpleNamespace

import pytest
import torch

import vllm.envs as envs
import vllm.models.muse_spark.nvidia.model as model_ops
from tests.kernels.moe.test_muse_spark_sm120_decode_moe import (
    FLAG_NO_PDL,
    LARGE_DEFAULT_FLAGS,
    LARGE_TS,
    LOADERS,
    MAX_LARGE_T,
    TOP_K,
    H,
    _Engine,
    _ids_cases,
    _Layer,
    _quant_cases,
    _routing,
    _stock_quant,
    _valid,
    _zipf_ids,
)
from tests.kernels.muse_spark_pdl_utils import slow_copy, synchronize_or_exit
from vllm.models.muse_spark.common.ops.fused_rmsnorm import fused_rmsnorm_fwd
from vllm.models.muse_spark.nvidia import sm120_nvfp4_moe
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MuseSparkModelOptNvFp4FusedMoE,
)
from vllm.models.muse_spark.nvidia.ops import fused_boundary, muse_c, muse_decode_moe
from vllm.models.muse_spark.nvidia.ops.muse_decode_moe import (
    MUSE_SMALL_MAX_TOKENS,
    MuseDecodeMoE,
)
from vllm.models.muse_spark.nvidia.sm120_nvfp4_moe import MuseSparkSm120NvFp4Experts
from vllm.platforms import current_platform

if not current_platform.is_device_capability_family(120):
    pytest.skip("Requires an SM120 GPU", allow_module_level=True)

_unusable = muse_c.preflight()
if _unusable is not None:
    pytest.skip(
        f"The _muse_C extension cannot be built: {_unusable}", allow_module_level=True
    )

FLAG_EXTERNAL_PLAN = 32
FLAG_EXTERNAL_XQ = 64
ALL_LARGE_TS = list(range(MUSE_SMALL_MAX_TOKENS + 1, MAX_LARGE_T + 1))
DEVICE = torch.device("cuda")
E_LOCAL_SMALL = 8


@pytest.fixture(scope="module", autouse=True)
def _extension() -> None:
    muse_c.load()


@pytest.fixture(
    scope="module",
    params=[(True, 32), (False, 64)],
    ids=lambda p: f"{'diff_s' if p[0] else 'same_s'}-e{p[1]}",
)
def layer(request) -> _Layer:
    diff_s, e_local = request.param
    with torch.inference_mode():
        yield _Layer(diff_s=diff_s, e_local=e_local)


def _all_ids(t: int, seed: int, e_local: int) -> dict[str, torch.Tensor]:
    cases = _ids_cases(t, seed, e_local)
    # Hot experts: more routes than one entry holds, several entries per expert.
    cases["zipf_1"] = _zipf_ids(t, seed, e_local, 1.0)
    cases["zipf_2"] = _zipf_ids(t, seed + 1, e_local, 2.0, n_local=5)
    return cases


def _plan_region(engine: _Engine) -> torch.Tensor:
    return engine.plan.clone()


def _poison(engine: _Engine, t: int) -> None:
    engine.plan.fill_(0x5A5A5A5A)
    engine.large_perm.fill_(-7)
    for scratch in (engine.xq_data, engine.xq_sf):
        scratch.fill_(0xFF)


def _plan_op(engine: _Engine, ids: torch.Tensor, flags: int = 0) -> None:
    t = ids.shape[0]
    torch.ops._muse_C.nvfp4_decode_moe_plan(
        ids,
        engine.layer.expert_map,
        engine.layer.e_local,
        engine.large_perm[: t * TOP_K],
        engine.plan,
        flags,
    )


# ------------------------------------------------------------------------------
# The plan launch: the bytes of the preparation launch's plan, for every token count
# ------------------------------------------------------------------------------
@torch.inference_mode()
def test_plan_launch_writes_the_plan_and_the_route_map_of_the_prep_launch(
    layer: _Layer,
) -> None:
    engine = _Engine(layer)
    for t in ALL_LARGE_TS:
        u, _, _ = _routing(t, 300 + t)
        for name, ids in _all_ids(t, 301 + t, layer.e_local).items():
            _poison(engine, t)
            _, perm = engine.run(u, ids)
            ref_plan, ref_perm = _plan_region(engine), perm.clone()
            for flags in (0, FLAG_NO_PDL):
                _poison(engine, t)
                _plan_op(engine, ids, flags)
                got_plan = _plan_region(engine)
                got_perm = engine.large_perm[: t * TOP_K]
                assert torch.equal(got_plan, ref_plan), (t, name, flags)
                assert torch.equal(got_perm, ref_perm), (t, name, flags)


@torch.inference_mode()
def test_plan_launch_rejects_bad_arguments(layer: _Layer) -> None:
    engine = _Engine(layer)
    t = 12
    ids = _all_ids(t, 5, layer.e_local)["random"]
    perm = engine.large_perm[: t * TOP_K]
    emap = layer.expert_map
    op = torch.ops._muse_C.nvfp4_decode_moe_plan
    op(ids, emap, layer.e_local, perm, engine.plan, 0)
    with pytest.raises(RuntimeError):  # more tokens than the kernel serves
        big = _all_ids(MAX_LARGE_T + 1, 6, layer.e_local)["random"]
        op(big, emap, layer.e_local, engine.large_perm[: big.numel()], engine.plan, 0)
    with pytest.raises(RuntimeError):  # perm_idx of another size
        op(ids, emap, layer.e_local, engine.large_perm[: t * TOP_K - 8], engine.plan, 0)
    with pytest.raises(RuntimeError):  # too many experts for the plan
        op(ids, emap, 257, perm, engine.plan, 0)
    with pytest.raises(RuntimeError):  # no experts
        op(ids, emap, 0, perm, engine.plan, 0)
    with pytest.raises(RuntimeError):  # plan scratch too small
        op(ids, emap, layer.e_local, perm, engine.plan[:1024], 0)
    with pytest.raises(RuntimeError):  # ids of another dtype
        op(ids.long(), emap, layer.e_local, perm, engine.plan, 0)


# ------------------------------------------------------------------------------
# The large-batch kernel on an external plan and / or external activations
# ------------------------------------------------------------------------------
def _combo(engine: _Engine, u, ids, loader, plan, xq, *, flags=0, pdl_xq=False):
    """One large-batch launch with the plan and / or the quantization done by their own
    launches, on poisoned scratch."""
    t = u.shape[0]
    _poison(engine, t)
    engine.out_buffers(t)[0].fill_(float("nan"))
    if plan:
        _plan_op(engine, ids, flags & FLAG_NO_PDL)
    if xq:
        torch.ops._muse_C.nvfp4_decode_act_quant(
            u, engine.layer.gs, engine.xq_data[:t], engine.xq_sf[:t], pdl_xq
        )
    extra = (FLAG_EXTERNAL_PLAN if plan else 0) | (FLAG_EXTERNAL_XQ if xq else 0)
    # An external quantization means ``u`` is not read: hand over other activations.
    hidden = torch.full_like(u, float("nan")) if xq else u
    return engine.run(hidden, ids, flags=flags | extra, loader=loader)


@pytest.mark.parametrize("pdl", [True, False], ids=["pdl", "no_pdl"])
@LOADERS
@torch.inference_mode()
def test_rows_do_not_depend_on_who_built_the_plan_and_the_quantization(
    layer: _Layer, loader: int, pdl: bool
) -> None:
    engine = _Engine(layer)
    base = 0 if pdl else FLAG_NO_PDL
    for t in sorted({*LARGE_TS, 10, 17, 40, 96, 127}):
        u, _, _ = _routing(t, 320 + t)
        for name, ids in _all_ids(t, 321 + t, layer.e_local).items():
            ref = _valid(*engine.run(u, ids, flags=FLAG_NO_PDL, loader=loader))
            engine.xq_data.fill_(0xFF)
            for plan, xq in (
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ):
                got = _valid(
                    *_combo(engine, u, ids, loader, plan, xq, flags=base, pdl_xq=pdl)
                )
                assert torch.equal(got[0], ref[0]), (t, name, plan, xq)
                assert torch.equal(got[1], ref[1]), (t, name, plan, xq)
            assert engine.counters_are_zero(), (t, name)


@torch.inference_mode()
def test_every_token_count_takes_an_external_plan_and_quantization(
    layer: _Layer,
) -> None:
    """The whole range 9..128, one routing per count (the graph sizes in between are
    the ones that production captures)."""
    engine = _Engine(layer)
    for t in ALL_LARGE_TS:
        u, ids, _ = _routing(t, 340 + t)
        ref = _valid(*engine.run(u, ids))
        got = _valid(*_combo(engine, u, ids, LARGE_DEFAULT_FLAGS, True, True))
        assert torch.equal(got[0], ref[0]), t
        assert torch.equal(got[1], ref[1]), t


# ------------------------------------------------------------------------------
# The quantization launch: the bytes of the stock op and of the preparation launch
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("t", [9, 10, 16, 31, 32, 33, 64, 100, MAX_LARGE_T])
@torch.inference_mode()
def test_act_quant_beyond_8_tokens_is_the_stock_op_and_the_prep_launch(
    layer: _Layer, t: int
) -> None:
    engine = _Engine(layer)
    ids = _all_ids(t, 3, layer.e_local)["random"]
    for name, x, gscale in _quant_cases(t):
        gs = layer.gs.clone()
        gs[0] = gscale
        q_ref, sf_ref = _stock_quant(x, gscale)
        for pdl in (False, True):
            data = torch.full((t, H // 2), 0xFF, dtype=torch.uint8, device=DEVICE)
            sf = torch.full((t, H // 16), 0xFF, dtype=torch.uint8, device=DEVICE)
            torch.ops._muse_C.nvfp4_decode_act_quant(x, gs, data, sf, pdl)
            assert torch.equal(data, q_ref), (name, pdl)
            assert torch.equal(sf, sf_ref), (name, pdl)
        # What the preparation launch of the large-batch op writes into its scratch.
        engine.run(x, ids, large=True, gs=gs)
        assert torch.equal(engine.xq_data[:t], data), name
        assert torch.equal(engine.xq_sf[:t], sf), name


@torch.inference_mode()
def test_act_quant_serves_every_token_count_up_to_128(layer: _Layer) -> None:
    for t in range(1, MAX_LARGE_T + 1):
        x = _routing(t, 360 + t)[0]
        gscale = float(layer.gs[0])
        q_ref, sf_ref = _stock_quant(x, gscale)
        data = torch.zeros(t, H // 2, dtype=torch.uint8, device=DEVICE)
        sf = torch.zeros(t, H // 16, dtype=torch.uint8, device=DEVICE)
        torch.ops._muse_C.nvfp4_decode_act_quant(x, layer.gs, data, sf)
        assert torch.equal(data, q_ref), t
        assert torch.equal(sf, sf_ref), t
    x = _routing(MAX_LARGE_T + 1, 5)[0]
    with pytest.raises(RuntimeError):
        torch.ops._muse_C.nvfp4_decode_act_quant(
            x,
            layer.gs,
            torch.zeros(x.shape[0], H // 2, dtype=torch.uint8, device=DEVICE),
            torch.zeros(x.shape[0], H // 16, dtype=torch.uint8, device=DEVICE),
        )
    with pytest.raises(RuntimeError):  # no programmatic launch of the small kernel
        torch.ops._muse_C.nvfp4_decode_act_quant(
            x[:4],
            layer.gs,
            torch.zeros(4, H // 2, dtype=torch.uint8, device=DEVICE),
            torch.zeros(4, H // 16, dtype=torch.uint8, device=DEVICE),
            True,
        )


# ------------------------------------------------------------------------------
# The kernels wait for what they read: slow producers of the ids and the activations
# ------------------------------------------------------------------------------
@pytest.mark.parametrize("t", [9, 33, 64, MAX_LARGE_T])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_plan_and_quantization_wait_for_their_producers_on_two_streams(
    layer: _Layer, t: int, use_graph: bool, capfd
) -> None:
    """The routing stand-in on the aux stream writes the ids late (``slow_copy`` lets
    its dependents launch at once) and the plan launch is its programmatic dependent;
    the norm stand-in on the main stream writes the activations late and the
    quantization is its dependent; the large-batch kernel is the dependent of the
    quantization on the main stream and joins the aux stream by an event. Stale ids and
    activations stand in the buffers until the producers overwrite them."""
    engine = _Engine(layer)
    u, _, _ = _routing(t, 380 + t)
    ids = _all_ids(t, 381 + t, layer.e_local)["three_local"]
    ref = _valid(*engine.run(u, ids, flags=FLAG_NO_PDL))
    stale_u, _, _ = _routing(t, 382 + t)
    stale_ids = _all_ids(t, 383 + t, layer.e_local)["all_local"]
    u_in, ids_in = stale_u.clone(), stale_ids.clone()
    side = torch.cuda.Stream()
    fork, join = torch.cuda.Event(), torch.cuda.Event()
    gs = layer.gs

    def step():
        fork.record()
        with torch.cuda.stream(side):
            fork.wait()
            slow_copy((ids_in, ids))
            _plan_op(engine, ids_in)
            join.record(side)
        slow_copy((u_in, u))
        torch.ops._muse_C.nvfp4_decode_act_quant(
            u_in, gs, engine.xq_data[:t], engine.xq_sf[:t], True
        )
        join.wait()
        return engine.run(
            u_in,
            ids_in,
            flags=FLAG_EXTERNAL_PLAN | FLAG_EXTERNAL_XQ | LARGE_DEFAULT_FLAGS,
        )

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
            _poison(engine, t)
            graph.replay()
            synchronize_or_exit("a graph replay", capfd)
            got = _valid(rows, perm)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    else:
        for _ in range(4):
            restore()
            _poison(engine, t)
            rows, perm = step()
            synchronize_or_exit("a launch", capfd)
            got = _valid(rows, perm)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    assert engine.counters_are_zero()


# ------------------------------------------------------------------------------
# The engine: the plan and the quantization are taken for the very ids / activations
# ------------------------------------------------------------------------------
def _engine(layer: _Layer) -> MuseDecodeMoE:
    return MuseDecodeMoE(
        layer.w13, layer.sf13, layer.w2, layer.sf2, prm=layer.prm, gs=layer.gs
    )


class _Spy:
    """Records the flags of the large-batch launches of the engine."""

    def __init__(self, monkeypatch):
        self.flags: list[int] = []
        original = torch.ops._muse_C.nvfp4_decode_moe_large

        def spy(*args):
            self.flags.append(args[-1])
            return original(*args)

        monkeypatch.setattr(torch.ops._muse_C, "nvfp4_decode_moe_large", spy)


@pytest.mark.parametrize("t", [9, 24, MAX_LARGE_T])
@torch.inference_mode()
def test_engine_takes_a_prepared_plan_once(layer: _Layer, t: int, monkeypatch) -> None:
    engine = _engine(layer)
    spy = _Spy(monkeypatch)
    u, ids, _ = _routing(t, 400 + t)
    ref = _valid(*engine.run(u, ids, layer.expert_map))
    engine.build_plan(ids, layer.expert_map)
    got = _valid(*engine.run(u, ids, layer.expert_map))
    # The plan is for one call: the next one builds its own.
    again = _valid(*engine.run(u, ids, layer.expert_map))
    assert [bool(f & FLAG_EXTERNAL_PLAN) for f in spy.flags] == [False, True, False]
    for x in (got, again):
        assert torch.equal(x[0], ref[0]) and torch.equal(x[1], ref[1])


@torch.inference_mode()
def test_engine_ignores_a_plan_built_for_other_ids(layer: _Layer, monkeypatch) -> None:
    engine = _engine(layer)
    spy = _Spy(monkeypatch)
    t = 20
    u, ids, _ = _routing(t, 420)
    other = _routing(t, 421)[1]
    ref = _valid(*engine.run(u, ids, layer.expert_map))
    engine.build_plan(other, layer.expert_map)
    got = _valid(*engine.run(u, ids, layer.expert_map))  # ids of another buffer
    assert not spy.flags[-1] & FLAG_EXTERNAL_PLAN
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    # Another map, or none, is another plan as well.
    engine.build_plan(ids, layer.expert_map)
    engine.run(u, ids, None)
    assert not spy.flags[-1] & FLAG_EXTERNAL_PLAN
    engine.build_plan(ids, layer.expert_map)
    engine.discard_plan()
    engine.run(u, ids, layer.expert_map)
    assert not spy.flags[-1] & FLAG_EXTERNAL_PLAN


@torch.inference_mode()
def test_engine_plan_serves_the_large_batch_kernel_only(layer: _Layer) -> None:
    engine = _engine(layer)
    for t in (1, 4, MUSE_SMALL_MAX_TOKENS):
        ids = _routing(t, 430)[1]
        with pytest.raises(RuntimeError):
            engine.build_plan(ids, layer.expert_map)
    ids = _routing(MAX_LARGE_T + 1, 431)[1]
    with pytest.raises(RuntimeError):
        engine.build_plan(ids, layer.expert_map)
    # A small batch run does not consume nor choke on a stale plan of a large one.
    engine.build_plan(_routing(12, 432)[1], layer.expert_map)
    u, ids, _ = _routing(4, 433)
    engine.run(u, ids, layer.expert_map)
    assert engine._plan_key is None


@pytest.mark.parametrize("t", [9, 33, MAX_LARGE_T])
@torch.inference_mode()
def test_engine_takes_activations_quantized_ahead(
    layer: _Layer, t: int, monkeypatch
) -> None:
    engine = _engine(layer)
    spy = _Spy(monkeypatch)
    u, ids, _ = _routing(t, 440 + t)
    ref = _valid(*engine.run(u, ids, layer.expert_map))
    engine.quantize_input(u)
    engine.build_plan(ids, layer.expert_map)
    got = _valid(*engine.run(u, ids, layer.expert_map))
    assert spy.flags[-1] & FLAG_EXTERNAL_XQ and spy.flags[-1] & FLAG_EXTERNAL_PLAN
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    # Other activations than the ones quantized: the kernel quantizes its own.
    v = _routing(t, 441 + t)[0]
    ref_v = _valid(*engine.run(v, ids, layer.expert_map))
    engine.quantize_input(u)
    got_v = _valid(*engine.run(v, ids, layer.expert_map))
    assert not spy.flags[-1] & FLAG_EXTERNAL_XQ
    assert torch.equal(got_v[0], ref_v[0]) and torch.equal(got_v[1], ref_v[1])
    engine.quantize_input(u)
    engine.discard_input()
    engine.run(u, ids, layer.expert_map)
    assert not spy.flags[-1] & FLAG_EXTERNAL_XQ


@torch.inference_mode()
def test_engine_input_scratch_is_the_large_batch_scratch(layer: _Layer) -> None:
    engine = _engine(layer)
    data, sf = engine.input_scratch(40)
    assert data.shape == (40, H // 2) and sf.shape == (40, H // 16)
    assert data.dtype == sf.dtype == torch.uint8
    assert engine.input_scale.shape == (2,)
    for t in (1, MUSE_SMALL_MAX_TOKENS, MAX_LARGE_T + 1):
        with pytest.raises(RuntimeError):
            engine.input_scratch(t)


@pytest.mark.parametrize("pdl", [True, False], ids=["pdl", "no_pdl"])
@pytest.mark.parametrize("t", [9, 16, 33, 64, MAX_LARGE_T])
@torch.inference_mode()
def test_engine_aux_stream_plan_in_a_graph_changes_nothing(
    layer: _Layer, t: int, pdl: bool, monkeypatch
) -> None:
    """The plan queued on a side stream behind a routing stand-in, joined by an event,
    as the model captures it: the replays give the rows of the plain launch."""
    monkeypatch.setattr(muse_decode_moe, "muse_pdl_enabled", lambda: pdl)
    engine = _engine(layer)
    u = _routing(t, 450 + t)[0]
    ids = _all_ids(t, 451 + t, layer.e_local)["three_local"]
    ref = _valid(*engine.run(u, ids, layer.expert_map))
    side = torch.cuda.Stream()
    fork, join = torch.cuda.Event(), torch.cuda.Event()
    routed = torch.empty_like(ids)

    def body():
        fork.record()
        with torch.cuda.stream(side):
            fork.wait()
            routed.copy_(ids)  # the routing stand-in
            engine.build_plan(routed, layer.expert_map)
            join.record(side)
        join.wait()
        return engine.run(u, routed, layer.expert_map)

    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        body()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rows, perm = body()
    for _ in range(4):
        engine._scratch.plan.fill_(0x5A5A5A5A)
        rows.zero_()
        perm.fill_(-2)
        graph.replay()
        torch.accelerator.synchronize()
        got = _valid(rows, perm)
        assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


# ------------------------------------------------------------------------------
# The model and the runner: where the plan is built, and what drops it
# ------------------------------------------------------------------------------
@pytest.fixture
def capturing(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)


def _moe(aux=True, engine=True, quantizes=True):
    moe = object.__new__(model_ops.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    moe.experts = SimpleNamespace(has_muse_decode_engine=engine)
    moe._routes_on_aux_stream = lambda n: aux
    # The quantization is made in front of the kernel (see _quantizes_before_experts).
    moe._quantizes_in_norm = lambda n: quantizes
    moe.shards_pooler = lambda n: True
    moe.fuses_boundary_collectives = False
    return moe


@pytest.mark.parametrize("t", [1, 8, 9, 33, 64, 65, MAX_LARGE_T, MAX_LARGE_T + 1])
def test_the_plan_is_built_for_the_large_batch_sizes_up_to_the_limit(
    capturing, monkeypatch, t
):
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN", "1")
    monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN_MAX_TOKENS", raising=False)
    assert _moe()._builds_large_plan(t) is (8 < t <= 64)  # the default limit
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN_MAX_TOKENS", "128")
    assert _moe()._builds_large_plan(t) is (8 < t <= MAX_LARGE_T)
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN_MAX_TOKENS", "16")
    assert _moe()._builds_large_plan(t) is (8 < t <= 16)


def test_the_plan_needs_the_flag_the_capture_the_aux_stream_and_the_engine(
    capturing, monkeypatch
):
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN", "1")
    assert _moe()._builds_large_plan(64)
    assert not _moe(aux=False)._builds_large_plan(64)
    assert not _moe(engine=False)._builds_large_plan(64)
    assert not _moe(quantizes=False)._builds_large_plan(64)  # the plan alone is a loss
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN", "0")
    assert not _moe()._builds_large_plan(64)
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN", "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not _moe()._builds_large_plan(64)


def _dispatch(monkeypatch, plain=0, quant=0):
    """The communicator's dispatch: the plain gather + norm kind up to ``plain`` rows,
    the quantizing one up to ``quant`` rows (host plugin: 16 and 0 by default)."""
    rows = {
        fused_boundary.KIND_ALL_GATHER_RMSNORM: plain,
        fused_boundary.KIND_ALL_GATHER_RMSNORM_NVFP4: quant,
    }
    monkeypatch.setattr(
        model_ops.fused_boundary,
        "supports_fused_boundary",
        lambda kind, n: 1 <= n <= rows.get(kind, 0),
    )


@pytest.mark.parametrize("t", [9, 16, 17, 24, 25, 32, 64, MAX_LARGE_T])
def test_the_quantization_is_ahead_of_the_kernel_unless_the_gather_hides_the_norm(
    capturing, monkeypatch, t
):
    moe = _moe()
    moe.fuses_boundary_collectives = True
    # Host plugin defaults: the plain gather + norm up to 16 rows, no quantizing kind.
    # The norm is hidden in the gather there, nothing quantizes ahead of the kernel.
    _dispatch(monkeypatch, plain=16)
    assert moe._quantizes_before_experts(t) is (t > 16)
    # MUSE_TOPO_BND_RMS_MAX_ROWS=24 alone: the same up to 24 rows.
    _dispatch(monkeypatch, plain=24)
    assert moe._quantizes_before_experts(t) is (t > 24)
    # MUSE_TOPO_BND_QUANT=1 as well: the gather's epilogue quantizes up to 24 rows, the
    # norm + quantization launch behind it above.
    _dispatch(monkeypatch, plain=24, quant=24)
    assert moe._quantizes_before_experts(t)
    # No communicator kinds at all (NCCL, or a model without the plugin).
    _dispatch(monkeypatch)
    assert moe._quantizes_before_experts(t)
    moe.fuses_boundary_collectives = False
    _dispatch(monkeypatch, plain=64)
    assert moe._quantizes_before_experts(t)
    moe.fuses_boundary_collectives = True
    moe.shards_pooler = lambda n: False  # an unsharded pooler: no gather to fuse
    assert moe._quantizes_before_experts(t)
    moe.shards_pooler = lambda n: True
    assert not _moe(quantizes=False)._quantizes_before_experts(t)


@pytest.mark.parametrize("t", [9, 16, 24, 64])
def test_the_aux_plan_follows_the_quantization_ahead_of_the_kernel(
    capturing, monkeypatch, t
):
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN", "1")
    monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN_MAX_TOKENS", raising=False)
    moe = _moe()
    moe.fuses_boundary_collectives = True
    _dispatch(monkeypatch, plain=16)
    assert moe._builds_large_plan(t) is (t > 16)  # c16 stays as it was
    _dispatch(monkeypatch, plain=24, quant=24)
    assert moe._builds_large_plan(t)


def test_the_flag_is_on_by_default(monkeypatch):
    monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN", raising=False)
    assert envs.VLLM_MUSE_SPARK_SM120_AUX_PLAN is True
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_AUX_PLAN", "0")
    assert envs.VLLM_MUSE_SPARK_SM120_AUX_PLAN is False


class _FakeRunner:
    """Stands in for the MoE runner of ``_route_topk_and_pool``: records the stream of
    its calls."""

    def __init__(self, log):
        self.log = log
        self.has_muse_decode_engine = True

    def select_experts(self, hidden, logits):
        self.log.append(("select", torch.cuda.current_stream()))
        w = torch.ones(hidden.shape[0], TOP_K, device=DEVICE)
        ids = torch.zeros(hidden.shape[0], TOP_K, dtype=torch.int32, device=DEVICE)
        return w, ids

    def build_decode_plan(self, ids):
        self.log.append(("plan", torch.cuda.current_stream(), ids))
        return True


def _routing_moe(log, builds):
    moe = object.__new__(model_ops.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    moe.experts = _FakeRunner(log)
    moe.use_mega_moe = False
    moe.pooler = object()
    moe.expert_prefetch = None
    moe._router_aux_stream = torch.cuda.Stream()
    moe._router_events = (torch.cuda.Event(), torch.cuda.Event())
    moe._router_logits = lambda x: x
    moe._pool_and_norm = lambda x: x
    moe._builds_large_plan = lambda n: builds
    moe.prefetches_experts = lambda n: False
    return moe


@pytest.mark.parametrize("builds", [True, False])
def test_the_plan_is_built_on_the_aux_stream_right_behind_the_top_k(builds):
    log = []
    moe = _routing_moe(log, builds)
    t = 32
    x = torch.randn(t, 8, device=DEVICE)
    pooled, w, ids = moe._route_topk_and_pool(x, x)
    torch.accelerator.synchronize()
    aux = moe._router_aux_stream
    assert log[0][:2] == ("select", aux)
    if builds:
        assert log[1][:2] == ("plan", aux)
        assert log[1][2] is ids  # the very ids the experts are given
        assert len(log) == 2
    else:
        assert len(log) == 1


class _RoutedRunner(model_ops.MuseMoERunner):
    """A runner whose ``forward_routed`` reaches its (failing) modular experts."""

    supports_routed_forward = True
    moe_config = SimpleNamespace(hidden_dim=H)
    _quant_method = SimpleNamespace(skip_forward_padding=False)

    def _sequence_parallel_context(self):
        return contextlib.nullcontext()


def test_forward_routed_drops_a_plan_the_experts_did_not_take():
    log = []
    experts = object.__new__(MuseSparkSm120NvFp4Experts)
    experts.use_muse_decode = True
    experts.muse = SimpleNamespace(
        discard_plan=lambda: log.append("discard_plan"),
        discard_input=lambda: log.append("discard_input"),
    )
    quant_method = object.__new__(MuseSparkModelOptNvFp4FusedMoE)
    quant_method.use_sm120_native = True
    quant_method.moe_kernel = SimpleNamespace(fused_experts=experts)

    def boom(**kwargs):
        raise ValueError("boom")

    runner = object.__new__(_RoutedRunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace(
        quant_method=quant_method,
        _ensure_moe_quant_config_init=lambda: None,
        forward_modular=boom,
    )
    x = torch.zeros(4, H, dtype=torch.bfloat16, device=DEVICE)
    with pytest.raises(ValueError, match="boom"):
        runner.forward_routed(x, x, x)
    assert log == ["discard_plan", "discard_input"]


def test_the_runner_builds_the_plan_through_the_native_experts():
    calls = []
    experts = object.__new__(MuseSparkSm120NvFp4Experts)
    experts.prepare_routed_plan = lambda ids, emap: calls.append((ids, emap)) or True
    quant_method = object.__new__(MuseSparkModelOptNvFp4FusedMoE)
    quant_method.use_sm120_native = True
    quant_method.moe_kernel = SimpleNamespace(fused_experts=experts)
    runner = object.__new__(model_ops.MuseMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace(
        quant_method=quant_method, expert_map="the map"
    )
    ids = torch.zeros(12, TOP_K, dtype=torch.int32, device=DEVICE)
    assert runner.build_decode_plan(ids)
    assert calls == [(ids, "the map")]
    quant_method.use_sm120_native = False
    assert not runner.build_decode_plan(ids)
    assert len(calls) == 1


@pytest.mark.parametrize("t", [4, 8, 9, 64, MAX_LARGE_T, MAX_LARGE_T + 1])
@pytest.mark.parametrize("capturing_now", [True, False])
def test_the_experts_queue_a_plan_in_a_capture_for_the_large_batch_sizes_only(
    monkeypatch, t, capturing_now
):
    queued = []
    experts = object.__new__(MuseSparkSm120NvFp4Experts)
    experts.use_muse_decode = True
    experts.muse = SimpleNamespace(
        max_tokens=MAX_LARGE_T,
        build_plan=lambda ids, emap: queued.append((ids, emap)),
    )
    monkeypatch.setattr(sm120_nvfp4_moe, "_capturing", lambda: capturing_now)
    ids = torch.zeros(t, TOP_K, dtype=torch.int32, device=DEVICE)
    got = experts.prepare_routed_plan(ids, "map")
    want = capturing_now and 8 < t <= MAX_LARGE_T
    assert got is want
    assert bool(queued) is want
    experts.use_muse_decode = False
    assert not experts.prepare_routed_plan(ids, "map")


# ------------------------------------------------------------------------------
# The pre-expert RMSNorm and the quantization of its output in one launch
# ------------------------------------------------------------------------------
def _norm_inputs(kind: str, t: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(900 + seed)

    def r(scale=1.0):
        x = torch.randn(t, H, device="cuda", generator=g) * scale
        return x.to(torch.bfloat16)

    if kind == "normal":
        return r(3.0)
    if kind == "heavy":  # outlier channels like the pooled latent of a real model
        x = r(1.0)
        idx = torch.randint(0, H, (t, 8), device="cuda", generator=g)
        outliers = torch.randn(t, 8, device="cuda", generator=g) * 300
        x.scatter_(1, idx, outliers.to(torch.bfloat16))
        return x
    if kind == "tiny":
        return r(1e-18)
    if kind == "zero":
        return torch.zeros(t, H, dtype=torch.bfloat16, device="cuda")
    if kind == "big":
        return r(3e4)
    if kind == "ties":  # equal magnitudes: ties of the e2m1 grid after the norm
        sign = torch.where(torch.arange(H, device="cuda") % 2 == 0, 1.0, -1.0)
        return (sign * 0.75).expand(t, H).to(torch.bfloat16).contiguous()
    raise ValueError(kind)


NORM_KINDS = ("normal", "heavy", "tiny", "zero", "big", "ties")
NORM_CONFIGS = ((1e-6, 1.0), (1e-8, 0.0), (1e-5, 0.5), (1e-6, -0.25))

# The CUDA norm (``_muse_C.rmsnorm_nvfp4_quant``) and the Triton reference
# (``fused_rmsnorm_fwd``) were bit-identical up to Triton 3.7. Triton 3.8 (torch 2.14)
# changed the bits of the Triton kernel's output: a rounding tie of the final bf16 cast
# now resolves the other way in a few elements (152 of ~203M, 7.5e-7, over the 768
# configs of the first test below), every one by exactly 1 bf16 ulp; both kernels are
# equally accurate against a float64 reference. The comparisons of the CUDA norm with the
# Triton reference therefore check the bf16 bit patterns exactly: every difference is at
# most 1 ulp, and the number of differing elements stays under 1e-5 of the elements
# (13x the measured rate; a wrong rounding mode or a changed summation order differs in
# orders of magnitude more elements, and the max-1-ulp check catches any error beyond
# the last bit). Everything downstream of the norm stays bitwise.
NORM_MAX_ULP = 1
NORM_MAX_DIFF_FRACTION = 1e-5


def _bf16_ordinal(x: torch.Tensor) -> torch.Tensor:
    """Monotonic integer image of the bf16 bit patterns: adjacent bf16 values (also
    across zero, where -0.0 and +0.0 coincide) are 1 apart."""
    bits = x.view(torch.int16).to(torch.int32)
    return torch.where(bits < 0, -32768 - bits, bits)


def _norm_ulp_mismatches(
    out: torch.Tensor,
    ref: torch.Tensor,
    what,
    keep: torch.Tensor | None = None,
) -> int:
    """Number of elements of ``out`` (restricted to ``keep``) whose bf16 bits differ from
    ``ref``; asserts that none is more than ``NORM_MAX_ULP`` ulps away."""
    dist = (_bf16_ordinal(out) - _bf16_ordinal(ref)).abs()
    if keep is not None:
        dist = dist[keep]
    worst = int(dist.max()) if dist.numel() else 0
    assert worst <= NORM_MAX_ULP, (what, "ulp distance", worst)
    return int((dist != 0).sum())


def _norm_diff_budget(numel: int) -> int:
    """Differing elements allowed among ``numel`` (at least one for a small call)."""
    return max(1, math.ceil(NORM_MAX_DIFF_FRACTION * numel))


@torch.inference_mode()
def test_rmsnorm_quant_is_the_triton_norm_and_the_quantization_of_its_output(
    layer: _Layer,
) -> None:
    g = torch.Generator(device="cuda").manual_seed(77)
    w = (torch.randn(H, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
    gscale = float(layer.gs[0])
    differing = total = 0
    for t in range(1, MAX_LARGE_T + 1):
        for kind in NORM_KINDS:
            x = _norm_inputs(kind, t, t)
            eps, gain = NORM_CONFIGS[(t + len(kind)) % len(NORM_CONFIGS)]
            ref = fused_rmsnorm_fwd(
                x, w, eps, gain_center=gain, input_scale=False, use_kahan_summation=True
            )
            out = torch.full_like(x, float("nan"))
            data = torch.full((t, H // 2), 0xFF, dtype=torch.uint8, device="cuda")
            sf = torch.full((t, H // 16), 0xFF, dtype=torch.uint8, device="cuda")
            for pdl in (False, True):
                torch.ops._muse_C.rmsnorm_nvfp4_quant(
                    x, w, eps, gain, layer.gs, out, data, sf, pdl
                )
                differing += _norm_ulp_mismatches(out, ref, (t, kind, pdl))
                total += out.numel()
                # The quantization is that of the CUDA norm's own output, bit for bit.
                q_ref, sf_ref = _stock_quant(out, gscale)
                assert torch.equal(data, q_ref), (t, kind, pdl)
                assert torch.equal(sf, sf_ref), (t, kind, pdl)
    assert differing <= _norm_diff_budget(total), (differing, total)


@pytest.mark.parametrize("t", [9, 33, MAX_LARGE_T])
@torch.inference_mode()
def test_rmsnorm_quant_on_non_finite_rows_is_the_triton_norm_and_act_quant(
    layer: _Layer, t: int
) -> None:
    """Rows with an Inf or a NaN (a broken step, never a served one) take the same
    arithmetic as in the Triton norm (same NaN positions, the same bits elsewhere up to
    the 1-ulp rounding of ``NORM_MAX_ULP``) and the same quantization as the stock
    activation quantization of the output."""
    g = torch.Generator(device="cuda").manual_seed(31)
    w = (torch.randn(H, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
    for kind in ("inf", "nan", "mixed"):
        x = _norm_inputs("normal", t, t + 5).clone()
        if kind in ("inf", "mixed"):
            x[1, 7] = float("inf")
            x[t // 2, 100] = float("-inf")
        if kind in ("nan", "mixed"):
            x[2, 0] = float("nan")
            x[t - 1, H - 1] = float("nan")
        ref = fused_rmsnorm_fwd(
            x, w, 1e-6, gain_center=1.0, input_scale=False, use_kahan_summation=True
        )
        out = torch.zeros_like(x)
        data = torch.full((t, H // 2), 0xFF, dtype=torch.uint8, device="cuda")
        sf = torch.full((t, H // 16), 0xFF, dtype=torch.uint8, device="cuda")
        torch.ops._muse_C.rmsnorm_nvfp4_quant(
            x, w, 1e-6, 1.0, layer.gs, out, data, sf, True
        )
        nan_ref, nan_out = torch.isnan(ref), torch.isnan(out)
        assert torch.equal(nan_ref, nan_out), kind
        differing = _norm_ulp_mismatches(out, ref, kind, keep=~nan_ref)
        assert differing <= _norm_diff_budget(int((~nan_ref).sum())), (kind, differing)
        q_data = torch.empty_like(data)
        q_sf = torch.empty_like(sf)
        torch.ops._muse_C.nvfp4_decode_act_quant(out, layer.gs, q_data, q_sf)
        assert torch.equal(data, q_data), kind
        assert torch.equal(sf, q_sf), kind


@torch.inference_mode()
def test_rmsnorm_quant_rejects_bad_arguments(layer: _Layer) -> None:
    t = 12
    x = _norm_inputs("normal", t, 1)
    w = torch.ones(H, dtype=torch.bfloat16, device="cuda")
    out = torch.empty_like(x)
    data = torch.empty(t, H // 2, dtype=torch.uint8, device="cuda")
    sf = torch.empty(t, H // 16, dtype=torch.uint8, device="cuda")
    op = torch.ops._muse_C.rmsnorm_nvfp4_quant
    op(x, w, 1e-6, 1.0, layer.gs, out, data, sf, False)
    with pytest.raises(RuntimeError):  # weight of another width
        op(x, w[:H // 2], 1e-6, 1.0, layer.gs, out, data, sf, False)
    with pytest.raises(RuntimeError):  # fp32 input
        op(x.float(), w, 1e-6, 1.0, layer.gs, out, data, sf, False)
    with pytest.raises(RuntimeError):  # scratch of another size
        op(x, w, 1e-6, 1.0, layer.gs, out, data[:5], sf, False)
    with pytest.raises(RuntimeError):  # misaligned weight
        op(x, torch.ones(H + 8, dtype=torch.bfloat16, device="cuda")[1:H + 1], 1e-6,
           1.0, layer.gs, out, data, sf, False)  # fmt: skip
    big = _norm_inputs("normal", MAX_LARGE_T + 1, 2)
    big_data = torch.empty(big.shape[0], H // 2, dtype=torch.uint8, device="cuda")
    big_sf = torch.empty(big.shape[0], H // 16, dtype=torch.uint8, device="cuda")
    with pytest.raises(RuntimeError):
        op(big, w, 1e-6, 1.0, layer.gs, torch.empty_like(big), big_data, big_sf, False)


@pytest.mark.parametrize("t", [9, 33, 64, MAX_LARGE_T])
@torch.inference_mode()
def test_engine_runs_on_the_output_of_the_norm_and_quantization_launch(
    layer: _Layer, t: int, monkeypatch
) -> None:
    engine = _engine(layer)
    spy = _Spy(monkeypatch)
    g = torch.Generator(device="cuda").manual_seed(78)
    w = (torch.randn(H, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
    x = _norm_inputs("normal", t, 5)
    ids = _all_ids(t, 460 + t, layer.e_local)["three_local"]
    normed = fused_rmsnorm_fwd(x, w, 1e-6, gain_center=1.0, use_kahan_summation=True)
    first = engine.rmsnorm_quantize(x, w, 1e-6, 1.0)
    differing = _norm_ulp_mismatches(first, normed, t)
    assert differing <= _norm_diff_budget(normed.numel()), (t, differing)
    # The reference run takes the rows of the CUDA norm through a copy (another data
    # pointer: the engine quantizes them itself instead of taking the scratch); the norm
    # launch is then repeated, so the scratch the engine reads below was written by it.
    ref = _valid(*engine.run(first.clone(), ids, layer.expert_map))
    got_normed = engine.rmsnorm_quantize(x, w, 1e-6, 1.0)
    assert torch.equal(got_normed.view(torch.int16), first.view(torch.int16))
    engine.build_plan(ids, layer.expert_map)
    got = _valid(*engine.run(got_normed, ids, layer.expert_map))
    assert spy.flags[-1] & FLAG_EXTERNAL_XQ and spy.flags[-1] & FLAG_EXTERNAL_PLAN
    assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    # Quantized rows of another tensor are not taken for these.
    engine.rmsnorm_quantize(x, w, 1e-6, 1.0)
    engine.run(normed, ids, layer.expert_map)
    assert not spy.flags[-1] & FLAG_EXTERNAL_XQ


@pytest.mark.parametrize("t", [9, 64, MAX_LARGE_T])
@pytest.mark.parametrize("use_graph", [False, True])
@torch.inference_mode()
def test_norm_quant_launch_waits_for_the_producer_of_its_input(
    layer: _Layer, t: int, use_graph: bool, capfd
) -> None:
    """The producer of the latent (the all-gather stand-in) lets its dependents launch at
    once and writes late; the norm + quantization launch is its programmatic dependent
    and the large-batch kernel the dependent of that."""
    engine = _engine(layer)
    g = torch.Generator(device="cuda").manual_seed(79)
    w = (torch.randn(H, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
    x = _norm_inputs("heavy", t, 6)
    stale = _norm_inputs("normal", t, 7)
    ids = _all_ids(t, 470 + t, layer.e_local)["random"]
    # The reference input is the CUDA norm's own output (through a copy, so the engine
    # quantizes it itself): the launches below are compared with it bit for bit, which
    # the Triton norm's 1-ulp rounding differences (NORM_MAX_ULP) would not allow.
    normed = engine.rmsnorm_quantize(x, w, 1e-6, 1.0).clone()
    ref = _valid(*engine.run(normed, ids, layer.expert_map))
    x_in = stale.clone()

    def step():
        slow_copy((x_in, x))
        out = engine.rmsnorm_quantize(x_in, w, 1e-6, 1.0)
        return engine.run(out, ids, layer.expert_map)

    x_in.copy_(stale)
    step()
    synchronize_or_exit("the first launch", capfd)
    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            rows, perm = step()
        for _ in range(4):
            x_in.copy_(stale)
            graph.replay()
            synchronize_or_exit("a graph replay", capfd)
            got = _valid(rows, perm)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])
    else:
        for _ in range(4):
            x_in.copy_(stale)
            rows, perm = step()
            synchronize_or_exit("a launch", capfd)
            got = _valid(rows, perm)
            assert torch.equal(got[0], ref[0]) and torch.equal(got[1], ref[1])


def test_the_norm_quantizes_for_the_large_batch_sizes_in_a_capture_only(
    capturing, monkeypatch
):
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT", "1")

    def moe(**kw):
        m = object.__new__(model_ops.MuseSpark1xMoE)
        torch.nn.Module.__init__(m)
        norm = SimpleNamespace(
            has_weight=kw.get("has_weight", True),
            applies_weight_before_norm=kw.get("weight_first", False),
            variance_size_override=kw.get("var_override", None),
        )
        m.pre_expert_norm = None if kw.get("no_norm") else norm
        runner = object.__new__(model_ops.MuseMoERunner)
        torch.nn.Module.__init__(runner)
        runner.__class__ = type(
            "R", (model_ops.MuseMoERunner,),
            {"has_muse_decode_engine": property(lambda self: kw.get("engine", True))},
        )  # fmt: skip
        m.experts = runner
        return m

    for t in (1, 8, 129):
        assert not moe()._quantizes_in_norm(t)
    for t in (9, 33, 64, MAX_LARGE_T):
        assert moe()._quantizes_in_norm(t)
    for kw in (
        {"has_weight": False}, {"weight_first": True}, {"var_override": 1024},
        {"no_norm": True}, {"engine": False},
    ):  # fmt: skip
        assert not moe(**kw)._quantizes_in_norm(64), kw
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT", "0")
    assert not moe()._quantizes_in_norm(64)
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT", "1")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not moe()._quantizes_in_norm(64)
    monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT")
    assert envs.VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT is True  # on by default
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT", "0")
    assert envs.VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT is False


def test_pool_and_norm_keeps_the_fused_gather_and_takes_the_norm_quant_launch_after_it(
    capturing, monkeypatch
):
    calls = []
    t = 32
    moe = object.__new__(model_ops.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    moe.pooler = object()
    moe.shards_pooler = lambda n: True
    moe._fused_gemv_all_gather_rmsnorm = lambda x: None
    moe._project_local_columns = lambda layer, x: x[:, :512].contiguous()
    moe._fused_all_gather_rmsnorm_nvfp4 = lambda shard: None  # (no quantizing gather)
    fuses = [True]
    moe._fused_all_gather_rmsnorm = lambda shard: (
        calls.append("fused_gather") or shard if fuses[0] else None
    )
    moe.pre_expert_norm = lambda x: calls.append("plain_norm") or x
    quantizes = [True]
    moe._quantizes_in_norm = lambda n: quantizes[0]
    moe._norm_quant_decode_input = lambda x: (
        calls.append("norm_quant") or x if quantizes[0] else None
    )
    monkeypatch.setattr(model_ops, "gather_columns", lambda shard: shard.repeat(1, 8))
    x = torch.zeros(t, 8192, dtype=torch.bfloat16, device=DEVICE)
    # The communicator fuses the gather with the norm: as before, nothing quantizes.
    moe._pool_and_norm(x)
    assert calls == ["fused_gather"]
    # No fused gather for this size: a plain gather, then the norm + quantization.
    calls.clear()
    fuses[0] = False
    moe._pool_and_norm(x)
    assert calls == ["norm_quant"]
    # Not quantizing: the model's own norm.
    calls.clear()
    quantizes[0] = False
    moe._pool_and_norm(x)
    assert calls == ["plain_norm"]


def test_the_experts_quantize_in_the_norm_in_a_capture_for_the_large_batch_only(
    monkeypatch,
):
    seen = []
    experts = object.__new__(MuseSparkSm120NvFp4Experts)
    experts.use_muse_decode = True
    experts.muse = SimpleNamespace(
        max_tokens=MAX_LARGE_T,
        rmsnorm_quantize=lambda x, w, eps, gain: seen.append((eps, gain)) or x,
    )
    w = torch.ones(H, dtype=torch.bfloat16, device=DEVICE)

    def call(t, capturing_now=True, **over):
        monkeypatch.setattr(sm120_nvfp4_moe, "_capturing", lambda: capturing_now)
        x = over.get("x", torch.zeros(t, H, dtype=torch.bfloat16, device=DEVICE))
        return experts.prepare_normed_input(x, over.get("w", w), 1e-6, 1.0)

    assert call(64) is not None and seen == [(1e-6, 1.0)]
    for t in (4, 8, MAX_LARGE_T + 1):
        assert call(t) is None
    assert call(64, capturing_now=False) is None
    assert call(64, x=torch.zeros(64, H, device=DEVICE)) is None  # fp32
    assert call(64, w=w[: H // 2]) is None
    transposed = torch.zeros(H, 64, dtype=torch.bfloat16, device=DEVICE).t()
    assert call(64, x=transposed) is None
    experts.use_muse_decode = False
    assert call(64) is None


# ------------------------------------------------------------------------------
# The communicator's gather + norm that also quantizes (the host plugin's
# all_gather_rmsnorm_nvfp4): the optional interface, and the model's use of it
# ------------------------------------------------------------------------------
def test_the_nvfp4_gather_norm_is_an_optional_communicator_method(monkeypatch):
    shard = torch.zeros(16, 512, dtype=torch.bfloat16, device=DEVICE)
    w = torch.ones(H, dtype=torch.bfloat16, device=DEVICE)
    gs = torch.ones(2, device=DEVICE)
    codes = torch.empty(16, H // 2, dtype=torch.uint8, device=DEVICE)
    scales = torch.empty(16, H // 16, dtype=torch.uint8, device=DEVICE)
    args = (shard, w, 1e-6, 1.0, gs, codes, scales)
    monkeypatch.setattr(fused_boundary, "_communicator", lambda: None)
    assert fused_boundary.all_gather_rmsnorm_nvfp4(*args) is None
    monkeypatch.setattr(fused_boundary, "_communicator", lambda: SimpleNamespace())
    assert fused_boundary.all_gather_rmsnorm_nvfp4(*args) is None
    seen = []

    def method(*a):
        seen.append(a)
        return "rows"

    communicator = SimpleNamespace(all_gather_rmsnorm_nvfp4=method)
    monkeypatch.setattr(fused_boundary, "_communicator", lambda: communicator)
    assert fused_boundary.all_gather_rmsnorm_nvfp4(*args) == "rows"
    assert seen == [args]
    assert fused_boundary.KIND_ALL_GATHER_RMSNORM_NVFP4 == "all_gather_rmsnorm_nvfp4"


class _QuantRunner:
    def __init__(self, scratch=True):
        self.scratch = scratch
        self.marked = []
        self.codes = torch.empty(16, H // 2, dtype=torch.uint8, device=DEVICE)
        self.sf = torch.empty(16, H // 16, dtype=torch.uint8, device=DEVICE)
        self.gs = torch.ones(2, device=DEVICE)

    def quant_decode_scratch(self, n):
        return (self.codes, self.sf, self.gs) if self.scratch else None

    def mark_quantized_decode_input(self, x):
        self.marked.append(x)


def _gather_moe(runner, quantizes=True):
    moe = object.__new__(model_ops.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    moe.experts = runner
    moe.fuses_boundary_collectives = True
    moe.pre_expert_norm = SimpleNamespace(
        weight=torch.ones(H, dtype=torch.bfloat16, device=DEVICE),
        variance_epsilon=1e-6,
        fused_gain_center=1.0,
    )
    moe._quantizes_in_norm = lambda n: quantizes
    return moe


def test_the_model_asks_the_communicator_for_the_quantizing_gather_norm(monkeypatch):
    shard = torch.zeros(16, 512, dtype=torch.bfloat16, device=DEVICE)
    calls = []
    supported = [True]
    monkeypatch.setattr(
        fused_boundary,
        "supports_fused_boundary",
        lambda kind, n: calls.append(("supports", kind, n)) or supported[0],
    )
    result = [torch.ones(16, H, dtype=torch.bfloat16, device=DEVICE)]

    def gather(*args):
        calls.append(("gather", args[4].shape, args[5].shape, args[6].shape))
        return result[0]

    monkeypatch.setattr(fused_boundary, "all_gather_rmsnorm_nvfp4", gather)
    runner = _QuantRunner()
    moe = _gather_moe(runner)
    out = moe._fused_all_gather_rmsnorm_nvfp4(shard)
    assert out is result[0] and runner.marked == [out]
    assert calls[0] == ("supports", "all_gather_rmsnorm_nvfp4", 16)
    assert calls[1][0] == "gather"
    # Each way to decline leaves nothing marked and launches nothing.
    for name, moe_, kw in (
        ("not quantizing", _gather_moe(_QuantRunner(), quantizes=False), {}),
        ("no scratch", _gather_moe(_QuantRunner(scratch=False)), {}),
    ):
        calls.clear()
        assert moe_._fused_all_gather_rmsnorm_nvfp4(shard) is None, name
        assert not any(c[0] == "gather" for c in calls), name
    supported[0] = False
    calls.clear()
    runner = _QuantRunner()
    assert _gather_moe(runner)._fused_all_gather_rmsnorm_nvfp4(shard) is None
    assert not any(c[0] == "gather" for c in calls)
    supported[0] = True
    result[0] = None  # the communicator declines: nothing to mark
    assert _gather_moe(runner)._fused_all_gather_rmsnorm_nvfp4(shard) is None
    assert runner.marked == []
    moe = _gather_moe(_QuantRunner())
    moe.fuses_boundary_collectives = False
    assert moe._fused_all_gather_rmsnorm_nvfp4(shard) is None
    bad = shard.float()
    assert _gather_moe(_QuantRunner())._fused_all_gather_rmsnorm_nvfp4(bad) is None


def test_pool_and_norm_prefers_the_quantizing_gather_norm(capturing, monkeypatch):
    calls = []
    moe = object.__new__(model_ops.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    moe.pooler = object()
    moe.shards_pooler = lambda n: True
    moe._fused_gemv_all_gather_rmsnorm = lambda x: None
    moe._project_local_columns = lambda layer, x: x[:, :512].contiguous()
    quantizing = [True]
    moe._fused_all_gather_rmsnorm_nvfp4 = lambda shard: (
        calls.append("nvfp4_gather") or shard if quantizing[0] else None
    )
    moe._fused_all_gather_rmsnorm = lambda shard: calls.append("fused_gather") or shard
    x = torch.zeros(16, 8192, dtype=torch.bfloat16, device=DEVICE)
    moe._pool_and_norm(x)
    assert calls == ["nvfp4_gather"]
    calls.clear()
    quantizing[0] = False
    moe._pool_and_norm(x)
    assert calls == ["fused_gather"]


def test_the_experts_hand_out_the_quantization_scratch_in_a_capture_only(monkeypatch):
    marked = []
    experts = object.__new__(MuseSparkSm120NvFp4Experts)
    experts.use_muse_decode = True
    gs = torch.ones(2, device=DEVICE)
    experts.muse = SimpleNamespace(
        max_tokens=MAX_LARGE_T,
        input_scratch=lambda n: (
            torch.empty(n, H // 2, dtype=torch.uint8, device=DEVICE),
            torch.empty(n, H // 16, dtype=torch.uint8, device=DEVICE),
        ),
        input_scale=gs,
        mark_input_quantized=lambda x: marked.append(x),
    )

    def scratch(n, capturing_now=True):
        monkeypatch.setattr(sm120_nvfp4_moe, "_capturing", lambda: capturing_now)
        return experts.quant_scratch(n)

    got = scratch(24)
    assert got is not None and got[0].shape == (24, H // 2) and got[2] is gs
    assert got[1].shape == (24, H // 16)
    for n in (1, 8, MAX_LARGE_T + 1):
        assert scratch(n) is None
    assert scratch(24, capturing_now=False) is None
    x = torch.zeros(24, H, dtype=torch.bfloat16, device=DEVICE)
    experts.mark_quantized_input(x)
    assert marked == [x]
    experts.use_muse_decode = False
    assert scratch(24) is None
