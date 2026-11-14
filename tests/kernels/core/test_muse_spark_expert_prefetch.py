# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 coverage for the routing-aware expert L2 prefetch of the decode layers.

``_muse_C::expert_prefetch`` maps the routes of a decode batch to this rank's local
experts and prefetches their weights and block scales into L2 in the order the muse
decode kernel consumes them. Its requests are checked through the test-only ``dump``
output (the addresses it would prefetch); as a cache hint it must also leave every
result of the decode kernel bit-for-bit as it was, on a side stream in a CUDA graph.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm.envs as envs
import vllm.models.muse_spark.nvidia.model as model_ops
from tests.kernels.moe.test_muse_spark_sm120_decode_moe import (
    E_GLOBAL,
    H,
    TOP_K,
    _Layer,
    _local_ids,
    _routing,
)
from vllm.models.muse_spark.nvidia.muse_moe_runner import (
    MuseSparkModelOptNvFp4FusedMoE,
)
from vllm.models.muse_spark.nvidia.ops import muse_c
from vllm.models.muse_spark.nvidia.ops.expert_prefetch import (
    EXPERT_PREFETCH_BYTES,
    EXPERT_PREFETCH_CTAS,
    ExpertPrefetch,
)
from vllm.models.muse_spark.nvidia.ops.muse_decode_moe import (
    EXPERT_PREFETCH_MAX_TOKENS,
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

DEVICE = torch.device("cuda")
E_LOCAL = 8
ROW_BYTES = H // 2
SF_BYTES = H // 16
ROW_REQ = 64 * 1024
SCALE_REQ = 32 * 1024  # the scales of 128 rows
# Requests of one expert: GEMM1 = 32 groups x (4 up + 4 gate row requests + 2 scale
# requests), GEMM2 = 32 groups x (4 + 1).
REQS_1 = 32 * 10
REQS_2 = 32 * 5
REQS_PER_EXPERT = REQS_1 + REQS_2
BYTES_1 = 32 * (8 * ROW_REQ + 2 * SCALE_REQ)
BYTES_2 = 32 * (4 * ROW_REQ + SCALE_REQ)
EXPERT_BYTES = BYTES_1 + BYTES_2


@pytest.fixture(scope="module", autouse=True)
def _extension() -> None:
    muse_c.load()


@pytest.fixture(scope="module")
def layer() -> _Layer:
    with torch.inference_mode():
        yield _Layer(diff_s=True, e_local=E_LOCAL)


def _dump(
    layer: _Layer, ids, budget=1 << 40, num_ctas=16, expert_map="layer", pdl=False
):
    """The requests the op reports for ``ids``, as (address, size) in issue order;
    unused slots are (-1, -1)."""
    if isinstance(expert_map, str):
        expert_map = layer.expert_map
    out = torch.full(
        (2 * (E_LOCAL * REQS_PER_EXPERT + 8),), -1, dtype=torch.int64, device=DEVICE
    )
    torch.ops._muse_C.expert_prefetch(
        ids,
        expert_map,
        layer.w13,
        layer.sf13,
        layer.w2,
        layer.sf2,
        False,
        num_ctas,
        budget,
        pdl,
        out,
    )
    torch.accelerator.synchronize()
    return out.view(-1, 2).cpu().numpy()


def _entries(layer: _Layer, ids: torch.Tensor) -> list[int]:
    """The distinct local experts of ``ids`` in order of first appearance."""
    seen: list[int] = []
    for g in ids.flatten().tolist():
        local = int(layer.expert_map[g])
        if local >= 0 and local not in seen:
            seen.append(local)
    return seen


def _expected(layer: _Layer, experts: list[int], up_first: bool = False):
    """The consumption order of the persistent kernel, derived from its units: the
    GEMM1 units of 16 rows come in groups of 8 that cover 128 consecutive rows of the
    up half and of the gate half and one 32 KiB block of scales of each (up tile
    first), GEMM2 units likewise 128 rows of w2 and one block of scales; all GEMM1
    units of every entry precede the GEMM2 units. A request is at most 64 KiB."""
    w13, sf13 = layer.w13.data_ptr(), layer.sf13.data_ptr()
    w2, sf2 = layer.w2.data_ptr(), layer.sf2.data_ptr()
    w13_e, sf13_e = 2 * H * ROW_BYTES, 2 * H * SF_BYTES
    w2_e, sf2_e = H * ROW_BYTES, H * SF_BYTES
    up_base, gate_base = (0, H) if up_first else (H, 0)
    out: list[tuple[int, int]] = []
    for e in experts:
        for a in range(32):
            for base in (up_base, gate_base):
                row0 = base + 128 * a
                out += [
                    (w13 + e * w13_e + row0 * ROW_BYTES + j * ROW_REQ, ROW_REQ)
                    for j in range(4)
                ]
            for base in (up_base, gate_base):
                # 128 rows x 256 scale bytes = one 32 KiB block of the swizzled scales
                block = (base + 128 * a) // 128
                out.append((sf13 + e * sf13_e + block * SCALE_REQ, SCALE_REQ))
    for e in experts:
        for q in range(32):
            out += [
                (w2 + e * w2_e + 128 * q * ROW_BYTES + j * ROW_REQ, ROW_REQ)
                for j in range(4)
            ]
            out.append((sf2 + e * sf2_e + q * SCALE_REQ, SCALE_REQ))
    return out


def _check_order(layer, got, experts):
    expected = _expected(layer, experts)
    n = len(expected)
    # The requests are stored at their position in the order, whatever CTA issued them.
    assert [tuple(x) for x in got[:n].tolist()] == expected
    assert (got[n:] == -1).all()


@pytest.mark.parametrize("t", [1, 2, 4, 8])
@pytest.mark.parametrize("case", ["one", "two", "three", "all", "none", "random"])
def test_requests_follow_the_consumption_order(layer, t, case):
    if case == "random":
        ids = _routing(t, 7)[1]
    else:
        n_local = {"one": 1, "two": 2, "three": 3, "all": 8, "none": 0}[case]
        ids = _local_ids(t, n_local, 11 + t, E_LOCAL)
    experts = _entries(layer, ids)
    _check_order(layer, _dump(layer, ids), experts)


def test_every_byte_of_every_entry_once_and_nothing_else(layer):
    ids = _local_ids(2, 3, 5, E_LOCAL)
    experts = _entries(layer, ids)
    got = _dump(layer, ids)
    got = got[got[:, 0] >= 0]
    assert len(got) == len(experts) * REQS_PER_EXPERT
    assert int(got[:, 1].sum()) == len(experts) * EXPERT_BYTES
    for ptr, tensor, per_expert in (
        (layer.w13.data_ptr(), layer.w13, 2 * H * ROW_BYTES),
        (layer.sf13.data_ptr(), layer.sf13, 2 * H * SF_BYTES),
        (layer.w2.data_ptr(), layer.w2, H * ROW_BYTES),
        (layer.sf2.data_ptr(), layer.sf2, H * SF_BYTES),
    ):
        # Bytes (in 1 KiB granules) covered by the requests that start in the tensor.
        size = tensor.numel() * tensor.element_size()
        covered = np.zeros(size // 1024, dtype=np.int32)
        for addr, length in got:
            if ptr <= addr < ptr + size:
                assert addr + length <= ptr + size
                covered[(addr - ptr) // 1024 : (addr - ptr + length) // 1024] += 1
        want = np.zeros(E_LOCAL, dtype=np.int32)
        want[experts] = 1
        assert (covered.reshape(E_LOCAL, per_expert // 1024) == want[:, None]).all()


def test_a_remote_only_batch_prefetches_nothing(layer):
    ids = _local_ids(4, 0, 3, E_LOCAL)
    assert (_dump(layer, ids) == -1).all()


def test_an_expert_with_many_routes_is_one_entry(layer):
    ids = torch.full((8, TOP_K), 5, dtype=torch.int32, device=DEVICE)
    _check_order(layer, _dump(layer, ids), [5])


def test_duplicates_and_first_appearance_order(layer):
    ids = torch.tensor(
        [[6, E_LOCAL + 1, 2, 6, 7, E_LOCAL + 9, 2, 0]], dtype=torch.int32, device=DEVICE
    )
    assert _entries(layer, ids) == [6, 2, 7, 0]
    _check_order(layer, _dump(layer, ids), [6, 2, 7, 0])


def test_out_of_range_ids_are_remote(layer):
    ids = torch.tensor(
        [[-1, E_GLOBAL, E_GLOBAL + 7, 3, 1 << 30, 4, E_LOCAL, 5]],
        dtype=torch.int32,
        device=DEVICE,
    )
    _check_order(layer, _dump(layer, ids), [3, 4, 5])


def test_without_an_expert_map_the_ids_are_local(layer):
    ids = torch.tensor(
        [[3, 1, 7, E_LOCAL, E_LOCAL + 2, 3, 0, 100]], dtype=torch.int32, device=DEVICE
    )
    got = _dump(layer, ids, expert_map=None)
    _check_order(layer, got, [3, 1, 7, 0])


@pytest.mark.parametrize(
    "budget",
    [
        0,
        1,
        ROW_REQ - 1,
        ROW_REQ,
        5 * ROW_REQ + 7,
        BYTES_1 - 1,
        BYTES_1,
        BYTES_1 + 4 * ROW_REQ + 9,
    ],
)
def test_the_budget_takes_the_longest_prefix_that_fits(layer, budget):
    ids = _local_ids(1, 2, 9, E_LOCAL)
    expected = _expected(layer, _entries(layer, ids))
    take, total = 0, 0
    while take < len(expected) and total + expected[take][1] <= budget:
        total += expected[take][1]
        take += 1
    got = _dump(layer, ids, budget=budget)
    assert [tuple(x) for x in got[:take].tolist()] == expected[:take]
    assert (got[take:] == -1).all()


@pytest.mark.parametrize("num_ctas", [1, 3, 16, 188])
def test_the_grid_does_not_change_the_requests(layer, num_ctas):
    ids = _local_ids(2, 2, 4, E_LOCAL)
    _check_order(layer, _dump(layer, ids, num_ctas=num_ctas), _entries(layer, ids))


def test_rejects_bad_arguments(layer):
    ids = _local_ids(1, 1, 1, E_LOCAL)
    args = (layer.expert_map, layer.w13, layer.sf13, layer.w2, layer.sf2, False)
    with pytest.raises(RuntimeError):  # more than 8 tokens
        torch.ops._muse_C.expert_prefetch(
            torch.zeros(9, TOP_K, dtype=torch.int32, device=DEVICE),
            *args, 4, 1 << 20, False,
        )  # fmt: skip
    with pytest.raises(RuntimeError):  # int64 ids
        torch.ops._muse_C.expert_prefetch(ids.long(), *args, 4, 1 << 20, False)
    with pytest.raises(RuntimeError):  # no CTAs
        torch.ops._muse_C.expert_prefetch(ids, *args, 0, 1 << 20, False)
    with pytest.raises(RuntimeError):  # scales of another shape
        torch.ops._muse_C.expert_prefetch(
            ids, layer.expert_map, layer.w13, layer.sf2, layer.w2, layer.sf2,
            False, 4, 1 << 20, False,
        )  # fmt: skip


def _engine(layer: _Layer) -> MuseDecodeMoE:
    return MuseDecodeMoE(
        layer.w13, layer.sf13, layer.w2, layer.sf2, prm=layer.prm, gs=layer.gs
    )


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("t", [1, 4, 8])
@pytest.mark.parametrize("n_local", [0, 1, 3])
def test_graph_side_stream_changes_nothing(layer, t, n_local, pdl):
    """The prefetch behind a routing stand-in on a side stream, joined at the end,
    leaves the rows and the permute map of the decode kernel as they were."""
    engine = _engine(layer)
    u = _routing(t, 3)[0]
    ids = _local_ids(t, n_local, 21 + t, E_LOCAL)
    rows, perm = engine.run(u, ids, layer.expert_map)
    ref_rows, ref_perm = rows.clone(), perm.clone()

    side = torch.cuda.Stream()
    fork, handoff, join = (torch.cuda.Event() for _ in range(3))
    routed = torch.empty_like(ids)

    def body():
        fork.record()
        with torch.cuda.stream(side):
            fork.wait()
            routed.copy_(ids)  # the routing stand-in: the primary of the prefetch
            handoff.record()
            engine.prefetch(
                routed, layer.expert_map, num_ctas=16, budget_bytes=96 << 20, pdl=pdl
            )
        handoff.wait()
        out = engine.run(u, routed, layer.expert_map)
        join.record(side)
        join.wait()
        return out

    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        body()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_rows, out_perm = body()
    for _ in range(3):
        rows.zero_()
        perm.fill_(-2)
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(out_perm, ref_perm)
        local = ref_perm >= 0
        assert torch.equal(out_rows[local], ref_rows[local])


class _Runner:
    """Stands in for the MoE runner: records the stream and the arguments."""

    def __init__(self, queued: bool = True):
        self.queued = queued
        self.calls: list[tuple[torch.cuda.Stream, int]] = []

    def prefetch_routed_experts(self, topk_ids, budget_bytes) -> bool:
        self.calls.append((torch.cuda.current_stream(), budget_bytes))
        return self.queued


def test_the_default_prefetch_is_the_head_of_one_expert(layer):
    # A prefetch of whole experts competes with the kernel for HBM (see the module).
    assert 0 < EXPERT_PREFETCH_BYTES < EXPERT_BYTES
    assert 1 <= EXPERT_PREFETCH_CTAS <= 256
    ids = _local_ids(1, 3, 2, E_LOCAL)
    got = _dump(layer, ids, budget=EXPERT_PREFETCH_BYTES, num_ctas=EXPERT_PREFETCH_CTAS)
    expected = _expected(layer, _entries(layer, ids))
    n, total = 0, 0
    while total + expected[n][1] <= EXPERT_PREFETCH_BYTES:
        total += expected[n][1]
        n += 1
    assert [tuple(x) for x in got[:n].tolist()] == expected[:n]
    assert (got[n:] == -1).all()
    assert total > EXPERT_PREFETCH_BYTES - ROW_REQ


def test_the_branch_does_nothing_outside_a_capture():
    side = torch.cuda.Stream()
    runner = _Runner()
    branch = ExpertPrefetch(side, 123)
    ids = torch.zeros(1, TOP_K, dtype=torch.int32, device=DEVICE)
    branch.launch(runner, ids)
    branch.join()
    assert runner.calls == []


def test_the_branch_launches_on_its_stream_and_joins():
    side = torch.cuda.Stream()
    runner = _Runner()
    branch = ExpertPrefetch(side, 123)
    ids = torch.zeros(1, TOP_K, dtype=torch.int32, device=DEVICE)
    x = torch.ones(16, device=DEVICE)
    warm = torch.cuda.Stream()
    with torch.cuda.stream(warm):
        x + 1
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        # The side stream is part of the capture (as the router's aux stream is by the
        # time the prefetch is launched), and the prefetch is queued behind its work.
        fork = torch.cuda.Event()
        fork.record()
        with torch.cuda.stream(side):
            fork.wait()
            z = x * 2
        branch.launch(runner, ids)
        y = x + 1
        branch.join()
        # A second join without a launch in between adds nothing.
        branch.join()
    assert runner.calls == [(side, 123)]
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(y, torch.full_like(x, 2.0))
    assert torch.equal(z, torch.full_like(x, 2.0))


def test_a_launch_that_queued_nothing_leaves_nothing_to_join():
    side = torch.cuda.Stream()
    runner = _Runner(queued=False)
    branch = ExpertPrefetch(side, 1)
    ids = torch.zeros(1, TOP_K, dtype=torch.int32, device=DEVICE)
    x = torch.ones(16, device=DEVICE)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y = x + 1
        branch.launch(runner, ids)
        branch.join()
    assert len(runner.calls) == 1 and not branch._pending
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.equal(y, torch.full_like(x, 2.0))


@pytest.fixture
def capturing(monkeypatch):
    """The forward is a CUDA graph capture, the only place that queues the prefetch."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)


def _moe(prefetch, overlaps=True, aux=True, fp32_input=False, sp=False, engine=True):
    """A MuseSpark1xMoE with just the state ``wants_cast_router_input`` reads."""
    moe = object.__new__(model_ops.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    moe.expert_prefetch = object() if prefetch else None
    moe.experts = SimpleNamespace(has_muse_decode_engine=engine)
    moe.is_sequence_parallel = sp
    moe.router = SimpleNamespace(widens_bf16_input=lambda: True)
    moe.wants_fp32_router_input = lambda n: fp32_input
    moe.router_overlaps_pooler = lambda n: overlaps
    moe._routes_on_aux_stream = lambda n: aux
    return moe


@pytest.mark.parametrize("t", [1, 4, EXPERT_PREFETCH_MAX_TOKENS])
def test_the_prefetch_wants_the_cast_in_the_boundary(capturing, t):
    moe = _moe(prefetch=True)
    assert moe.prefetches_experts(t)
    assert moe.wants_cast_router_input(t)


def test_without_the_prefetch_the_cast_hides_beside_the_pooler(capturing):
    moe = _moe(prefetch=False)
    assert not moe.prefetches_experts(1)
    assert not moe.wants_cast_router_input(1)
    # Not beside the pooler (routing inside the experts' own stream): fused as before.
    assert _moe(prefetch=False, overlaps=False).wants_cast_router_input(1)


def test_the_prefetch_needs_the_muse_kernel_and_the_aux_routing(capturing):
    assert not _moe(prefetch=True).prefetches_experts(EXPERT_PREFETCH_MAX_TOKENS + 1)
    # Another expert path (CUTLASS, b12x, the engine not prepared yet): nothing to
    # prefetch for, so the boundary does not pay for the fused router cast either.
    assert not _moe(prefetch=True, engine=False).prefetches_experts(1)
    assert not _moe(prefetch=True, engine=False).wants_cast_router_input(1)
    assert not _moe(prefetch=True, aux=False).prefetches_experts(1)
    assert not _moe(prefetch=True, aux=False).wants_cast_router_input(1)
    assert not _moe(prefetch=True, sp=True).wants_cast_router_input(1)
    assert not _moe(prefetch=True, fp32_input=True).wants_cast_router_input(1)


def test_the_op_has_a_fake_implementation():
    """The module registers fakes for its ops so that a torch.compile trace of the
    model can pass them (``expert_prefetch`` only hints the cache and returns
    nothing)."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    def u8(*shape):
        return torch.empty(shape, dtype=torch.uint8, device=DEVICE)

    muse_c.load()
    with FakeTensorMode():
        ids = torch.zeros(1, TOP_K, dtype=torch.int32, device=DEVICE)
        out = torch.ops._muse_C.expert_prefetch(
            ids,
            None,
            u8(2, 8192, 2048),
            u8(2, 8192, 256),
            u8(2, 4096, 2048),
            u8(2, 4096, 256),
            True,
            EXPERT_PREFETCH_CTAS,
            EXPERT_PREFETCH_BYTES,
            False,
            None,
        )
    assert out is None


def test_an_eager_forward_neither_prefetches_nor_fuses_the_cast(monkeypatch):
    """Eager decode (enforce-eager, an uncaptured small mixed batch) queues no
    prefetch, so its boundary must not pay for the fused router cast either."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    moe = _moe(prefetch=True)
    assert not moe.prefetches_experts(1)
    assert not moe.wants_cast_router_input(1)
    # Where the cast is on the critical path anyway it is fused as before.
    assert _moe(prefetch=True, overlaps=False).wants_cast_router_input(1)


def _runner(native=True, use_muse=True, engine=True, kernel=True, queued=True):
    """A MuseMoERunner over fake SM120 experts, with just the state that
    ``has_muse_decode_engine`` and ``prefetch_routed_experts`` read."""
    calls = []
    experts = object.__new__(MuseSparkSm120NvFp4Experts)
    experts.use_muse_decode = use_muse
    experts.muse = object() if engine else None
    experts.prefetch_routed_experts = lambda ids, emap, budget: (
        calls.append((ids, emap, budget)) or queued
    )
    quant_method = object.__new__(MuseSparkModelOptNvFp4FusedMoE)
    quant_method.use_sm120_native = native
    quant_method.moe_kernel = SimpleNamespace(fused_experts=experts) if kernel else None
    runner = object.__new__(model_ops.MuseMoERunner)
    torch.nn.Module.__init__(runner)
    runner.routed_experts = SimpleNamespace(
        quant_method=quant_method, expert_map="the map"
    )
    return runner, calls


def test_the_runner_reports_the_muse_engine_of_native_experts_only():
    assert _runner()[0].has_muse_decode_engine
    for kwargs in (
        {"native": False},
        {"use_muse": False},  # another decode engine, or the engine failed to set up
        {"engine": False},  # not prepared yet
        {"kernel": False},  # the experts are not built yet
    ):
        assert not _runner(**kwargs)[0].has_muse_decode_engine, kwargs


def test_the_runner_hands_the_ids_and_the_map_to_the_native_experts():
    runner, calls = _runner()
    ids = torch.zeros(1, TOP_K, dtype=torch.int32, device=DEVICE)
    assert runner.prefetch_routed_experts(ids, 123)
    assert calls == [(ids, "the map", 123)]
    runner, calls = _runner(native=False)
    assert not runner.prefetch_routed_experts(ids, 123)
    assert calls == []
    assert not _runner(queued=False)[0].prefetch_routed_experts(ids, 123)


class _Experts(model_ops.MuseMoERunner):
    supports_after_gemm2_hook = True


def _model_with_layers(hookable: bool, aux: bool = True):
    layers = []
    for _ in range(3):
        moe = object.__new__(model_ops.MuseSpark1xMoE)
        torch.nn.Module.__init__(moe)
        experts = object.__new__(_Experts)
        torch.nn.Module.__init__(experts)
        experts.supports_after_gemm2_hook = hookable
        moe.experts = experts
        moe.expert_prefetch = None
        moe._router_aux_stream = torch.cuda.Stream() if aux else None
        layers.append(SimpleNamespace(feed_forward=moe))
    model = object.__new__(model_ops.MuseSpark1xModel)
    model.layers = layers
    model.start_layer, model.end_layer = 0, 3
    model._expert_prefetch = None
    return model


def test_the_model_shares_one_branch_between_the_hookable_layers():
    model = _model_with_layers(hookable=True)
    model._configure_expert_prefetch()
    branch = model._expert_prefetch
    assert isinstance(branch, ExpertPrefetch)
    assert all(layer.feed_forward.expert_prefetch is branch for layer in model.layers)
    assert branch.stream is model.layers[0].feed_forward._router_aux_stream


@pytest.mark.parametrize("hookable,aux", [(False, True), (True, False)])
def test_a_model_without_the_call_site_or_the_aux_stream_has_no_branch(hookable, aux):
    model = _model_with_layers(hookable=hookable, aux=aux)
    model._configure_expert_prefetch()
    assert model._expert_prefetch is None
    assert all(layer.feed_forward.expert_prefetch is None for layer in model.layers)


def test_the_default_budget_is_the_constant(monkeypatch):
    monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES", raising=False)
    # envs.py cannot import the constant, so it spells the value out.
    assert envs.environment_variables[
        "VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES"
    ]() == EXPERT_PREFETCH_BYTES
    model = _model_with_layers(hookable=True)
    model._configure_expert_prefetch()
    assert model._expert_prefetch.budget_bytes == EXPERT_PREFETCH_BYTES


@pytest.mark.parametrize("mib", [6, 20, 28])
def test_the_budget_follows_the_environment(monkeypatch, mib):
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES", str(mib << 20))
    model = _model_with_layers(hookable=True)
    model._configure_expert_prefetch()
    branch = model._expert_prefetch
    assert branch.budget_bytes == mib << 20
    assert all(layer.feed_forward.expert_prefetch is branch for layer in model.layers)


@pytest.mark.parametrize("value", ["0", "-4096"])
def test_a_budget_that_is_not_positive_is_refused(monkeypatch, value):
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES", value)
    model = _model_with_layers(hookable=True)
    with pytest.raises(ValueError, match="EXPERT_PREFETCH_BYTES"):
        model._configure_expert_prefetch()
    assert model._expert_prefetch is None
