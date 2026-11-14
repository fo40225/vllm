# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Residual boundaries fused into the TP collectives by the device communicator.

The model hands a boundary whose branch is still a partial sum (or this rank's
column block) to a communicator that offers ``ops/fused_boundary.py``'s interface, and
runs the two steps itself when the communicator declines. These tests use a fake
communicator whose "collectives" are exact (the all-reduce of one rank, the gather of
eight copies) and whose epilogue is the Triton kernel itself, so a routing mistake
shows as a different tensor or a wrong call count; the communicator's own kernels are
compared with the Triton ones bit for bit in the host plugin's tests.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.core.test_muse_spark_norms import (  # noqa: F401 (autouse dtype)
    HIDDEN,
    bf16_default_dtype,
    make_norm,
    make_residual_boundary,
)

import vllm.models.muse_spark.nvidia.model as model_ops
import vllm.models.muse_spark.nvidia.ops.fused_boundary as fused_boundary
from vllm.models.muse_spark.common.ops.fused_rmsnorm import (
    fused_norm_residual_norm_fwd,
    fused_rmsnorm_fwd,
)
from vllm.models.muse_spark.nvidia.model import (
    MuseSpark1xColumnShardedStream,
    MuseSpark1xPartialSumStream,
    MuseSpark1xProjectedStream,
    MuseSpark1xResidualStream,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
    norm_residual_then_norm,
)
from vllm.platforms import current_platform

TP_SIZE = 8

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the boundary is a Triton kernel",
)


class FakeCommunicator:
    """Implements the optional interface with exact collectives and Triton epilogues."""

    def __init__(self, *, max_tokens=64, returns_none=False):
        self.max_tokens = max_tokens
        self.returns_none = returns_none
        self.calls: list[str] = []

    def supports_fused_residual_boundary(self, kind, num_tokens):
        return num_tokens <= self.max_tokens

    def _boundary(self, branch, stream, out_fp32, **kw):
        out, new_stream = fused_norm_residual_norm_fwd(
            branch,
            stream,
            kw["post_weight"],
            kw["pre_weight"],
            kw["post_eps"],
            kw["pre_eps"],
            alpha=kw["alpha"],
            beta=kw["beta"],
            post_gain_center=kw["post_gain_center"],
            pre_gain_center=kw["pre_gain_center"],
            post_input_scale=kw["post_weight"] is not None,
            out_dtype=torch.bfloat16,
            out_fp32=out_fp32,
        )
        return out, new_stream

    def all_reduce_residual_boundary(self, branch, stream, *, out_fp32=None, **kw):
        self.calls.append("all_reduce_boundary")
        if self.returns_none:
            return None
        return self._boundary(branch, stream, out_fp32, **kw)

    def all_gather_residual_boundary(self, shard, stream, *, out_fp32=None, **kw):
        self.calls.append("all_gather_boundary")
        if self.returns_none:
            return None
        return self._boundary(gather(shard), stream, out_fp32, **kw)

    def all_gather_rmsnorm(self, shard, weight, eps, gain_center):
        self.calls.append("all_gather_rmsnorm")
        if self.returns_none:
            return None
        return fused_rmsnorm_fwd(gather(shard), weight, eps, gain_center=gain_center)

    # The GEMV the communicator runs itself: here the same bf16 product as the model's.
    def gemv_all_gather_residual_boundary(
        self, x, weight, stream, *, out_fp32=None, **kw
    ):
        self.calls.append("gemv_all_gather_boundary")
        if self.returns_none:
            return None
        return self._boundary(gather(x @ weight.t()), stream, out_fp32, **kw)

    def gemv_all_gather_rmsnorm(self, x, weight, norm_weight, eps, gain_center):
        self.calls.append("gemv_all_gather_rmsnorm")
        if self.returns_none:
            return None
        return fused_rmsnorm_fwd(
            gather(x @ weight.t()), norm_weight, eps, gain_center=gain_center
        )


def fp32_buffer(rows):
    return torch.empty(rows, HIDDEN, device="cuda", dtype=torch.float32)


def gather(shard):
    """The column concatenation of TP_SIZE equal blocks."""
    return torch.cat([shard] * TP_SIZE, dim=-1).contiguous()


@pytest.fixture
def two_step(monkeypatch):
    """The model's own collectives, exact: counted, not run."""
    calls = []

    def all_reduce(x):
        calls.append("all_reduce")
        return x

    def gather_columns(x):
        calls.append("gather_columns")
        return gather(x)

    monkeypatch.setattr(model_ops, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(model_ops, "gather_columns", gather_columns)
    monkeypatch.setattr(
        model_ops, "get_tensor_model_parallel_world_size", lambda: TP_SIZE
    )
    monkeypatch.setattr(model_ops, "gated_residual_next_rmsnorm", None)
    return calls


def use_communicator(monkeypatch, communicator):
    monkeypatch.setattr(fused_boundary, "_communicator", lambda: communicator)
    return communicator


def partial_sum(boundary):
    return MuseSpark1xPartialSumStream(
        stream=boundary.stream,
        branch=boundary.branch,
        post_norm=boundary.post_norm,
        connection=boundary.connection,
    )


@requires_cuda
@pytest.mark.parametrize("rows", [1, 8, 33, 64])
@pytest.mark.parametrize("post_input_scale", [False, True])
@pytest.mark.parametrize("fp32_copy", [False, True])
def test_all_reduce_and_boundary_run_in_the_communicator(
    monkeypatch, two_step, rows, post_input_scale, fp32_copy
):
    torch.manual_seed(rows)
    boundary, pre_norm = make_residual_boundary(rows, post_input_scale=post_input_scale)
    expected_fp32 = fp32_buffer(rows) if fp32_copy else None
    expected = norm_residual_then_norm(
        boundary, pre_norm, out_dtype=torch.bfloat16, out_fp32=expected_fp32
    )
    two_step.clear()

    communicator = use_communicator(monkeypatch, FakeCommunicator())
    out_fp32 = fp32_buffer(rows) if fp32_copy else None
    out, new_stream = norm_residual_then_norm(
        partial_sum(boundary), pre_norm, out_dtype=torch.bfloat16, out_fp32=out_fp32
    )

    assert communicator.calls == ["all_reduce_boundary"]
    assert two_step == []
    assert torch.equal(out, expected[0])
    assert torch.equal(new_stream, expected[1])
    if fp32_copy:
        assert torch.equal(out_fp32, expected_fp32)


@requires_cuda
@pytest.mark.parametrize(
    "decline",
    ["no communicator", "too many tokens", "returns None"],
)
def test_all_reduce_boundary_falls_back_to_the_two_steps(
    monkeypatch, two_step, decline
):
    rows = 8
    torch.manual_seed(1)
    boundary, pre_norm = make_residual_boundary(rows)
    expected = norm_residual_then_norm(boundary, pre_norm, out_dtype=torch.bfloat16)
    two_step.clear()

    communicator = None
    if decline == "too many tokens":
        communicator = FakeCommunicator(max_tokens=rows - 1)
    elif decline == "returns None":
        communicator = FakeCommunicator(returns_none=True)
    use_communicator(monkeypatch, communicator)
    out, new_stream = norm_residual_then_norm(
        partial_sum(boundary), pre_norm, out_dtype=torch.bfloat16
    )

    assert two_step == ["all_reduce"]
    assert torch.equal(out, expected[0])
    assert torch.equal(new_stream, expected[1])


@requires_cuda
def test_boundaries_the_triton_kernel_does_not_serve_keep_the_two_steps(
    monkeypatch, two_step
):
    """The communicator stands in for the Triton kernel only; all else is unchanged."""
    rows = 4
    torch.manual_seed(2)
    boundary, pre_norm = make_residual_boundary(rows)
    communicator = use_communicator(monkeypatch, FakeCommunicator())

    # an unrounded fp32 router input is another kernel's result
    router_input = fp32_buffer(rows)
    norm_residual_then_norm(
        partial_sum(boundary),
        pre_norm,
        out_dtype=torch.bfloat16,
        router_input=router_input,
    )
    # an fp32 output
    norm_residual_then_norm(partial_sum(boundary), pre_norm, out_dtype=torch.float32)
    # a post-norm with its weight after the normalization
    weighted_after = make_residual_boundary(rows)[0]
    weighted_after.post_norm = make_norm(MuseSpark1xRMSNorm, eps=1e-8)
    norm_residual_then_norm(
        partial_sum(weighted_after), pre_norm, out_dtype=torch.bfloat16
    )
    # a pre-norm without a weight
    weightless = make_norm(MuseSpark1xRMSNorm, has_weight=False, is_post_norm=False)
    norm_residual_then_norm(partial_sum(boundary), weightless, out_dtype=torch.bfloat16)
    # an fp16 branch
    half = partial_sum(boundary)
    half.branch = half.branch.half()
    norm_residual_then_norm(half, pre_norm, out_dtype=torch.bfloat16)
    # where FlashInfer's dual norm could run the boundary (a device other than SM120)
    monkeypatch.setattr(model_ops, "_on_sm120", lambda: False)
    monkeypatch.setattr(model_ops, "gated_residual_next_rmsnorm", object())
    monkeypatch.setattr(
        model_ops, "supports_gated_residual_next_rmsnorm", lambda *_: False
    )
    norm_residual_then_norm(partial_sum(boundary), pre_norm, out_dtype=torch.bfloat16)

    assert communicator.calls == []
    assert two_step.count("all_reduce") == 6


@requires_cuda
@pytest.mark.parametrize("rows", [1, 8, 32])
@pytest.mark.parametrize("post_input_scale", [False, True])
def test_column_gather_and_boundary_run_in_the_communicator(
    monkeypatch, two_step, rows, post_input_scale
):
    torch.manual_seed(rows)
    boundary, pre_norm = make_residual_boundary(rows, post_input_scale=post_input_scale)
    shard = boundary.branch[:, : HIDDEN // TP_SIZE].contiguous()
    full = MuseSpark1xResidualStream(
        stream=boundary.stream,
        branch=gather(shard),
        post_norm=boundary.post_norm,
        connection=boundary.connection,
    )
    expected = norm_residual_then_norm(full, pre_norm, out_dtype=torch.bfloat16)
    sharded = MuseSpark1xColumnShardedStream(
        stream=boundary.stream,
        branch=shard,
        post_norm=boundary.post_norm,
        connection=boundary.connection,
        all_gather=None,
    )

    communicator = use_communicator(monkeypatch, FakeCommunicator())
    out, new_stream = norm_residual_then_norm(
        sharded, pre_norm, out_dtype=torch.bfloat16
    )
    assert communicator.calls == ["all_gather_boundary"]
    assert two_step == []
    assert torch.equal(out, expected[0])
    assert torch.equal(new_stream, expected[1])

    # declined: the columns are gathered by the model, as before
    use_communicator(monkeypatch, FakeCommunicator(returns_none=True))
    out, new_stream = norm_residual_then_norm(
        sharded, pre_norm, out_dtype=torch.bfloat16
    )
    assert two_step == ["gather_columns"]
    assert torch.equal(out, expected[0])
    assert torch.equal(new_stream, expected[1])


@requires_cuda
def test_column_gather_boundary_is_not_used_for_a_router_input(
    monkeypatch, two_step
):
    boundary, pre_norm = make_residual_boundary(4)
    shard = boundary.branch[:, : HIDDEN // TP_SIZE].contiguous()
    sharded = MuseSpark1xColumnShardedStream(
        stream=boundary.stream,
        branch=shard,
        post_norm=boundary.post_norm,
        connection=boundary.connection,
        all_gather=None,
    )
    communicator = use_communicator(monkeypatch, FakeCommunicator())
    out_fp32 = fp32_buffer(4)
    norm_residual_then_norm(
        sharded, pre_norm, out_dtype=torch.bfloat16, out_fp32=out_fp32
    )
    assert communicator.calls == []
    assert two_step == ["gather_columns"]


def make_pooler_stub(monkeypatch, *, weight_first=False, has_weight=True):
    moe = object.__new__(model_ops.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    cls = MuseSpark1xRMSNormWithInputScale if weight_first else MuseSpark1xRMSNorm
    moe.pre_expert_norm = cls(
        4096,
        eps=1e-6,
        has_weight=has_weight,
        zero_centered_gamma=True,
        residual_id=3,
    ).cuda()
    if has_weight:
        with torch.no_grad():
            moe.pre_expert_norm.weight.copy_(
                torch.randn(4096, device="cuda", dtype=torch.bfloat16) * 0.1
            )
    moe.fuses_boundary_collectives = True
    return moe


@requires_cuda
@pytest.mark.parametrize("rows", [1, 5, 64])
def test_pooler_gather_and_norm_run_in_the_communicator(monkeypatch, rows):
    moe = make_pooler_stub(monkeypatch)
    shard = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    expected = moe.pre_expert_norm(gather(shard))

    communicator = use_communicator(monkeypatch, FakeCommunicator())
    assert torch.equal(moe._fused_all_gather_rmsnorm(shard), expected)
    assert communicator.calls == ["all_gather_rmsnorm"]


@requires_cuda
@pytest.mark.parametrize(
    "case", ["flag off", "no weight", "weight first", "too many tokens", "fp32"]
)
def test_pooler_gather_and_norm_decline(monkeypatch, case):
    moe = make_pooler_stub(
        monkeypatch, weight_first=case == "weight first", has_weight=case != "no weight"
    )
    moe.fuses_boundary_collectives = case != "flag off"
    shard = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    communicator = FakeCommunicator(max_tokens=4 if case == "too many tokens" else 64)
    use_communicator(monkeypatch, communicator)
    if case == "fp32":
        shard = shard.float()
    assert moe._fused_all_gather_rmsnorm(shard) is None
    assert communicator.calls == []


def make_layer_stub(*, fused=True, router_fp32=False, moe=True, column_gather=None):
    layer = object.__new__(model_ops.MuseSpark1xDecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.fuses_boundary_collectives = fused
    layer.fuses_column_gather = fused if column_gather is None else column_gather
    layer.all_gather_dual_norm = None
    if moe:
        layer.feed_forward = object.__new__(model_ops.MuseSpark1xMoE)
        torch.nn.Module.__init__(layer.feed_forward)
        layer.feed_forward.wants_fp32_router_input = lambda num_tokens: router_fp32
    else:
        layer.feed_forward = torch.nn.Identity()
    return layer


@pytest.mark.parametrize("moe", [True, False])
def test_all_reduce_is_deferred_where_the_communicator_fuses_it(monkeypatch, moe):
    use_communicator(monkeypatch, FakeCommunicator(max_tokens=64))
    layer = make_layer_stub(moe=moe)
    assert layer._defers_all_reduce(1, sp_active=False)
    assert layer._defers_all_reduce(64, sp_active=False)
    assert not layer._defers_all_reduce(65, sp_active=False)
    assert not layer._defers_all_reduce(8, sp_active=True)
    assert not make_layer_stub(fused=False, moe=moe)._defers_all_reduce(8, False)
    use_communicator(monkeypatch, None)
    assert not layer._defers_all_reduce(8, sp_active=False)


def test_all_reduce_is_not_deferred_for_the_unrounded_router_input(monkeypatch):
    use_communicator(monkeypatch, FakeCommunicator())
    assert not make_layer_stub(router_fp32=True)._defers_all_reduce(48, False)
    assert make_layer_stub(router_fp32=False)._defers_all_reduce(48, False)


def make_unpooler_stub(layer, shards):
    layer.feed_forward.shards_unpooler = lambda num_tokens: shards


def test_unpooler_shard_goes_to_the_next_boundary_where_the_communicator_gathers(
    monkeypatch,
):
    use_communicator(monkeypatch, FakeCommunicator(max_tokens=8))
    layer = make_layer_stub()
    make_unpooler_stub(layer, True)
    assert layer._defers_column_gather(1)
    assert layer._defers_column_gather(8)
    assert not layer._defers_column_gather(9)  # the communicator declines
    make_unpooler_stub(layer, False)
    assert not layer._defers_column_gather(1)  # the layer does not shard the unpooler
    use_communicator(monkeypatch, None)
    make_unpooler_stub(layer, True)
    assert not layer._defers_column_gather(1)  # no communicator that fuses
    use_communicator(monkeypatch, FakeCommunicator())
    assert not make_layer_stub(fused=False)._defers_column_gather(1)
    assert not make_layer_stub(moe=False)._defers_column_gather(1)


def test_unpooler_shard_stays_in_the_layer_without_fusion_across_layers(monkeypatch):
    """Pipeline parallelism (or the config switch): the model normalizes the stream of
    every layer right away, which needs the gathered branch."""
    use_communicator(monkeypatch, FakeCommunicator())
    layer = make_layer_stub(column_gather=False)
    make_unpooler_stub(layer, True)
    assert not layer._defers_column_gather(1)
    # a FlashInfer workspace keeps its own gate (it is only created under fusion)
    layer.all_gather_dual_norm = object()
    assert layer._defers_column_gather(1)


def make_projected(boundary, rows, k=64):
    """A projected stream whose GEMV is `x @ w.T` (this rank's rows of the weight)."""
    x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(HIDDEN // TP_SIZE, k, device="cuda", dtype=torch.bfloat16) * 0.1
    projected = MuseSpark1xProjectedStream(
        stream=boundary.stream,
        branch=x,
        post_norm=boundary.post_norm,
        connection=boundary.connection,
        all_gather=None,
        weight=w,
        project=lambda v: v @ w.t(),
    )
    return projected, x, w


@requires_cuda
@pytest.mark.parametrize("rows", [1, 3, 8])
@pytest.mark.parametrize("post_input_scale", [False, True])
def test_unpooler_gemv_and_column_gather_run_in_the_communicator(
    monkeypatch, two_step, rows, post_input_scale
):
    torch.manual_seed(rows)
    boundary, pre_norm = make_residual_boundary(rows, post_input_scale=post_input_scale)
    projected, x, w = make_projected(boundary, rows)
    full = MuseSpark1xResidualStream(
        stream=boundary.stream,
        branch=gather(x @ w.t()),
        post_norm=boundary.post_norm,
        connection=boundary.connection,
    )
    expected = norm_residual_then_norm(full, pre_norm, out_dtype=torch.bfloat16)
    two_step.clear()

    communicator = use_communicator(monkeypatch, FakeCommunicator())
    out, new_stream = norm_residual_then_norm(
        projected, pre_norm, out_dtype=torch.bfloat16
    )
    assert communicator.calls == ["gemv_all_gather_boundary"]
    assert two_step == []
    assert torch.equal(out, expected[0])
    assert torch.equal(new_stream, expected[1])

    # declined by the GEMV call: the layer's own GEMV, then the fused column gather
    communicator = use_communicator(monkeypatch, FakeCommunicator(returns_none=True))
    out, new_stream = norm_residual_then_norm(
        projected, pre_norm, out_dtype=torch.bfloat16
    )
    assert communicator.calls == ["gemv_all_gather_boundary", "all_gather_boundary"]
    assert two_step == ["gather_columns"]
    assert torch.equal(out, expected[0])
    assert torch.equal(new_stream, expected[1])

    # no communicator: GEMV, gather and boundary by the model
    two_step.clear()
    use_communicator(monkeypatch, None)
    out, new_stream = norm_residual_then_norm(
        projected, pre_norm, out_dtype=torch.bfloat16
    )
    assert two_step == ["gather_columns"]
    assert torch.equal(out, expected[0])
    assert torch.equal(new_stream, expected[1])
    # materialize (the speculative-decoding taps): the same branch
    assert torch.equal(projected.materialize().branch, full.branch)


@requires_cuda
def test_unpooler_gemv_is_not_fused_for_a_router_input_or_a_fp32_copy(
    monkeypatch, two_step
):
    boundary, pre_norm = make_residual_boundary(4)
    projected, x, w = make_projected(boundary, 4)
    communicator = use_communicator(monkeypatch, FakeCommunicator())
    norm_residual_then_norm(
        projected, pre_norm, out_dtype=torch.bfloat16, out_fp32=fp32_buffer(4)
    )
    assert communicator.calls == []
    assert two_step == ["gather_columns"]


@requires_cuda
def test_unpooler_gemv_declines_for_what_the_communicator_cannot_serve(
    monkeypatch, two_step
):
    boundary, pre_norm = make_residual_boundary(4)
    projected, x, w = make_projected(boundary, 4)
    communicator = use_communicator(monkeypatch, FakeCommunicator())
    # an fp16 activation is not what the kernels read
    projected.branch = x.half()
    projected.project = lambda v: (v @ w.half().t()).bfloat16()
    projected.weight = w.half()
    out, new_stream = norm_residual_then_norm(
        projected, pre_norm, out_dtype=torch.bfloat16
    )
    # the layer's own GEMV, then the communicator's column gather + boundary
    assert communicator.calls == ["all_gather_boundary"]
    assert two_step == []
    assert out.dtype == torch.bfloat16 and new_stream.dtype == torch.float32


def make_unpooler_moe_stub(rows, *, fuses=True):
    moe = object.__new__(model_ops.MuseSpark1xMoE)
    torch.nn.Module.__init__(moe)
    k = 64
    moe.unpooler = torch.nn.Module()
    moe.unpooler.output_size = HIDDEN
    moe.unpooler.weight = torch.nn.Parameter(
        torch.randn(HIDDEN, k, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    moe.tp_size = TP_SIZE
    moe.tp_rank = 3
    moe.fuses_gemv_gather = fuses
    moe._projection_gemm = lambda layer, x, weight, bias: x @ weight.t()
    return moe, torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)


@requires_cuda
@pytest.mark.parametrize("rows", [1, 8])
def test_deferred_unpooler_hands_its_input_to_the_boundary(monkeypatch, rows):
    moe, x = make_unpooler_moe_stub(rows)
    communicator = use_communicator(monkeypatch, FakeCommunicator())
    deferred = moe._deferred_unpooler_columns(x)
    assert isinstance(deferred, model_ops.MuseSpark1xUnpoolerInput)
    assert deferred.x is x
    shard_rows = HIDDEN // TP_SIZE
    assert torch.equal(
        deferred.weight, moe.unpooler.weight[3 * shard_rows : 4 * shard_rows]
    )
    assert torch.equal(deferred.project(x), x @ deferred.weight.t())
    assert communicator.calls == []


@requires_cuda
@pytest.mark.parametrize(
    "case", ["flag off", "no communicator", "too many tokens", "fp32", "strided"]
)
def test_deferred_unpooler_computes_its_columns_where_nothing_fuses(
    monkeypatch, case
):
    moe, x = make_unpooler_moe_stub(8, fuses=case != "flag off")
    use_communicator(
        monkeypatch,
        None
        if case == "no communicator"
        else FakeCommunicator(max_tokens=4 if case == "too many tokens" else 64),
    )
    if case == "fp32":
        x = x.float()
        moe.unpooler.weight.data = moe.unpooler.weight.data.float()
    if case == "strided":
        x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)[:, ::2]
        assert not x.is_contiguous()
    columns = moe._deferred_unpooler_columns(x)
    assert isinstance(columns, torch.Tensor)
    shard_rows = HIDDEN // TP_SIZE
    assert columns.shape == (8, shard_rows)


def make_pooler_gemv_stub(monkeypatch, *, k=64, fuses=True, **kwargs):
    moe = make_pooler_stub(monkeypatch, **kwargs)
    moe.fuses_gemv_gather = fuses
    moe.pooler = torch.nn.Module()
    moe.pooler.output_size = 4096
    moe.pooler.weight = torch.nn.Parameter(
        torch.randn(4096, k, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    moe.tp_size = TP_SIZE
    moe.tp_rank = 5
    return moe


@requires_cuda
@pytest.mark.parametrize("rows", [1, 5, 8])
def test_pooler_gemv_gather_and_norm_run_in_the_communicator(monkeypatch, rows):
    moe = make_pooler_gemv_stub(monkeypatch)
    x = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)
    shard_rows = 4096 // TP_SIZE
    w = moe.pooler.weight[5 * shard_rows : 6 * shard_rows]
    expected = moe.pre_expert_norm(gather(x @ w.t()))

    communicator = use_communicator(monkeypatch, FakeCommunicator())
    assert torch.equal(moe._fused_gemv_all_gather_rmsnorm(x), expected)
    assert communicator.calls == ["gemv_all_gather_rmsnorm"]


@requires_cuda
@pytest.mark.parametrize(
    "case",
    [
        "flag off",
        "no weight",
        "weight first",
        "too many tokens",
        "fp32",
        "returns None",
    ],
)
def test_pooler_gemv_gather_and_norm_decline(monkeypatch, case):
    moe = make_pooler_gemv_stub(
        monkeypatch,
        fuses=case != "flag off",
        weight_first=case == "weight first",
        has_weight=case != "no weight",
    )
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    if case == "fp32":
        x = x.float()
    communicator = FakeCommunicator(
        max_tokens=4 if case == "too many tokens" else 64,
        returns_none=case == "returns None",
    )
    use_communicator(monkeypatch, communicator)
    assert moe._fused_gemv_all_gather_rmsnorm(x) is None
    expected_calls = ["gemv_all_gather_rmsnorm"] if case == "returns None" else []
    assert communicator.calls == expected_calls


class GemvLimitedCommunicator(FakeCommunicator):
    """Fuses the GEMV kinds for fewer rows than the plain gathers (the host plugin's
    MUSE_TOPO_GEMV_MAX_ROWS is 4 where its gather kernels take 8)."""

    def __init__(self, *, gemv_max_tokens=4, **kwargs):
        super().__init__(**kwargs)
        self.gemv_max_tokens = gemv_max_tokens

    def supports_fused_residual_boundary(self, kind, num_tokens):
        if kind.startswith("gemv_"):
            return num_tokens <= self.gemv_max_tokens
        return super().supports_fused_residual_boundary(kind, num_tokens)


def make_unpooler_gemv_layer(*, fuses=True, column_gather=True):
    """A layer whose MoE block has a sharded unpooler (``make_unpooler_moe_stub``)."""
    layer = make_layer_stub(column_gather=column_gather)
    layer.feed_forward, _ = make_unpooler_moe_stub(1, fuses=fuses)
    layer.feed_forward.shards_unpooler = lambda num_tokens: True
    return layer


def run_deferred_unpooler(layer, rows):
    """Whether the layer hands the unpooler GEMV to the boundary for ``rows`` rows,
    the way ``MuseSpark1xDecoderLayer`` decides it."""
    if not layer._defers_column_gather(rows):
        return False
    x = torch.randn(rows, 64, device="cuda", dtype=torch.bfloat16)
    columns = layer.feed_forward._deferred_unpooler_columns(x)
    return isinstance(columns, model_ops.MuseSpark1xUnpoolerInput)


@requires_cuda
def test_the_l2_prefetch_predicts_the_fused_unpooler_gemv(monkeypatch):
    """``fuses_unpooler_gemv`` (what the L2 prefetch asks) agrees with the layer's own
    decision for every batch size, also where the GEMV limit is below the gather's."""
    use_communicator(monkeypatch, GemvLimitedCommunicator(gemv_max_tokens=4))
    layer = make_unpooler_gemv_layer()
    rows = [1, 2, 4, 5, 8, 9]
    assert [layer.fuses_unpooler_gemv(n) for n in rows] == [True] * 3 + [False] * 3
    assert [run_deferred_unpooler(layer, n) for n in rows] == [
        layer.fuses_unpooler_gemv(n) for n in rows
    ]


@requires_cuda
@pytest.mark.parametrize(
    "case",
    [
        "gemv flag off",
        "no communicator",
        "gemv declined",
        "no column gather",
        "unsharded unpooler",
        "fp32 weight",
        "no unpooler",
        "dense layer",
    ],
)
def test_the_l2_prefetch_keeps_the_unpooler_where_the_gemv_is_not_fused(
    monkeypatch, case
):
    gemv_max_tokens = 0 if case == "gemv declined" else 4
    use_communicator(
        monkeypatch,
        None
        if case == "no communicator"
        else GemvLimitedCommunicator(gemv_max_tokens=gemv_max_tokens),
    )
    layer = make_unpooler_gemv_layer(
        fuses=case != "gemv flag off", column_gather=case != "no column gather"
    )
    moe = layer.feed_forward
    if case == "unsharded unpooler":
        moe.shards_unpooler = lambda num_tokens: False
    if case == "fp32 weight":
        moe.unpooler.weight.data = moe.unpooler.weight.data.float()
    if case == "no unpooler":
        moe.unpooler = None
    if case == "dense layer":
        layer.feed_forward = torch.nn.Identity()
    assert not layer.fuses_unpooler_gemv(1)


@pytest.fixture
def flashinfer_dual_norm_exported(monkeypatch, two_step):
    """A FlashInfer that exports ``gated_residual_next_rmsnorm`` (0.7.2 does) whose
    kernel does not take this device (``supports_gated_residual_next_rmsnorm`` is False,
    as on SM120)."""
    monkeypatch.setattr(model_ops, "gated_residual_next_rmsnorm", object())
    monkeypatch.setattr(
        model_ops, "supports_gated_residual_next_rmsnorm", lambda *_: False
    )
    return two_step


@requires_cuda
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_sm120_keeps_the_communicator_boundaries_beside_an_exported_flashinfer_norm(
    monkeypatch, flashinfer_dual_norm_exported, rows
):
    """The FlashInfer symbol is present but its kernel never runs on SM120: the
    communicator-fused boundaries (attention all-reduce, unpooler gather, unpooler GEMV
    + gather) must stay on, as they were while FlashInfer did not export it."""
    two_step = flashinfer_dual_norm_exported
    torch.manual_seed(rows)
    boundary, pre_norm = make_residual_boundary(rows)
    expected = norm_residual_then_norm(boundary, pre_norm, out_dtype=torch.bfloat16)
    shard = boundary.branch[:, : HIDDEN // TP_SIZE].contiguous()
    sharded = MuseSpark1xColumnShardedStream(
        stream=boundary.stream,
        branch=shard,
        post_norm=boundary.post_norm,
        connection=boundary.connection,
        all_gather=None,
    )
    projected, x, w = make_projected(boundary, rows)
    two_step.clear()

    monkeypatch.setattr(model_ops, "_on_sm120", lambda: True)
    communicator = use_communicator(monkeypatch, FakeCommunicator())
    for stream, kind in (
        (partial_sum(boundary), "all_reduce_boundary"),
        (sharded, "all_gather_boundary"),
        (projected, "gemv_all_gather_boundary"),
    ):
        communicator.calls.clear()
        out, new_stream = norm_residual_then_norm(
            stream, pre_norm, out_dtype=torch.bfloat16
        )
        assert communicator.calls == [kind]
        assert two_step == []
        if kind == "all_reduce_boundary":
            assert torch.equal(out, expected[0])
            assert torch.equal(new_stream, expected[1])


@requires_cuda
def test_other_devices_keep_the_two_steps_where_the_flashinfer_norm_may_run(
    monkeypatch, flashinfer_dual_norm_exported
):
    """Off SM120 nothing changes: an exported FlashInfer symbol still routes the
    boundary away from the communicator (its kernel may run there)."""
    two_step = flashinfer_dual_norm_exported
    boundary, pre_norm = make_residual_boundary(4)
    monkeypatch.setattr(model_ops, "_on_sm120", lambda: False)
    communicator = use_communicator(monkeypatch, FakeCommunicator())
    norm_residual_then_norm(partial_sum(boundary), pre_norm, out_dtype=torch.bfloat16)
    assert communicator.calls == []
    assert two_step == ["all_reduce"]
    # and without the symbol (FlashInfer 0.7.0) the communicator takes it, any device
    monkeypatch.setattr(model_ops, "gated_residual_next_rmsnorm", None)
    two_step.clear()
    norm_residual_then_norm(partial_sum(boundary), pre_norm, out_dtype=torch.bfloat16)
    assert communicator.calls == ["all_reduce_boundary"]
    assert two_step == []


@requires_cuda
@pytest.mark.parametrize("symbol", [None, object()], ids=["no symbol", "symbol"])
@pytest.mark.parametrize("on_sm120", [False, True], ids=["other", "sm120"])
def test_the_boundary_gate_depends_on_the_device_not_on_the_tensors(
    monkeypatch, symbol, on_sm120
):
    """``_triton_boundary_kwargs`` is the one gate of the three communicator boundaries:
    closed only where the FlashInfer norm is exported and the device is not SM120."""
    boundary, pre_norm = make_residual_boundary(4)
    monkeypatch.setattr(model_ops, "gated_residual_next_rmsnorm", symbol)
    monkeypatch.setattr(model_ops, "_on_sm120", lambda: on_sm120)
    kwargs = model_ops._triton_boundary_kwargs(
        boundary.stream,
        boundary.post_norm,
        boundary.connection,
        pre_norm,
        tuple(boundary.branch.shape),
        torch.bfloat16,
        None,
    )
    assert (kwargs is None) == (symbol is not None and not on_sm120)


@requires_cuda
@pytest.mark.parametrize("rows", [1, 2, 4])
def test_the_l2_prefetch_hint_matches_the_gate_beside_an_exported_flashinfer_norm(
    monkeypatch, flashinfer_dual_norm_exported, rows
):
    """Where ``fuses_unpooler_gemv`` (the L2 prefetch drops the unpooler shard then)
    answers True, the boundary after the layer really runs the communicator's GEMV."""
    two_step = flashinfer_dual_norm_exported
    monkeypatch.setattr(model_ops, "_on_sm120", lambda: True)
    communicator = use_communicator(monkeypatch, GemvLimitedCommunicator())
    layer = make_unpooler_gemv_layer()
    assert layer.fuses_unpooler_gemv(rows)
    boundary, pre_norm = make_residual_boundary(rows)
    projected, _, _ = make_projected(boundary, rows)
    two_step.clear()
    norm_residual_then_norm(projected, pre_norm, out_dtype=torch.bfloat16)
    assert communicator.calls == ["gemv_all_gather_boundary"]
    assert two_step == []


@pytest.mark.parametrize("on_sm120", [False, True], ids=["other", "sm120"])
def test_flashinfer_unpooler_gather_workspace_is_not_built_on_sm120(
    monkeypatch, on_sm120
):
    """The FlashInfer gather + dual norm would shadow the communicator's gather
    boundary (``_defers_column_gather`` takes the workspace branch first)."""
    built = []
    workspace = SimpleNamespace(compile=lambda **_: None)

    def create(*args, **kwargs):
        built.append(args)
        return workspace

    monkeypatch.setattr(model_ops, "maybe_create_all_gather_dual_norm", create)
    monkeypatch.setattr(model_ops, "_on_sm120", lambda: on_sm120)
    layer = make_layer_stub()
    layer.feed_forward.shard_unpooler = True
    layer.post_ffn_norm = SimpleNamespace(
        has_weight=False,
        variance_epsilon=1e-8,
        fused_gain_center=0.0,
    )
    model = SimpleNamespace(
        layers=[layer],
        start_layer=0,
        end_layer=1,
        fuse_across_layers=True,
        config=SimpleNamespace(hidden_size=HIDDEN),
        norm=SimpleNamespace(variance_epsilon=1e-5, fused_gain_center=0.0),
    )
    model_ops.MuseSpark1xModel._configure_column_gather(model)
    assert bool(built) == (not on_sm120)
    assert (layer.all_gather_dual_norm is workspace) == (not on_sm120)
