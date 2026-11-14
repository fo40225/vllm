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
    # where FlashInfer's dual norm could run the boundary
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
