# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The aux taps of Muse Spark are made in the model dtype (CPU only).

The residual stream is FP32 and so were the taps a drafter takes, 5 x [T, 8192] FP32
(1.25 GiB at 8192 tokens) held through the target forward and the draft step, with
the cast copies of all of them on top at the draft step. Every consumer casts the taps
to the model dtype before its first matmul, so the target casts them where it makes
them: the same bits, at half the width, and the cast at the draft step is a no-op.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.models.muse_spark.nvidia.model as model_ops
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

pytestmark = pytest.mark.cpu_test

HIDDEN = 64
TOKENS = 6
NUM_LAYERS = 4


def _model(aux_dtype, aux_layers=(2, 4)) -> model_ops.MuseSpark1xModel:
    """NUM_LAYERS layers that add 1/3 to the stream (so FP32 and BF16 differ)."""
    model = object.__new__(model_ops.MuseSpark1xModel)
    model.start_layer = 0
    model.end_layer = NUM_LAYERS
    model.layers = [(lambda positions, x: x + 1.0 / 3.0) for _ in range(NUM_LAYERS)]
    model.residual_dtype = torch.float32
    model.default_dtype = torch.bfloat16
    model.aux_hidden_state_dtype = aux_dtype
    model.aux_hidden_state_layers = aux_layers
    model._aux_upstream_total_cached = 0
    model.is_sequence_parallel = False
    model.is_sequence_parallel_input = False
    model.sp_min_tokens = 1
    model.sp_row_gate = None
    model.fuse_across_layers = True
    model._l2_prefetcher = None
    model._expert_prefetch = None
    model.norm = lambda x, out_dtype=None: x.to(out_dtype or x.dtype)
    return model


@pytest.fixture(autouse=True)
def _single_pipeline_stage(monkeypatch):
    monkeypatch.setattr(
        model_ops,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True, world_size=1),
    )


def _forward(model):
    generator = torch.Generator().manual_seed(0)
    inputs = torch.randn(TOKENS, HIDDEN, generator=generator).to(torch.bfloat16)
    positions = torch.arange(TOKENS)
    return model.forward(None, positions, None, inputs_embeds=inputs)


def test_taps_are_made_in_the_model_dtype():
    _, taps = _forward(_model(torch.bfloat16))
    _, fp32_taps = _forward(_model(None))

    assert [t.dtype for t in fp32_taps] == [torch.float32] * 2
    assert [t.dtype for t in taps] == [torch.bfloat16] * 2
    # The bits are those of the cast the drafter used to do.
    for tap, fp32_tap in zip(taps, fp32_taps, strict=True):
        assert torch.equal(tap, fp32_tap.to(torch.bfloat16))


def test_the_taps_of_the_start_layer_are_cast_too():
    _, taps = _forward(_model(torch.bfloat16, aux_layers=(0, 2)))
    assert [t.dtype for t in taps] == [torch.bfloat16] * 2


def test_a_cast_and_an_uncast_tap_give_the_drafter_the_same_features():
    """What the drafter feeds its fc: the concat of the taps in the model dtype."""
    _, taps = _forward(_model(torch.bfloat16))
    _, fp32_taps = _forward(_model(None))
    speculator = object.__new__(DFlashSpeculator)
    speculator.dtype = torch.bfloat16
    seen = []
    speculator.model = SimpleNamespace(
        combine_hidden_states=lambda features: seen.append(features) or features
    )

    speculator.combine_aux_hidden_states(taps)
    speculator.combine_aux_hidden_states(fp32_taps)

    assert seen[0].dtype == seen[1].dtype == torch.bfloat16
    assert torch.equal(seen[0], seen[1])


def test_a_stage_that_keeps_the_stream_dtype_hands_the_tensor_on_as_it_is():
    state = torch.ones(2, 2, dtype=torch.float32)
    assert _model(None)._as_aux_tap(state) is state
    assert _model(torch.bfloat16)._as_aux_tap(state).dtype == torch.bfloat16


class _SPLayer:
    """A layer that adds 1/3 to the stream and takes part in the sequence-parallel
    stream like the real ones (``is_sequence_parallel``)."""

    is_sequence_parallel = False

    def __call__(self, positions, x):
        return x + 1.0 / 3.0


def _sp_model(aux_layers=(2, 4)) -> model_ops.MuseSpark1xModel:
    model = _model(torch.bfloat16, aux_layers)
    model.layers = [_SPLayer() for _ in range(NUM_LAYERS)]
    return model


def test_the_combiner_replaces_the_taps_with_its_projection():
    model = _sp_model()
    seen = []

    def combiner(taps, local_rows):
        seen.append((len(taps), local_rows, taps[0].shape[0]))
        return torch.cat(taps, dim=-1)[:, :HIDDEN] * 2

    model.set_aux_combiner(combiner)
    output, aux = _forward(model)
    _, plain = _forward(_sp_model())

    # One tensor, from the taps of all the rows (no sequence-parallel stream here).
    assert seen == [(2, False, TOKENS)]
    assert len(aux) == 1 and output.shape == (TOKENS, HIDDEN)
    assert torch.equal(aux[0], torch.cat(plain, dim=-1)[:, :HIDDEN] * 2)


def test_no_combiner_returns_the_taps_as_before():
    _, aux = _forward(_sp_model())
    assert len(aux) == 2


def test_with_the_sequence_parallel_stream_the_combiner_projects_the_local_rows(
    monkeypatch,
):
    """Under the sequence-parallel stream the taps are the rank's own rows; the one
    projected result is gathered, instead of every tap."""
    gathers = []
    local_rows = TOKENS // 2

    def sp_shard(x):
        return x[:local_rows]

    def sp_all_gather(x):
        gathers.append(tuple(x.shape))
        return torch.cat([x, x], dim=0)

    monkeypatch.setattr(model_ops, "sp_shard", sp_shard)
    monkeypatch.setattr(model_ops, "sp_all_gather", sp_all_gather)
    monkeypatch.setattr(model_ops, "_sequence_parallel_active", lambda *a: True)

    model = _sp_model()
    model.is_sequence_parallel = True
    model.is_sequence_parallel_input = True
    seen = []

    def combiner(taps, local):
        seen.append((len(taps), local, taps[0].shape[0]))
        return torch.cat(taps, dim=-1)[:, :HIDDEN]

    model.set_aux_combiner(combiner)
    output, aux = _forward(model)

    assert seen == [(2, True, local_rows)]
    # The output gather and the gather of the one projected tensor: not one per tap.
    assert gathers == [(local_rows, HIDDEN), (local_rows, HIDDEN)]
    assert len(aux) == 1 and aux[0].shape == (TOKENS, HIDDEN)

    gathers.clear()
    plain_model = _sp_model()
    plain_model.is_sequence_parallel = True
    plain_model.is_sequence_parallel_input = True
    _, plain = _forward(plain_model)
    assert len(gathers) == 1 + 2 and len(plain) == 2


def test_the_combiner_only_runs_where_taps_are_made():
    # A forward without aux layers has no taps to combine.
    model = _sp_model(aux_layers=())
    called = []
    model.set_aux_combiner(lambda taps, local: called.append(1))
    output = _forward(model)
    assert not called
    assert isinstance(output, torch.Tensor)
