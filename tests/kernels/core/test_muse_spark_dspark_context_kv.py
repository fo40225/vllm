# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The per-layer context of the Muse Spark DSpark draft is filled in place (CPU only).

``torch.stack`` of one normed copy per layer held every copy twice at the K/V
precompute of a long prefill chunk (5 x 128 MiB each at 8192 context rows, 1.25 GiB
for the list and the stack together); the context is now filled layer by layer. The
values do not change.
"""

import pytest
import torch
from torch import nn

from vllm.models.muse_spark.common.dspark import MuseSparkDSparkModel

pytestmark = pytest.mark.cpu_test

LAYERS = 5
TOKENS = 12
HIDDEN = 32
KV = 8


class _Norm(nn.Module):
    """A distinct row-wise norm per layer (a weight, a scale and an eps)."""

    def __init__(self, layer: int, out_dtype: torch.dtype):
        super().__init__()
        generator = torch.Generator().manual_seed(layer)
        self.weight = torch.randn(HIDDEN, generator=generator)
        self.out_dtype = out_dtype
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        x32 = x.float()
        normed = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (normed * self.weight).to(self.out_dtype)


def _model(out_dtype: torch.dtype) -> MuseSparkDSparkModel:
    model = object.__new__(MuseSparkDSparkModel)
    nn.Module.__init__(model)
    model.hidden_norm = _Norm(100, torch.bfloat16)
    model._context_pre_norms = tuple(_Norm(i, out_dtype) for i in range(LAYERS))
    generator = torch.Generator().manual_seed(7)
    model._stacked_kv_weight = torch.randn(
        LAYERS, 2 * KV, HIDDEN, generator=generator
    ).to(out_dtype)
    return model


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32])
def test_the_filled_context_is_the_stacked_one(out_dtype):
    model = _model(out_dtype)
    generator = torch.Generator().manual_seed(1)
    context = torch.randn(TOKENS, HIDDEN, generator=generator).to(torch.bfloat16)

    out = model._project_muse_context_kv(context)

    fc_output = model.hidden_norm(context)
    stacked = torch.stack([norm(fc_output) for norm in model._context_pre_norms])
    expected = torch.bmm(stacked, model._stacked_kv_weight.transpose(1, 2))
    assert out.shape == (LAYERS, TOKENS, 2 * KV)
    assert out.dtype == out_dtype
    assert torch.equal(out, expected)


def test_each_layer_norm_runs_once():
    model = _model(torch.bfloat16)
    model._project_muse_context_kv(torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16))
    assert [norm.calls for norm in model._context_pre_norms] == [1] * LAYERS
