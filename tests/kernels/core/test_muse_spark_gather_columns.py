# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Column-block all-gather of the sharded Muse Spark projections.

``gather_columns`` asks the TP communicator for the ``[T, tp * w]`` column layout
(``dim=-1``) instead of gathering rank-major and reordering in the model. These
tests check that every route yields exactly the tensor of the old
gather-then-reorder code, without a GPU or a process group.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.models.common.ops.sequence_parallel as sp_ops
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.models.muse_spark.nvidia.model import gather_columns

TP_SIZE = 8
TOKENS = [1, 2, 3, 8, 17, 64, 128]
WIDTHS = [512, 1024]


def make_shards(num_tokens: int, width: int) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(num_tokens * 7919 + width)
    return [
        torch.randn(num_tokens, width, generator=generator).to(torch.bfloat16)
        for _ in range(TP_SIZE)
    ]


def reference_columns(shards: list[torch.Tensor]) -> torch.Tensor:
    """The code gather_columns replaced: rank-major gather, then the reorder."""
    num_tokens, width = shards[0].shape
    return (
        torch.cat(shards)
        .view(TP_SIZE, num_tokens, width)
        .transpose(0, 1)
        .reshape(num_tokens, TP_SIZE * width)
    )


def fake_nccl_all_gather(shards: list[torch.Tensor]):
    """CudaCommunicator.all_gather on a stubbed NCCL that fills rank-major rows."""

    def nccl_all_gather(output: torch.Tensor, input_: torch.Tensor) -> None:
        output.copy_(torch.cat(shards))

    communicator = SimpleNamespace(
        pynccl_comm=SimpleNamespace(disabled=False, all_gather=nccl_all_gather),
        world_size=TP_SIZE,
    )

    def all_gather(input_: torch.Tensor, dim: int) -> torch.Tensor:
        return CudaCommunicator.all_gather(communicator, input_, dim)

    return all_gather


@pytest.fixture
def tp_group(monkeypatch):
    """A TP group of 8 whose communicator has no custom all-gather."""
    device_communicator = SimpleNamespace()
    monkeypatch.setattr(
        sp_ops,
        "get_tp_group",
        lambda: SimpleNamespace(device_communicator=device_communicator),
    )
    monkeypatch.setattr(sp_ops, "get_tensor_model_parallel_world_size", lambda: TP_SIZE)
    monkeypatch.setattr(sp_ops, "should_nccl_symm_mem_ag_rs", lambda: False)
    return device_communicator


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_requests_the_column_layout_from_the_communicator(
    monkeypatch, tp_group, num_tokens, width
):
    shards = make_shards(num_tokens, width)
    all_gather = Mock(side_effect=lambda x, dim: torch.cat(shards, dim=dim))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", all_gather)

    output = gather_columns(shards[0])

    all_gather.assert_called_once_with(shards[0], -1)
    assert output.shape == (num_tokens, TP_SIZE * width)
    assert torch.equal(output, reference_columns(shards))


@pytest.mark.parametrize("width", WIDTHS)
@pytest.mark.parametrize("num_tokens", TOKENS)
def test_nccl_communicator_matches_the_old_gather_then_reorder(
    monkeypatch, tp_group, num_tokens, width
):
    shards = make_shards(num_tokens, width)
    monkeypatch.setattr(
        sp_ops, "tensor_model_parallel_all_gather", fake_nccl_all_gather(shards)
    )

    output = gather_columns(shards[0])

    assert torch.equal(output, reference_columns(shards))


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_custom_all_gather_keeps_the_rank_major_route(
    monkeypatch, tp_group, num_tokens
):
    shards = make_shards(num_tokens, 512)
    tp_group.custom_all_gather = Mock(return_value=torch.cat(shards))
    fallback = Mock(side_effect=AssertionError("unexpected fallback"))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", fallback)

    output = gather_columns(shards[0])

    tp_group.custom_all_gather.assert_called_once_with(shards[0])
    fallback.assert_not_called()
    assert torch.equal(output, reference_columns(shards))


def test_custom_all_gather_declining_falls_back_to_the_column_gather(
    monkeypatch, tp_group
):
    shards = make_shards(8, 512)
    tp_group.custom_all_gather = Mock(return_value=None)
    all_gather = Mock(side_effect=lambda x, dim: torch.cat(shards, dim=dim))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", all_gather)

    output = gather_columns(shards[0])

    all_gather.assert_called_once_with(shards[0], -1)
    assert torch.equal(output, reference_columns(shards))


@pytest.mark.parametrize("num_tokens", [1, 8])
def test_nccl_symmetric_memory_keeps_the_rank_major_route(
    monkeypatch, tp_group, num_tokens
):
    # The symmetric-memory gather only handles dim 0.
    shards = make_shards(num_tokens, 512)
    monkeypatch.setattr(sp_ops, "should_nccl_symm_mem_ag_rs", lambda: True)
    all_gather = Mock(side_effect=lambda x, dim: torch.cat(shards, dim=dim))
    monkeypatch.setattr(sp_ops, "tensor_model_parallel_all_gather", all_gather)

    output = gather_columns(shards[0])

    all_gather.assert_called_once_with(shards[0], 0)
    assert torch.equal(output, reference_columns(shards))
