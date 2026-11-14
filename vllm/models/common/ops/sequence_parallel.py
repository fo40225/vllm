# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache

import torch

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
    tensor_model_parallel_all_gather,
    tensor_model_parallel_reduce_scatter,
)
from vllm.distributed.device_communicators.all_reduce_utils import (
    should_nccl_symm_mem_ag_rs,
)


def _custom_collective(name: str, x: torch.Tensor) -> torch.Tensor | None:
    device_communicator = get_tp_group().device_communicator
    if device_communicator is None:
        return None
    collective = getattr(device_communicator, name, None)
    return None if collective is None else collective(x)


@lru_cache
def is_sequence_parallel_active(
    is_enabled: bool, num_tokens: int, min_tokens: int | None = None
) -> bool:
    """Whether a batch of ``num_tokens`` shards the stream along the sequence.

    Below ``min_tokens`` (default: one token per rank, under which the shards
    are mostly padding) the batch stays replicated.
    """
    if min_tokens is None:
        min_tokens = get_tensor_model_parallel_world_size()
    return is_enabled and num_tokens >= min_tokens


def sp_all_gather(x: torch.Tensor) -> torch.Tensor:
    output = _custom_collective("custom_all_gather", x)
    if output is not None:
        return output
    return tensor_model_parallel_all_gather(x, 0)


def sp_all_gather_columns(x: torch.Tensor) -> torch.Tensor:
    """Concatenate the ranks' ``[T, w]`` column blocks into ``[T, tp * w]``.

    The communicator is asked for the column layout (``dim=-1``), so one that can
    write it directly skips the reorder copy. NCCL gathers rank-major and reorders
    exactly as the model did before (a view for a single token), which keeps the
    result bit-identical. The custom all-gather and NCCL symmetric memory only
    gather along dim 0, so they keep the gather-then-reorder route.
    """
    assert x.ndim == 2
    output = _custom_collective("custom_all_gather", x)
    if output is None and not should_nccl_symm_mem_ag_rs():
        return tensor_model_parallel_all_gather(x, -1)
    if output is None:
        output = tensor_model_parallel_all_gather(x, 0)
    tp_size = get_tensor_model_parallel_world_size()
    num_tokens, shard_size = x.shape
    return (
        output.view(tp_size, num_tokens, shard_size)
        .transpose(0, 1)
        .reshape(num_tokens, tp_size * shard_size)
    )


def sp_reduce_scatter(x: torch.Tensor) -> torch.Tensor:
    assert x.ndim == 2
    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        x = torch.nn.functional.pad(x, (0, 0, 0, sp_pad))
    output = _custom_collective("custom_reduce_scatter", x)
    if output is not None:
        return output
    return tensor_model_parallel_reduce_scatter(x, 0)


def sp_shard(x: torch.Tensor) -> torch.Tensor:
    tp_size = get_tensor_model_parallel_world_size()
    tp_rank = get_tensor_model_parallel_rank()
    sp_pad = (-x.shape[0]) % tp_size
    if sp_pad > 0:
        pad = (0, 0) * (x.ndim - 1) + (0, sp_pad)
        x = torch.nn.functional.pad(x, pad)
    chunk = x.shape[0] // tp_size
    return x[tp_rank * chunk : (tp_rank + 1) * chunk]


def sp_padding_mask(
    is_padding: torch.Tensor | None,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    num_tokens = hidden_states.shape[0]
    if is_padding is None:
        is_padding = hidden_states.new_zeros(num_tokens, dtype=torch.bool)
    assert is_padding.shape[0] == num_tokens

    tp_size = get_tensor_model_parallel_world_size()
    sp_pad = (-num_tokens) % tp_size
    if sp_pad > 0:
        is_padding = torch.nn.functional.pad(is_padding, (0, sp_pad), value=True)
    chunk = is_padding.shape[0] // tp_size
    tp_rank = get_tensor_model_parallel_rank()
    return is_padding[tp_rank * chunk : (tp_rank + 1) * chunk]
