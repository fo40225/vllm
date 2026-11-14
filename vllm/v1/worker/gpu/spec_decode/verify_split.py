# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifying the drafts of some requests of a step from the top-K candidates and of the
others from the whole logits (``VLLM_MUSE_SPARK_SM120_VERIFY_TOPK_LOGITS`` 3 and 4).

A step in which one request needs the logits (penalties, a top_k above K, ...) used to
send every row to the full path. The verification of a request reads its own rows only
(its logits, its draft tokens and positions, its sampling state, its seed), so a step
can be cut into two request subsets that are verified by the two routes and written
back to their places in the batch: the candidates of the first subset are gathered
(8 bytes per row and rank and candidate) and the logits of the second only.

The pieces here are the batch views the two verifiers get and the merge of their
outputs; the routing decision is host state (``Sampler.topk_verify_split_plan``) and the
gathers are ``MuseSpark1xForCausalLM.compute_split_verify_logits``.
"""

import copy

import numpy as np
import torch

from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.sample.output import SamplerOutput

# A split step has a fixed cost of about 0.5 ms (35 small launches: four pinned
# host-to-device copies, 16 view gathers, two index_select, the merge) and saves about
# 0.11 ms per candidate request (14 us per row, 8 rows at K = 7): it is repaid from
# about 5 candidate requests. Fewer than this many candidate requests: the whole step
# takes the full path.
SPLIT_MIN_CANDIDATE_REQS = 5


def request_rows(cu_num_logits_np: np.ndarray, reqs: np.ndarray) -> np.ndarray:
    """The logit rows of the requests ``reqs`` (batch positions), request by request."""
    starts = cu_num_logits_np[reqs].astype(np.int64)
    counts = cu_num_logits_np[reqs + 1].astype(np.int64) - starts
    first = np.repeat(starts - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    return first + np.arange(int(counts.sum()), dtype=np.int64)


class SplitIndices:
    """The index tensors of a request subset, built with one host-to-device copy."""

    def __init__(self, batch: InputBatch, reqs: np.ndarray, device: torch.device):
        cu = batch.cu_num_logits_np
        self.reqs = reqs
        self.rows = request_rows(cu, reqs)
        counts = (cu[reqs + 1] - cu[reqs]).astype(np.int64)
        self.cu_np = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
        n, m = reqs.shape[0], self.rows.shape[0]
        packed = np.concatenate([reqs.astype(np.int64), self.rows])
        packed = async_tensor_h2d(packed, device=device)
        self.reqs_t = packed[:n]
        self.rows_t = packed[n : n + m]
        self.cu_t = async_tensor_h2d(self.cu_np, device=device)


def sampler_view(batch: InputBatch, sub: SplitIndices) -> InputBatch:
    """The part of ``batch`` that a verifier reads, for the requests of ``sub`` only.

    The verifiers (the rejection sampler and ``verify_topk``) read, per request, the
    state index, the sequence length and the logit rows, and per logit row the input
    token (the draft token), the position and the index in the request: these are
    cut down to the subset, with ``logits_indices`` made the identity over the subset's
    rows (``input_ids`` and ``positions`` hold the logit rows only). The other fields
    are the full batch's and mean nothing for the view.
    """
    reqs, rows = sub.reqs, sub.rows
    token_rows = batch.logits_indices[sub.rows_t]
    view = copy.copy(batch)
    view.req_ids = [batch.req_ids[r] for r in reqs.tolist()]
    view.num_reqs = view.num_reqs_after_padding = reqs.shape[0]
    view.idx_mapping = batch.idx_mapping[sub.reqs_t]
    view.idx_mapping_np = batch.idx_mapping_np[reqs]
    view.expanded_idx_mapping = batch.expanded_idx_mapping[sub.rows_t]
    view.expanded_local_pos = batch.expanded_local_pos[sub.rows_t]
    view.num_scheduled_tokens = batch.num_scheduled_tokens[reqs]
    view.num_tokens = view.num_tokens_after_padding = rows.shape[0]
    view.num_draft_tokens = rows.shape[0] - reqs.shape[0]
    if batch.num_draft_tokens_per_req is not None:
        view.num_draft_tokens_per_req = batch.num_draft_tokens_per_req[reqs]
    view.seq_lens = batch.seq_lens[sub.reqs_t]
    view.seq_lens_cpu_upper_bound = batch.seq_lens_cpu_upper_bound[
        torch.from_numpy(reqs)
    ]
    view.input_ids = batch.input_ids[token_rows]
    view.positions = batch.positions[token_rows]
    view.logits_indices = torch.arange(
        rows.shape[0], dtype=torch.int64, device=token_rows.device
    )
    view.cu_num_logits = sub.cu_t
    view.cu_num_logits_np = sub.cu_np
    return view


def merge_outputs(
    num_reqs: int,
    parts: list[tuple[SplitIndices, SamplerOutput]],
) -> SamplerOutput:
    """The sampler output of the whole batch from the outputs of its request subsets.

    Only what the verifiers of the split path return: the sampled tokens and the number
    of sampled and rejected tokens per request (a step with logprobs, a sampling mask or
    NaN counts is never split).
    """
    first = parts[0][1]
    sampled = torch.empty(
        (num_reqs, first.sampled_token_ids.shape[1]),
        dtype=first.sampled_token_ids.dtype,
        device=first.sampled_token_ids.device,
    )
    num_sampled = torch.empty(
        num_reqs, dtype=first.num_sampled.dtype, device=first.num_sampled.device
    )
    num_rejected = torch.empty_like(num_sampled)
    for sub, out in parts:
        assert out.logprobs_tensors is None and out.num_nans is None
        assert out.sampling_mask_tensors is None
        sampled.index_copy_(0, sub.reqs_t, out.sampled_token_ids)
        num_sampled.index_copy_(0, sub.reqs_t, out.num_sampled)
        num_rejected.index_copy_(0, sub.reqs_t, out.num_rejected)
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
        sampling_mask_tensors=None,
    )
