# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifying the drafts of some requests of a step from the top-K candidates and of the
others from the whole logits (levels 3 and 4 of
``VLLM_MUSE_SPARK_SM120_VERIFY_TOPK_LOGITS``).

* the plan (``Sampler.topk_verify_split_plan``): which request takes which route, from
  the host copy of the request state only;
* the batch views, the index tensors and the merge of the outputs;
* the runner's routing (the order of the gathers and the verifiers);
* the whole thing on real kernels: the outputs of the split step are those of the full
  path for every request, whichever way it was verified (the rows of a request are
  verified by one route only: the projection and the draws do not depend on the
  neighbours).
"""

import functools
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip(
        "CUDA required for the split verification tests", allow_module_level=True
    )

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.topk_logits import (
    TOPP_CANDIDATE_MAX_TEMPERATURE,
    TOPP_CANDIDATE_MAX_TOP_P,
    TOPP_CANDIDATE_MIN_TOP_K,
    RouteStats,
    VerifyPlan,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.verify_split import (
    SPLIT_MIN_CANDIDATE_REQS,
    SplitIndices,
    merge_outputs,
    request_rows,
    sampler_view,
)
from vllm.v1.worker.gpu.states import RequestState

from .test_gpu_topk_logits_routing import (
    GREEDY,
    T1K64,
    K,
    MockReasoningConfig,
    sampler_with,
)
from .test_gpu_topk_verify import (
    DEVICE,
    VOCAB,
    Batch,
    as_lists,
    gather_candidates,
    soft_cap,
)

STEPS = 7


def split_plan(sampler, rows, level=3, sampled=True, k=K, min_candidates=1):
    return sampler.topk_verify_split_plan(
        np.asarray(rows, dtype=np.intp), k, sampled, level, min_candidates
    )


@pytest.fixture(autouse=True)
def one_candidate_request_is_enough(monkeypatch):
    """The routing tests use steps of a few requests: the guard that sends a split with
    fewer than SPLIT_MIN_CANDIDATE_REQS candidate requests to the full path is tested
    on its own, with the real constant."""
    monkeypatch.setattr(model_runner_module, "SPLIT_MIN_CANDIDATE_REQS", 1)


# --------------------------------------------------------------------------------
# The plan.
# --------------------------------------------------------------------------------

TOP_P = SamplingParams(temperature=1.0, top_k=64, top_p=0.9)
MIN_P = SamplingParams(temperature=1.0, top_k=64, min_p=0.1)
BOTH = SamplingParams(temperature=1.0, top_k=64, top_p=0.9, min_p=0.1)
PENALTY = SamplingParams(temperature=1.0, top_k=64, frequency_penalty=0.1)
GREEDY_PENALTY = SamplingParams(temperature=0.0, repetition_penalty=1.1)
BIG_K = SamplingParams(temperature=1.0, top_k=200)
NO_K = SamplingParams(temperature=1.0)


def test_a_step_of_plain_requests_has_no_full_row():
    sampler = sampler_with(GREEDY, T1K64, GREEDY)
    plan = split_plan(sampler, [0, 1, 2])
    assert plan == VerifyPlan(None, K, None, None, {}, True, False, False)


def test_a_greedy_only_step_needs_one_candidate_per_rank():
    sampler = sampler_with(GREEDY, GREEDY, TOP_P)
    plan = split_plan(sampler, [0, 1])
    assert (plan.reason, plan.k, plan.has_sampled, plan.full) == (None, 1, False, None)


@pytest.mark.parametrize(
    ("params", "level", "reason"),
    [
        (TOP_P, 3, "top-p"),
        (PENALTY, 3, "logits-processing"),
        (GREEDY_PENALTY, 3, "logits-processing"),
        (BIG_K, 3, "top-k-unset-or-above-k"),
        (NO_K, 3, "top-k-unset-or-above-k"),
        (BOTH, 3, "top-p-with-min-p"),
        (BOTH, 4, "top-p-with-min-p"),
        (PENALTY, 4, "logits-processing"),
    ],
)
def test_a_request_that_needs_the_logits_leaves_its_neighbours_on_the_candidates(
    params, level, reason
):
    sampler = sampler_with(T1K64, params, GREEDY, T1K64)
    plan = split_plan(sampler, [0, 1, 2, 3], level)
    assert plan.reason is None
    assert plan.full.tolist() == [1] and plan.candidate.tolist() == [0, 2, 3]
    assert plan.full_reasons == {reason: 1}
    assert plan.k == K and plan.has_sampled
    # The same request alone: no candidate request, the step takes the full path.
    assert split_plan(sampler, [1], level).reason == reason


def test_min_p_rows_are_candidates_from_level_3_and_top_p_rows_from_level_4():
    sampler = sampler_with(T1K64, MIN_P, TOP_P)
    plan3 = split_plan(sampler, [0, 1, 2], 3)
    assert plan3.full.tolist() == [2] and plan3.min_p and not plan3.top_p
    plan4 = split_plan(sampler, [0, 1, 2], 4)
    assert plan4.full is None and plan4.min_p and plan4.top_p


def test_the_flags_of_the_plan_count_the_candidate_requests_only():
    sampler = sampler_with(T1K64, TOP_P, MIN_P, PENALTY)
    plan = split_plan(sampler, [0, 3], 4)
    assert plan.full.tolist() == [1] and not plan.min_p and not plan.top_p
    plan = split_plan(sampler, [0, 1, 2], 4)
    assert plan.full is None and plan.min_p and plan.top_p


# Level 4 verifies a top_p request from the candidates only inside the class bounded by
# TOPP_CANDIDATE_MAX_TEMPERATURE, TOPP_CANDIDATE_MAX_TOP_P and TOPP_CANDIDATE_MIN_TOP_K.
IN_CLASS = {
    "at-the-bounds": SamplingParams(
        temperature=TOPP_CANDIDATE_MAX_TEMPERATURE,
        top_k=K,
        top_p=TOPP_CANDIDATE_MAX_TOP_P,
    ),
    "lowest-top_k": SamplingParams(
        temperature=0.7, top_k=TOPP_CANDIDATE_MIN_TOP_K, top_p=0.8
    ),
    "cold": SamplingParams(temperature=0.1, top_k=40, top_p=0.5),
}
OUT_OF_CLASS = {
    "hot": SamplingParams(temperature=1.5, top_k=K, top_p=0.9),
    "just-hotter": SamplingParams(
        temperature=float(np.nextafter(np.float32(TOPP_CANDIDATE_MAX_TEMPERATURE), 2)),
        top_k=K,
        top_p=0.9,
    ),
    "top_p-0.99": SamplingParams(temperature=1.0, top_k=K, top_p=0.99),
    "just-above-top_p": SamplingParams(
        temperature=1.0,
        top_k=K,
        top_p=float(np.nextafter(np.float32(TOPP_CANDIDATE_MAX_TOP_P), 1)),
    ),
    "small-top_k": SamplingParams(
        temperature=1.0, top_k=TOPP_CANDIDATE_MIN_TOP_K - 1, top_p=0.9
    ),
}


@pytest.mark.parametrize("name", sorted(OUT_OF_CLASS))
def test_level_4_keeps_a_top_p_request_outside_the_class_on_the_full_path(name):
    params = OUT_OF_CLASS[name]
    sampler = sampler_with(T1K64, params, GREEDY, T1K64)
    plan = split_plan(sampler, [0, 1, 2, 3], 4)
    assert plan.reason is None
    assert plan.full.tolist() == [1] and plan.candidate.tolist() == [0, 2, 3]
    assert plan.full_reasons == {"top-p-guard": 1} and not plan.top_p
    # Level 3 keeps every top_p request on the full path, under its own reason.
    assert split_plan(sampler, [0, 1, 2, 3], 3).full_reasons == {"top-p": 1}
    # Alone, the request takes the whole step to the full path.
    assert split_plan(sampler, [1], 4).reason == "top-p-guard"


@pytest.mark.parametrize("name", sorted(IN_CLASS))
def test_level_4_verifies_a_top_p_request_inside_the_class_from_the_candidates(name):
    sampler = sampler_with(T1K64, IN_CLASS[name], GREEDY, T1K64)
    plan = split_plan(sampler, [0, 1, 2, 3], 4)
    assert plan == VerifyPlan(None, K, None, None, {}, True, False, True)
    assert split_plan(sampler, [0, 1, 2, 3], 3).full_reasons == {"top-p": 1}


def test_level_4_splits_a_step_by_the_class_and_counts_the_candidates_only():
    sampler = sampler_with(
        IN_CLASS["at-the-bounds"],
        OUT_OF_CLASS["hot"],
        IN_CLASS["cold"],
        OUT_OF_CLASS["top_p-0.99"],
        OUT_OF_CLASS["small-top_k"],
        T1K64,
    )
    plan = split_plan(sampler, list(range(6)), 4)
    assert plan.candidate.tolist() == [0, 2, 5] and plan.full.tolist() == [1, 3, 4]
    assert plan.full_reasons == {"top-p-guard": 3} and plan.top_p
    # Only requests of the guarded kind: no candidate request, the whole step is full.
    assert split_plan(sampler, [1, 3, 4], 4).reason == "top-p-guard"
    # Without a request of the class the compiled variant has no top_p branch.
    assert not split_plan(sampler, [1, 3, 5], 4).top_p


def test_the_guard_leaves_requests_without_top_p_and_greedy_rows_alone():
    sampler = sampler_with(
        SamplingParams(temperature=0.0, top_k=K, top_p=0.99),
        SamplingParams(temperature=2.0, top_k=K),
        SamplingParams(temperature=1.0, top_k=K, top_p=1.0),
    )
    # Greedy ignores top_p; a request without top_p is a plain sampled request at any
    # temperature: both stay candidates, and no row needs the top_p branch.
    plan = split_plan(sampler, [0, 1, 2], 4)
    assert plan == VerifyPlan(None, K, None, None, {}, True, False, False)


def test_a_top_p_request_without_a_top_k_stays_on_the_full_path_at_level_4():
    sampler = sampler_with(T1K64, SamplingParams(temperature=0.7, top_p=0.9), T1K64)
    plan = split_plan(sampler, [0, 1, 2], 4)
    assert plan.full.tolist() == [1]
    assert plan.full_reasons == {"top-k-unset-or-above-k": 1}


def test_a_greedy_row_ignores_top_k_top_p_and_min_p():
    sampler = sampler_with(
        SamplingParams(temperature=0.0, top_k=500, top_p=0.5, min_p=0.3), T1K64
    )
    plan = split_plan(sampler, [0, 1], 3)
    assert (
        plan.reason is None and plan.full is None and not plan.top_p and not plan.min_p
    )


def test_sampled_rows_are_full_rows_when_the_launch_cannot_verify_them():
    sampler = sampler_with(GREEDY, T1K64, GREEDY)
    plan = split_plan(sampler, [0, 1, 2], sampled=False)
    assert plan.full.tolist() == [1] and plan.full_reasons == {"sampled-rows": 1}
    assert plan.k == 1 and not plan.has_sampled


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"return_sampling_mask": True}, "sampler-feature"),
        ({"compute_nans": True}, "sampler-feature"),
    ],
)
def test_a_sampler_feature_sends_the_whole_step_to_the_full_path(kwargs, reason):
    sampler = sampler_with(T1K64, T1K64)
    for name, value in kwargs.items():
        setattr(sampler, name, value)
    assert split_plan(sampler, [0, 1]).reason == reason


def test_logprobs_send_the_whole_step_to_the_full_path():
    sampler = sampler_with(T1K64, SamplingParams(temperature=1.0, top_k=64, logprobs=1))
    assert split_plan(sampler, [0, 1]).reason == "logprobs"
    assert split_plan(sampler, [0]).reason is None


def test_a_step_of_requests_that_all_need_the_logits_has_the_first_reason():
    sampler = sampler_with(PENALTY, TOP_P, BIG_K)
    plan = split_plan(sampler, [0, 1, 2], 3)
    assert plan.reason == "logits-processing" and plan.k == 0
    plan = split_plan(sampler, [2, 1], 3)
    assert plan.reason == "top-k-unset-or-above-k"


def test_a_split_with_few_candidate_requests_takes_the_full_path():
    # The fixed cost of a split is repaid from about SPLIT_MIN_CANDIDATE_REQS candidate
    # requests: below that the whole step takes the full path, a step without a full
    # request is not a split and keeps the candidates whatever its size.
    n = SPLIT_MIN_CANDIDATE_REQS
    sampler = sampler_with(*([T1K64] * n), PENALTY)
    plan = split_plan(sampler, range(n), 3, min_candidates=n)
    assert plan.reason is None and plan.full is None
    plan = split_plan(sampler, range(n + 1), 3, min_candidates=n)
    assert plan.reason is None and plan.full.tolist() == [n]
    assert plan.candidate.tolist() == list(range(n))
    plan = split_plan(sampler, [*range(n - 1), n], 3, min_candidates=n)
    assert plan.reason == "few-candidates" and plan.k == 0 and plan.full is None
    # A step in which every request needs the logits keeps the reason of the first.
    assert split_plan(sampler, [n], 3, min_candidates=n).reason == "logits-processing"
    # Greedy requests count as candidates.
    sampler = sampler_with(GREEDY, GREEDY, GREEDY, GREEDY, GREEDY, PENALTY)
    assert split_plan(sampler, range(6), 3, min_candidates=5).full.tolist() == [5]


def test_the_plan_reads_only_the_requests_of_the_batch():
    sampler = sampler_with(T1K64, PENALTY, GREEDY)
    assert split_plan(sampler, [0, 2]).full is None
    # Another order of the batch: the positions follow the batch, not the state slots.
    plan = split_plan(sampler, [2, 1, 0])
    assert plan.full.tolist() == [1] and plan.candidate.tolist() == [0, 2]


def test_the_split_route_stats_say_what_they_count():
    stats = RouteStats(
        K, label="steps with drafts", threshold_note=False, action="verified from"
    )
    line = stats.record(None, split={"top-p": 2, "logits-processing": 1}, num_reqs=16)
    assert line is not None
    assert "1 of 1 steps with drafts (100.0%) verified from the top-64" in line
    assert "1 of them with 3 of 16 requests on the full logits" in line
    assert "top-p: 2" in line and "logits-processing: 1" in line
    line = stats.record(None)
    assert stats.split_steps == 1 and stats.taken == 2
    line = stats.record("logprobs")
    assert line is not None and "logprobs: 1" in line


# --------------------------------------------------------------------------------
# The batch views and the merge.
# --------------------------------------------------------------------------------


def fake_input_batch(num_draft: list[int], state: list[int]):
    """A batch of requests with ``num_draft[r]`` draft tokens (rows = drafts + 1) in
    the request state slots ``state``: the fields the verifiers read."""
    rows_per = [d + 1 for d in num_draft]
    n = len(num_draft)
    cu = np.concatenate([[0], np.cumsum(rows_per)]).astype(np.int32)
    total = int(cu[-1])
    # The token stream holds two extra tokens per request (a prefill chunk) before the
    # logit rows: logits_indices points into it.
    stream = total + 2 * n
    logits_indices = np.concatenate(
        [np.arange(cu[r], cu[r + 1]) + 2 * (r + 1) for r in range(n)]
    )
    return SimpleNamespace(
        req_ids=[f"req-{r}" for r in range(n)],
        num_reqs=n,
        num_reqs_after_padding=n,
        idx_mapping=torch.tensor(state, dtype=torch.int64, device=DEVICE),
        idx_mapping_np=np.asarray(state, dtype=np.intp),
        expanded_idx_mapping=torch.tensor(
            np.repeat(state, rows_per), dtype=torch.int64, device=DEVICE
        ),
        expanded_local_pos=torch.tensor(
            np.concatenate([np.arange(c) for c in rows_per]),
            dtype=torch.int32,
            device=DEVICE,
        ),
        num_scheduled_tokens=np.asarray(rows_per, dtype=np.int32),
        num_tokens=stream,
        num_tokens_after_padding=stream,
        num_draft_tokens=int(sum(num_draft)),
        num_draft_tokens_per_req=np.asarray(num_draft, dtype=np.int32),
        seq_lens=torch.arange(100, 100 + n, dtype=torch.int32, device=DEVICE),
        seq_lens_cpu_upper_bound=torch.arange(100, 100 + n, dtype=torch.int32),
        input_ids=torch.arange(1000, 1000 + stream, dtype=torch.int64, device=DEVICE),
        positions=torch.arange(5000, 5000 + stream, dtype=torch.int64, device=DEVICE),
        logits_indices=torch.tensor(logits_indices, dtype=torch.int64, device=DEVICE),
        cu_num_logits=torch.tensor(cu, device=DEVICE),
        cu_num_logits_np=cu,
    )


def test_request_rows_lists_the_rows_of_the_requests_in_order():
    cu = np.asarray([0, 3, 4, 12, 14], dtype=np.int32)
    assert request_rows(cu, np.asarray([0, 2])).tolist() == [0, 1, 2] + list(
        range(4, 12)
    )
    assert request_rows(cu, np.asarray([3, 1])).tolist() == [12, 13, 3]
    assert request_rows(cu, np.asarray([1])).tolist() == [3]


def test_the_view_holds_the_rows_and_requests_of_the_subset_only():
    batch = fake_input_batch([7, 0, 3, 7, 2], [4, 0, 6, 2, 5])
    reqs = np.asarray([0, 2, 4])
    sub = SplitIndices(batch, reqs, torch.device(DEVICE))
    view = sampler_view(batch, sub)
    rows = request_rows(batch.cu_num_logits_np, reqs)
    token_rows = batch.logits_indices.cpu().numpy()[rows]
    assert view.num_reqs == 3 and view.req_ids == ["req-0", "req-2", "req-4"]
    assert view.idx_mapping.tolist() == [4, 6, 5]
    assert view.idx_mapping_np.tolist() == [4, 6, 5]
    assert view.cu_num_logits.tolist() == [0, 8, 12, 15]
    assert view.cu_num_logits_np.tolist() == [0, 8, 12, 15]
    assert view.expanded_idx_mapping.tolist() == [4] * 8 + [6] * 4 + [5] * 3
    assert view.expanded_local_pos.tolist() == list(range(8)) + list(range(4)) + list(
        range(3)
    )
    assert view.seq_lens.tolist() == [100, 102, 104]
    assert view.seq_lens_cpu_upper_bound.tolist() == [100, 102, 104]
    # The input token and the position of each logit row, in a stream of logit rows.
    assert view.input_ids.tolist() == (1000 + token_rows).tolist()
    assert view.positions.tolist() == (5000 + token_rows).tolist()
    assert view.logits_indices.tolist() == list(range(15))
    assert view.num_draft_tokens == 15 - 3
    assert view.num_draft_tokens_per_req.tolist() == [7, 3, 2]
    # The original batch is untouched.
    assert batch.num_reqs == 5 and batch.cu_num_logits_np.tolist() == [
        0,
        8,
        9,
        13,
        21,
        24,
    ]


def test_the_views_of_a_real_input_batch():
    """The batch the warm-up builds (a dummy ``InputBatch`` with the logit rows of every
    request, as `_dummy_rejection_sampler_run` does): the view is a real batch again and
    the merge of its subsets fills every request."""
    num_reqs, query_len = 6, STEPS + 1
    num_logits = num_reqs * query_len
    device = torch.device(DEVICE)
    batch = InputBatch.make_dummy(
        num_reqs, num_logits, InputBuffers(num_reqs, num_logits, device)
    )
    cu_np = np.arange(num_reqs + 1, dtype=np.int32) * query_len
    batch = replace(
        batch,
        expanded_idx_mapping=batch.idx_mapping.repeat_interleave(query_len),
        expanded_local_pos=torch.arange(
            query_len, dtype=torch.int32, device=device
        ).repeat(num_reqs),
        num_draft_tokens=num_reqs * (query_len - 1),
        num_draft_tokens_per_req=np.full(num_reqs, query_len - 1, dtype=np.int32),
        logits_indices=torch.arange(num_logits, dtype=torch.int64, device=device),
        cu_num_logits=torch.from_numpy(cu_np).to(device),
        cu_num_logits_np=cu_np,
    )
    cand = SplitIndices(batch, np.asarray([0, 2, 4]), device)
    full = SplitIndices(batch, np.asarray([1, 3, 5]), device)
    views = [sampler_view(batch, cand), sampler_view(batch, full)]
    assert all(type(v) is InputBatch and v.num_reqs == 3 for v in views)
    assert views[0].cu_num_logits.tolist() == [0, 8, 16, 24]
    assert views[1].idx_mapping.tolist() == [1, 3, 5]
    assert views[1].expanded_local_pos.tolist() == list(range(8)) * 3
    assert (
        batch.num_reqs == num_reqs and batch.cu_num_logits_np.tolist() == cu_np.tolist()
    )
    merged = merge_outputs(
        num_reqs, [(cand, fake_output(cand, 0)), (full, fake_output(full, 5000))]
    )
    assert merged.sampled_token_ids[:, 0].tolist() == [
        0,
        5100,
        200,
        5300,
        400,
        5500,
    ]


def fake_output(sub: SplitIndices, base: int, steps: int = STEPS) -> SamplerOutput:
    n = sub.reqs.shape[0]
    sampled = (
        base + sub.reqs_t[:, None] * 100 + torch.arange(steps + 1, device=DEVICE)
    ).to(torch.int64)
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=(sub.reqs_t % 3 + 1).to(torch.int32),
        num_rejected=(base + torch.arange(n, device=DEVICE)).to(torch.int32),
        sampling_mask_tensors=None,
    )


def test_the_merge_puts_every_request_back_in_its_place():
    batch = fake_input_batch([7, 0, 3, 7, 2], [4, 0, 6, 2, 5])
    cand = SplitIndices(batch, np.asarray([0, 2, 3]), torch.device(DEVICE))
    full = SplitIndices(batch, np.asarray([1, 4]), torch.device(DEVICE))
    merged = merge_outputs(
        5, [(cand, fake_output(cand, 0)), (full, fake_output(full, 5000))]
    )
    expected_sampled = [
        [0 + r * 100 + i for i in range(STEPS + 1)]
        if r in (0, 2, 3)
        else [5000 + r * 100 + i for i in range(STEPS + 1)]
        for r in range(5)
    ]
    assert merged.sampled_token_ids.tolist() == expected_sampled
    assert merged.num_sampled.tolist() == [r % 3 + 1 for r in range(5)]
    assert merged.num_rejected.tolist() == [0, 5000, 1, 2, 5001]
    assert merged.logprobs_tensors is None and merged.num_nans is None


# --------------------------------------------------------------------------------
# The runner's routing.
# --------------------------------------------------------------------------------


class _Recorder:
    def __init__(self, events: list, name: str, fn=None):
        self.events, self.name, self.fn = events, name, fn
        self.calls: list = []

    def __call__(self, *args, **kwargs):
        self.events.append(self.name)
        self.calls.append((args, kwargs))
        return self.fn(*args, **kwargs) if self.fn is not None else None


def split_runner(events: list, sampler: Sampler, level: int = 3, **kwargs):
    def cand_out(topk, view, draft_logits, has_sampled, **kw):
        return fake_output(
            SplitIndices(view, np.arange(view.num_reqs), torch.device(DEVICE)), 0
        )

    def full_out(logits, view, draft_logits, invalid):
        return fake_output(
            SplitIndices(view, np.arange(view.num_reqs), torch.device(DEVICE)), 5000
        )

    rejection = _Recorder(events, "rejection_sampler", full_out)
    rejection.verify_topk = _Recorder(events, "verify_topk", cand_out)
    model = SimpleNamespace(
        compute_topk_logits=_Recorder(
            events, "compute_topk_logits", lambda h, k=None: "topk"
        ),
        compute_split_verify_logits=_Recorder(
            events, "compute_split_verify_logits", lambda h, k, e, f: ("topk", "logits")
        ),
    )
    runner = SimpleNamespace(
        model=model,
        sampler=sampler,
        rejection_sampler=rejection,
        speculator=SimpleNamespace(draft_logits="draft_logits"),
        topk_logits_k=K,
        topk_verify_level=level,
        topk_verify_sampled=True,
        topk_verify_stats=RouteStats(
            K, label="steps with drafts", threshold_note=False, action="verified from"
        ),
        _topk_verify_logged=True,
        adaptive_verification=None,
        pcp_manager=None,
        **kwargs,
    )
    runner._verify_split = functools.partial(GPUModelRunner._verify_split, runner)
    return runner


def run_verify(runner, batch, grammar=None):
    hidden = torch.zeros(batch.logits_indices.shape[0], 4, device=DEVICE)
    return GPUModelRunner._verify_from_topk_logits(runner, hidden, batch, grammar)


def test_a_step_with_requests_on_the_logits_gathers_both_and_merges():
    events: list = []
    sampler = sampler_with(T1K64, TOP_P, GREEDY, PENALTY, T1K64)
    runner = split_runner(events, sampler, level=3)
    batch = fake_input_batch([7, 7, 3, 7, 1], [0, 1, 2, 3, 4])
    out = run_verify(runner, batch)
    assert events == ["compute_split_verify_logits", "verify_topk", "rejection_sampler"]
    ((args, _),) = runner.model.compute_split_verify_logits.calls
    assert args[1] == K
    assert (
        args[2].tolist()
        == request_rows(batch.cu_num_logits_np, np.asarray([0, 2, 4])).tolist()
    )
    assert (
        args[3].tolist()
        == request_rows(batch.cu_num_logits_np, np.asarray([1, 3])).tolist()
    )
    ((vargs, vkw),) = runner.rejection_sampler.verify_topk.calls
    # The candidates, the view of the candidate requests, the draft logits, the sampled
    # flag; no min_p or top_p request among them at level 3.
    assert vargs[0] == "topk" and vargs[1].num_reqs == 3 and vargs[2] == "draft_logits"
    assert vargs[3] is True and vkw == {"use_min_p": False, "use_top_p": False}
    ((fargs, _),) = runner.rejection_sampler.calls
    assert fargs[0] == "logits" and fargs[1].num_reqs == 2 and fargs[3] is None
    # The outputs of the two routes sit in the batch order.
    assert [row[0] // 5000 for row in out.sampled_token_ids.tolist()] == [0, 1, 0, 1, 0]
    assert runner.topk_verify_stats.split_steps == 1
    assert runner.topk_verify_stats.split_reasons == {
        "top-p": 1,
        "logits-processing": 1,
    }


def test_the_runner_keeps_a_split_below_the_candidate_minimum_on_the_full_path(
    monkeypatch,
):
    monkeypatch.setattr(
        model_runner_module, "SPLIT_MIN_CANDIDATE_REQS", SPLIT_MIN_CANDIDATE_REQS
    )
    n = SPLIT_MIN_CANDIDATE_REQS
    for cands, split in ((n - 1, False), (n, True)):
        events: list = []
        sampler = sampler_with(*([T1K64] * cands), PENALTY)
        runner = split_runner(events, sampler, level=3)
        batch = fake_input_batch([7] * (cands + 1), list(range(cands + 1)))
        out = run_verify(runner, batch)
        stats = runner.topk_verify_stats
        if split:
            assert out is not None and stats.split_steps == 1
            assert events == [
                "compute_split_verify_logits",
                "verify_topk",
                "rejection_sampler",
            ]
        else:
            assert out is None and events == []
            assert stats.fallbacks == {"few-candidates": 1} and stats.taken == 0


def test_a_step_without_a_full_row_gathers_the_candidates_only():
    events: list = []
    sampler = sampler_with(T1K64, MIN_P, GREEDY)
    runner = split_runner(events, sampler, level=3)
    batch = fake_input_batch([7, 7, 7], [0, 1, 2])
    run_verify(runner, batch)
    assert events == ["compute_topk_logits", "verify_topk"]
    ((_, vkw),) = runner.rejection_sampler.verify_topk.calls
    assert vkw == {"use_min_p": True, "use_top_p": False}
    assert (
        runner.topk_verify_stats.split_steps == 0
        and runner.topk_verify_stats.taken == 1
    )


def test_top_p_requests_take_the_candidates_at_level_4():
    events: list = []
    sampler = sampler_with(T1K64, TOP_P, PENALTY)
    runner = split_runner(events, sampler, level=4)
    batch = fake_input_batch([7, 7, 7], [0, 1, 2])
    run_verify(runner, batch)
    ((_, vkw),) = runner.rejection_sampler.verify_topk.calls
    assert vkw == {"use_min_p": False, "use_top_p": True}
    ((fargs, _),) = runner.rejection_sampler.calls
    assert fargs[1].num_reqs == 1


def test_a_top_p_request_outside_the_class_is_gathered_whole_at_level_4():
    events: list = []
    sampler = sampler_with(T1K64, OUT_OF_CLASS["hot"], PENALTY)
    runner = split_runner(events, sampler, level=4)
    batch = fake_input_batch([7, 7, 7], [0, 1, 2])
    run_verify(runner, batch)
    ((_, vkw),) = runner.rejection_sampler.verify_topk.calls
    assert vkw == {"use_min_p": False, "use_top_p": False}
    ((fargs, _),) = runner.rejection_sampler.calls
    assert fargs[1].num_reqs == 2
    assert runner.topk_verify_stats.split_reasons == {
        "top-p-guard": 1,
        "logits-processing": 1,
    }


@pytest.mark.parametrize("case", ["no-candidate-request", "logprobs", "grammar"])
def test_a_step_with_no_candidate_request_takes_the_full_path(case):
    events: list = []
    if case == "no-candidate-request":
        sampler = sampler_with(PENALTY, TOP_P)
    elif case == "logprobs":
        sampler = sampler_with(
            T1K64, SamplingParams(temperature=1.0, top_k=64, logprobs=1)
        )
    else:
        sampler = sampler_with(T1K64, T1K64)
    runner = split_runner(events, sampler, level=3)
    batch = fake_input_batch([7, 7], [0, 1])
    assert (
        run_verify(runner, batch, grammar=object() if case == "grammar" else None)
        is None
    )
    assert events == []
    assert sum(runner.topk_verify_stats.fallbacks.values()) == 1


# --------------------------------------------------------------------------------
# The whole thing on real kernels: the split step equals the full path.
# --------------------------------------------------------------------------------


def real_sampler(max_reqs: int) -> Sampler:
    req_states = RequestState(
        max_num_reqs=max_reqs,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=STEPS,
        vocab_size=VOCAB,
        device=torch.device(DEVICE),
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=max_reqs,
        vocab_size=VOCAB,
        device=torch.device(DEVICE),
        req_states=req_states,
        num_speculative_tokens=STEPS,
    )


def real_rejection_sampler(sampler: Sampler) -> RejectionSampler:
    rs = object.__new__(RejectionSampler)
    rs.sampler = sampler
    rs.watermark_key = None
    rs.synthetic_conditional_rates = None
    rs.enable_adaptive_verification = False
    rs.use_block_verification = True
    rs.num_speculative_steps = STEPS
    return rs


# Tokens that every row has in its head: a penalty or a logit bias on them changes the
# distribution of every row (so that the requests that need them are not a no-op).
SHARED_IDS = [17 + 6000 * i for i in range(32)]


def head_logits(rows: int, seed: int) -> torch.Tensor:
    """Large logits (a head over a noisy floor) so that Qrita's round trip is exact."""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    x = torch.randn(rows, VOCAB, generator=g, device=DEVICE) * 3.0 - 3.0
    ids = torch.randint(0, VOCAB, (rows, 160), generator=g, device=DEVICE)
    ids[:, :32] = torch.tensor(SHARED_IDS, device=DEVICE)
    x.scatter_(1, ids, 60.0 + 3.0 * torch.randn(rows, 160, generator=g, device=DEVICE))
    return x.to(torch.bfloat16)


@pytest.mark.parametrize("level", [3, 4])
def test_the_split_step_equals_the_full_path_for_every_request(level: int):
    """Requests of every kind in one step: greedy, plain, min_p, top_p, and the ones
    that need the logits (penalty, top_k above K, top_p at level 3). The accepted tokens
    and the resampled token of each equal what the full path gives for the same batch
    (every request is verified by one route only, on its own rows)."""
    kinds = np.random.default_rng(level).integers(0, 11, 44)
    kinds[:11] = np.arange(11)
    plan = check_split_equals_full(level, kinds, seed=level)
    # Every kind of request and both routes took part; a top_p request without a top_k
    # or outside the class of level 4 is on the full route, one inside the class from
    # the candidates at level 4.
    assert set(kinds.tolist()) == set(range(11))
    assert all(r in plan.full for r in range(len(kinds)) if kinds[r] in (8, 9))
    in_class = [r for r in range(len(kinds)) if kinds[r] in (3, 10)]
    assert all((r in plan.candidate) == (level == 4) for r in in_class)
    assert len(plan.candidate) >= 10 and len(plan.full) >= 10
    assert plan.min_p and (plan.top_p == (level == 4))


def test_a_top_p_request_without_a_top_k_gets_the_tokens_of_the_whole_step():
    """The nucleus kernel takes another pipeline for 64 rows or fewer without an active
    top_k than for a bigger batch: the full subset of 2 such requests (16 rows) next to
    12 candidate requests (112 rows in all) is such a case. On rows whose nucleus
    boundary is no tie the tokens equal those of the whole step (the law is the same;
    at a tie of the boundary the full path alone differs between small and big
    batches)."""
    kinds = np.array([1] * 6 + [0] * 6 + [8] * 2)
    plan = check_split_equals_full(3, kinds, seed=77, num_draft=[7] * len(kinds))
    assert plan.full.tolist() == [12, 13] and len(plan.candidate) == 12


def check_split_equals_full(level, kinds, seed, num_draft=None):
    n = len(kinds)
    rng = np.random.default_rng(seed)
    # 0 greedy, 1 plain, 2 min_p, 3 top_p, 4 frequency penalty, 5 top_k above K,
    # 6 greedy with a logit bias, 7 sampled with a logit bias, 8 top_p without a top_k,
    # 9 top_p at temperature 1.5 (outside the class of level 4), 10 top_p at temperature
    # 0.7 with top_k 40 (inside it)
    params = []
    for r, kind in enumerate(kinds):
        seed = int(rng.integers(1, 1 << 30))
        if kind == 0:
            p = SamplingParams(temperature=0.0)
        elif kind == 1:
            p = SamplingParams(
                temperature=float(rng.choice([0.7, 1.0])),
                top_k=int(rng.choice([17, 64])),
                seed=seed,
            )
        elif kind == 2:
            p = SamplingParams(
                temperature=1.0,
                top_k=64,
                min_p=float(rng.choice([0.05, 0.3])),
                seed=seed,
            )
        elif kind == 3:
            p = SamplingParams(
                temperature=1.0,
                top_k=64,
                top_p=float(rng.choice([0.8, 0.95])),
                seed=seed,
            )
        elif kind == 4:
            p = SamplingParams(
                temperature=1.0, top_k=64, frequency_penalty=0.2, seed=seed
            )
        elif kind == 5:
            p = SamplingParams(temperature=1.0, top_k=200, seed=seed)
        elif kind == 8:
            p = SamplingParams(temperature=1.0, top_p=0.9, seed=seed)
        elif kind == 9:
            p = SamplingParams(temperature=1.5, top_k=64, top_p=0.95, seed=seed)
        elif kind == 10:
            p = SamplingParams(temperature=0.7, top_k=40, top_p=0.9, seed=seed)
        else:
            bias = {
                SHARED_IDS[int(i)]: float(rng.choice([-4.0, 3.0]))
                for i in rng.choice(32, 6, replace=False)
            }
            if kind == 6:
                p = SamplingParams(temperature=0.0, logit_bias=bias)
            else:
                p = SamplingParams(
                    temperature=1.0, top_k=64, logit_bias=bias, seed=seed
                )
        params.append(p)
    sampler = real_sampler(n + 3)
    state = rng.permutation(n + 3)[:n].astype(int).tolist()
    for slot, p in zip(state, params):
        sampler.add_request(slot, p)
    sampler.apply_staged_writes()
    # The output counts of the penalty requests: tokens of every row's head, seen 1-5
    # times.
    for slot, kind in zip(state, kinds):
        if kind == 4:
            counts = torch.tensor(
                rng.integers(1, 6, 32), dtype=torch.int32, device=DEVICE
            )
            sampler.penalties_state.output_bin_counts[slot, SHARED_IDS] = counts
    if num_draft is None:
        num_draft = rng.choice([0, 7, 7, 7, 3, 1], n).tolist()
    rows = sum(d + 1 for d in num_draft)
    raw = head_logits(rows, seed=level)
    # The rows of the top_p requests without a top_k: a ladder of 24 distinct logits
    # (steps of 1.7: the nucleus is the first two tokens, the boundary is no tie), so
    # that the two pipelines of the nucleus kernel (small subset against whole step)
    # must agree.
    start = 0
    for r in range(n):
        if kinds[r] == 8:
            g = torch.Generator(device=DEVICE).manual_seed(900 + r)
            for row in range(start, start + num_draft[r] + 1):
                ids = torch.randperm(VOCAB, generator=g, device=DEVICE)[:24]
                raw[row, ids] = (
                    60.0 - 1.7 * torch.arange(24, device=DEVICE, dtype=torch.float32)
                ).to(torch.bfloat16)
        start += num_draft[r] + 1
    # The drafts and the draft distribution of the kernel tests' batch; the request
    # state of the sampler holds the sampling parameters and seeds.
    harness = Batch(
        num_draft,
        [0.0] * n,
        [VOCAB] * n,
        STEPS,
        seed=level + 20,
        logits=raw,
        peaky=False,
        accept_prefix=rng.integers(0, 8, n).tolist(),
    )
    batch = fake_input_batch(num_draft, state)
    batch.input_ids = harness.draft_sampled.clone()
    batch.positions = harness.pos.clone()
    batch.logits_indices = torch.arange(rows, dtype=torch.int64, device=DEVICE)
    batch.num_tokens = batch.num_tokens_after_padding = rows
    # The harness drafts live in its own state slots: move them to the sampler's.
    draft_logits = torch.full((n + 3, STEPS, VOCAB), float("-inf"), device=DEVICE)
    draft_logits[torch.tensor(state, device=DEVICE)] = harness.draft_logits[
        harness.idx_mapping.long()
    ]

    rs = real_rejection_sampler(sampler)
    full_logits = soft_cap(raw).contiguous()
    expected = rs(full_logits.clone(), batch, draft_logits, None)

    plan = sampler.topk_verify_split_plan(batch.idx_mapping_np, K, True, level)
    assert plan.reason is None and plan.full is not None
    cand = SplitIndices(batch, plan.candidate, torch.device(DEVICE))
    full = SplitIndices(batch, plan.full, torch.device(DEVICE))
    topk = gather_candidates(raw[cand.rows_t])
    cand_out = rs.verify_topk(
        topk,
        sampler_view(batch, cand),
        draft_logits,
        plan.has_sampled,
        use_min_p=plan.min_p,
        use_top_p=plan.top_p,
    )
    full_out = rs(
        full_logits[full.rows_t].clone(), sampler_view(batch, full), draft_logits, None
    )
    merged = merge_outputs(n, [(cand, cand_out), (full, full_out)])
    got = as_lists(merged.sampled_token_ids, merged.num_sampled)
    want = as_lists(expected.sampled_token_ids, expected.num_sampled)
    bad = [r for r in range(n) if got[r] != want[r]]
    assert not bad, [(r, int(kinds[r]), want[r], got[r]) for r in bad]
    assert merged.num_rejected.tolist() == expected.num_rejected.tolist()
    return plan
