# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When a V2 step samples from the top-K logits of each TP rank, and when it does not.

Every trigger that needs more than the top K of the vocabulary, or the logits
themselves, has to send the step down the full path, and every TP rank has to decide
alike: the decision reads host copies of the request state, nothing rank-specific.
"""

import functools
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for top-k logits routing tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.logits_processor import (
    LogitsContext,
    LogitsProcessor,
)
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.sample.topk_logits import RouteStats
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128
K = 64


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def make_sampler(
    custom_logits_processors: Sequence[LogitsProcessor] = (), **kwargs
) -> Sampler:
    req_states = RequestState(
        max_num_reqs=8,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=8,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        custom_logits_processors=custom_logits_processors,
        **kwargs,
    )


def supported(sampler: Sampler, rows: Sequence[int], k: int = K) -> bool:
    return sampler.topk_logits_supported(np.asarray(rows, dtype=np.intp), k)


def sampler_with(*params: SamplingParams, **kwargs) -> Sampler:
    sampler = make_sampler(**kwargs)
    for i, p in enumerate(params):
        sampler.add_request(i, p)
    return sampler


GREEDY = SamplingParams(temperature=0.0)
T1K64 = SamplingParams(temperature=1.0, top_k=64)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param(GREEDY, True, id="greedy"),
        pytest.param(T1K64, True, id="t1k64"),
        pytest.param(SamplingParams(temperature=0.7, top_k=5), True, id="t0.7-k5"),
        pytest.param(SamplingParams(temperature=1.0, top_k=1), True, id="k1"),
        # Greedy ignores top_k, top_p and min_p.
        pytest.param(
            SamplingParams(temperature=0.0, top_p=0.5, min_p=0.3, top_k=500),
            True,
            id="greedy-with-filters",
        ),
        pytest.param(SamplingParams(temperature=1.0, top_k=65), False, id="top_k>K"),
        pytest.param(SamplingParams(temperature=1.0), False, id="no-top_k"),
        pytest.param(SamplingParams(temperature=0.7), False, id="temperature-only"),
        pytest.param(SamplingParams(temperature=1.0, top_k=-1), False, id="top_k=-1"),
        pytest.param(
            SamplingParams(temperature=1.0, top_k=64, top_p=0.9), False, id="top_p"
        ),
        pytest.param(
            SamplingParams(temperature=1.0, top_k=64, min_p=0.1), False, id="min_p"
        ),
        pytest.param(
            SamplingParams(temperature=1.0, top_p=0.9), False, id="top_p-no-top_k"
        ),
        pytest.param(SamplingParams(temperature=0.0, logprobs=1), False, id="logprobs"),
        pytest.param(
            SamplingParams(temperature=0.0, logprobs=0), False, id="logprobs-0"
        ),
        pytest.param(
            SamplingParams(temperature=0.0, logprob_token_ids=[1, 2]),
            False,
            id="logprob_token_ids",
        ),
        pytest.param(
            SamplingParams(temperature=0.0, frequency_penalty=0.1),
            False,
            id="frequency-penalty",
        ),
        pytest.param(
            SamplingParams(temperature=0.0, presence_penalty=0.1),
            False,
            id="presence-penalty",
        ),
        pytest.param(
            SamplingParams(temperature=0.0, repetition_penalty=1.1),
            False,
            id="repetition-penalty",
        ),
        pytest.param(
            SamplingParams(temperature=0.0, logit_bias={1: 1.0}), False, id="logit-bias"
        ),
        pytest.param(
            SamplingParams(temperature=0.0, allowed_token_ids=[1, 2]),
            False,
            id="allowed-token-ids",
        ),
        pytest.param(
            SamplingParams(temperature=0.0, _bad_words_token_ids=[[1]]),
            False,
            id="bad-words",
        ),
        pytest.param(
            SamplingParams(temperature=0.0, thinking_token_budget=3),
            False,
            id="thinking-budget",
        ),
        pytest.param(
            SamplingParams(temperature=0.0, min_tokens=2, stop_token_ids=[3]),
            False,
            id="min-tokens",
        ),
    ],
)
def test_single_request_predicate(params: SamplingParams, expected: bool):
    assert supported(sampler_with(params), [0]) == expected


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        (GREEDY, None),
        (T1K64, None),
        (SamplingParams(temperature=1.0, top_k=65), "top-k-unset-or-above-k"),
        (SamplingParams(temperature=1.0), "top-k-unset-or-above-k"),
        (SamplingParams(temperature=1.0, top_k=64, top_p=0.9), "top-p-or-min-p"),
        (SamplingParams(temperature=1.0, top_p=0.95), "top-p-or-min-p"),
        (SamplingParams(temperature=1.0, top_k=64, min_p=0.1), "top-p-or-min-p"),
        (SamplingParams(temperature=0.0, logprobs=1), "logprobs"),
        (SamplingParams(temperature=0.0, repetition_penalty=1.1), "logits-processing"),
        (SamplingParams(temperature=0.0, logit_bias={1: 1.0}), "logits-processing"),
        (SamplingParams(temperature=0.0, thinking_token_budget=3), "logits-processing"),
    ],
    ids=lambda v: v if isinstance(v, str) else None,
)
def test_the_fallback_reason(params: SamplingParams, reason: str | None):
    sampler = sampler_with(params)
    assert sampler.topk_logits_fallback_reason(np.zeros(1, dtype=np.intp), K) == reason
    assert supported(sampler, [0]) == (reason is None)


def test_the_fallback_reason_of_a_sampler_feature():
    sampler = sampler_with(T1K64, return_sampling_mask=True)
    assert (
        sampler.topk_logits_fallback_reason(np.zeros(1, dtype=np.intp), K)
        == "sampler-feature"
    )


@pytest.mark.parametrize("use_flashinfer", [True, False])
@pytest.mark.parametrize(
    "params",
    [
        [T1K64] * 4,
        [T1K64, SamplingParams(temperature=0.7, top_k=20), T1K64],
        [GREEDY, T1K64, T1K64],
        [T1K64, SamplingParams(temperature=1.0, top_k=64, seed=5), T1K64],
        [SamplingParams(temperature=1.0, top_k=64, seed=5)] * 3,
        [GREEDY] * 3,
        [SamplingParams(temperature=1.0, top_k=1)] * 2,
    ],
    ids=["unseeded", "unseeded-mixed", "greedy-row", "seeded-row", "all-seeded",
         "all-greedy", "k1"],
)
def test_the_threshold_route_is_the_choice_of_the_full_sampler(
    monkeypatch, params, use_flashinfer
):
    """The full sampler draws a step with FlashInfer's top-k sampler (every token tied
    with the top_k-th logit stays) or by Gumbel-max (exactly top_k tokens). The
    candidates take the matching rule: the same decision, from the same state."""
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers

    seen = []

    def spy(self, processed_logits, expanded_idx_mapping, idx_mapping_np, pos, top_k,
            top_p, use_fused_sampler):
        seen.append(use_fused_sampler)
        return processed_logits.argmax(-1), processed_logits

    monkeypatch.setattr(Sampler, "_sample_random", spy)
    sampler = sampler_with(*params)
    sampler.apply_staged_writes()
    sampler.use_flashinfer = use_flashinfer
    n = len(params)
    assert supported(sampler, list(range(n)))
    buffers = InputBuffers(n, n, DEVICE)
    batch = InputBatch.make_dummy(n, n, buffers)
    sampler(torch.randn(n, VOCAB_SIZE, device=DEVICE), batch)
    assert len(seen) == 1
    assert sampler.topk_logits_threshold_route(batch.idx_mapping_np) == seen[0]
    # ... and the candidates of the step carry the closure of the ties exactly then.
    tie_draw = sampler.topk_logits_tie_draw(batch)
    assert (tie_draw is not None) == seen[0]


def test_a_batch_takes_the_path_only_if_every_request_can():
    sampler = sampler_with(
        GREEDY,
        T1K64,
        SamplingParams(temperature=1.0, top_k=64, frequency_penalty=0.2),
        SamplingParams(temperature=1.0, top_k=100),
        SamplingParams(temperature=0.0, logprobs=2),
    )
    assert supported(sampler, [0, 1])
    assert not supported(sampler, [0, 1, 2])
    assert not supported(sampler, [1, 3])
    assert not supported(sampler, [0, 4])
    assert supported(sampler, [1, 0])


def test_the_limit_is_the_k_of_the_path():
    sampler = sampler_with(SamplingParams(temperature=1.0, top_k=16))
    assert supported(sampler, [0], k=16)
    assert not supported(sampler, [0], k=15)


def test_a_reused_slot_forgets_the_old_request():
    sampler = sampler_with(SamplingParams(temperature=0.0, frequency_penalty=0.5))
    assert not supported(sampler, [0])
    sampler.add_request(0, T1K64)
    assert supported(sampler, [0])


class GateProcessor(LogitsProcessor):
    def __init__(self, admitted: bool):
        self.admitted = admitted

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        return self.admitted

    def apply(self, logits: torch.Tensor, ctx: LogitsContext) -> torch.Tensor:
        raise AssertionError("never runs here")


@pytest.mark.parametrize("admitted", [True, False])
def test_a_custom_processor_that_takes_the_request_needs_the_logits(admitted: bool):
    sampler = make_sampler(custom_logits_processors=[GateProcessor(admitted)])
    sampler.add_request(0, T1K64)
    assert supported(sampler, [0]) == (not admitted)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"return_sampling_mask": True},
        {"enable_trace_replay": True},
    ],
    ids=["sampling-mask", "trace-replay"],
)
def test_sampler_features_that_read_the_logits_turn_the_path_off(kwargs):
    sampler = sampler_with(T1K64, **kwargs)
    assert not supported(sampler, [0])


def test_the_nan_counter_turns_the_path_off(monkeypatch):
    sampler = sampler_with(T1K64)
    assert supported(sampler, [0])
    sampler.compute_nans = True
    assert not supported(sampler, [0])


def test_ranks_decide_alike(monkeypatch):
    """The decision does not read what differs between TP ranks (the random seeds
    of requests without one, the order of dict-valued parameters, the device)."""
    rng = np.random.default_rng(0)
    pool = [
        GREEDY,
        T1K64,
        SamplingParams(temperature=0.8, top_k=20),
        SamplingParams(temperature=1.0),
        SamplingParams(temperature=1.0, top_k=64, top_p=0.95),
        SamplingParams(temperature=0.0, logprobs=1),
        SamplingParams(temperature=0.0, logit_bias={3: 1.0, 1: 2.0}),
        SamplingParams(temperature=1.0, top_k=64, min_p=0.05),
    ]
    for trial in range(20):
        chosen = [pool[i] for i in rng.integers(0, len(pool), 8)]
        decisions = []
        for rank_seed in (11, 22):
            np.random.seed(rank_seed)  # what differs: the seeds of unseeded requests
            sampler = sampler_with(*chosen)
            decisions.append(
                [supported(sampler, list(range(n))) for n in range(1, 9)]
            )
        assert decisions[0] == decisions[1], trial


# --------------------------------------------------------------------------------
# The sampler: the output of the candidate path is that of ``Sampler.__call__``.
# --------------------------------------------------------------------------------


def test_sample_topk_logits_matches_the_sampler_call():
    from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
    from vllm.v1.worker.gpu.sample.topk_logits import (
        TopKLogits,
        local_topk_candidates,
    )

    tp, k, vocab, num_reqs = 4, 16, 8192, 8
    req_states = RequestState(
        max_num_reqs=num_reqs,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=vocab,
        device=DEVICE,
    )
    sampler = Sampler(
        vllm_config=SimpleNamespace(reasoning_config=MockReasoningConfig()),
        max_num_reqs=num_reqs,
        vocab_size=vocab,
        device=DEVICE,
        req_states=req_states,
    )
    params = [
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=1.0, top_k=16, seed=11),
        SamplingParams(temperature=0.7, top_k=5, seed=12),
        SamplingParams(temperature=0.0, top_k=3),
        SamplingParams(temperature=1.3, top_k=1, seed=13),
        SamplingParams(temperature=1.0, top_k=9, seed=14),
        SamplingParams(temperature=0.0),
        SamplingParams(temperature=0.5, top_k=16, seed=15),
    ]
    for i, p in enumerate(params):
        sampler.add_request(i, p)
    sampler.apply_staged_writes()
    assert supported(sampler, list(range(num_reqs)), k=k)

    buffers = InputBuffers(num_reqs, num_reqs, DEVICE)
    batch = InputBatch.make_dummy(num_reqs, num_reqs, buffers)
    buffers.positions[:num_reqs] = torch.randint(1, 1000, (num_reqs,), device=DEVICE)
    logits = (torch.randn(num_reqs, vocab, device=DEVICE) * 3).to(torch.bfloat16)

    expected = sampler(logits.float(), batch)
    shard = vocab // tp
    words = torch.cat(
        [
            local_topk_candidates(logits[:, r * shard : (r + 1) * shard], k)
            for r in range(tp)
        ]
    ).view(torch.bfloat16)
    topk = TopKLogits(words, words.float(), num_reqs, k, tp, shard)
    got = sampler.sample_topk_logits(topk, batch)

    assert torch.equal(got.sampled_token_ids, expected.sampled_token_ids)
    assert torch.equal(got.num_sampled, expected.num_sampled)
    assert torch.equal(got.num_rejected, expected.num_rejected)
    assert got.logprobs_tensors is None and got.sampling_mask_tensors is None


# --------------------------------------------------------------------------------
# The runner: the step-level predicate and the launch-level switch.
# --------------------------------------------------------------------------------


def fake_batch(rows: int, draft_tokens: int = 0, reqs: int | None = None):
    return SimpleNamespace(
        logits_indices=torch.zeros(rows, dtype=torch.int64),
        num_draft_tokens=draft_tokens,
        num_reqs=rows if reqs is None else reqs,
        idx_mapping_np=np.arange(rows, dtype=np.intp),
    )


def fake_runner(
    k: int = K,
    min_tokens: int = 1,
    reason: str | None = None,
    threshold: bool = False,
    **kwargs,
):
    sampler = SimpleNamespace(
        calls=[],
        topk_logits_fallback_reason=lambda idx, kk: (
            sampler.calls.append((idx.tolist(), kk)) or reason
        ),
        topk_logits_threshold_route=lambda idx: threshold,
    )
    runner = SimpleNamespace(
        topk_logits_k=k,
        topk_logits_min_tokens=min_tokens,
        topk_logits_stats=RouteStats(k),
        pcp_manager=None,
        sampler=sampler,
        **kwargs,
    )
    runner._topk_logits_fallback_reason = functools.partial(
        GPUModelRunner._topk_logits_fallback_reason, runner
    )
    return runner


def step_uses_topk(runner, batch, grammar_output=None) -> bool:
    return GPUModelRunner._sample_from_topk_logits(runner, batch, grammar_output)


def test_the_step_predicate_asks_the_sampler_about_the_batch():
    runner = fake_runner()
    assert step_uses_topk(runner, fake_batch(3))
    assert runner.sampler.calls == [([0, 1, 2], K)]


@pytest.mark.parametrize(
    "case",
    [
        "switch-off",
        "grammar",
        "draft-tokens",
        "more-logits-than-requests",
        "below-min-tokens",
        "prefill-context-parallel",
        "sampler-refuses",
    ],
)
def test_the_step_predicate_falls_back(case: str):
    runner, batch, grammar = fake_runner(), fake_batch(4), None
    if case == "switch-off":
        runner.topk_logits_k = 0
    elif case == "grammar":
        grammar = object()
    elif case == "draft-tokens":
        batch = fake_batch(4, draft_tokens=2)
    elif case == "more-logits-than-requests":
        batch = fake_batch(4, reqs=2)
    elif case == "below-min-tokens":
        runner.topk_logits_min_tokens = 5
    elif case == "prefill-context-parallel":
        runner.pcp_manager = object()
    elif case == "sampler-refuses":
        runner.sampler.topk_logits_fallback_reason = lambda idx, k: "logprobs"
    assert not step_uses_topk(runner, batch, grammar)


def test_a_step_without_drafts_takes_the_path_under_speculation():
    # Dynamic speculative decoding at K == 0: the batch has no draft token, one
    # logits row per request.
    runner = fake_runner(speculative_config=object())
    assert step_uses_topk(runner, fake_batch(4, draft_tokens=0))
    assert runner.topk_logits_stats.taken == 1


@pytest.mark.parametrize(
    ("rows", "reqs", "drafts"),
    [(4 + 4 * 7, 4, 4 * 7), (4 + 1, 4, 1), (3 + 2, 3, 2)],
    ids=["full-block", "one-request-drafts", "mixed-widths"],
)
def test_a_step_with_any_draft_token_takes_the_full_path(rows, reqs, drafts):
    runner = fake_runner(speculative_config=object())
    assert not step_uses_topk(runner, fake_batch(rows, draft_tokens=drafts, reqs=reqs))
    assert runner.topk_logits_stats.fallbacks == {"draft-tokens": 1}
    # The sampler is not even asked: the batch shape decides alone.
    assert runner.sampler.calls == []


def test_the_step_predicate_reads_no_device_data():
    # Decided from the batch shape and host state: no device read, no sync, and the
    # same answer on every TP rank (a rank-specific tensor value cannot enter).
    runner = fake_runner(speculative_config=object())
    batch = fake_batch(4)
    batch.logits_indices = torch.empty(4, dtype=torch.int64, device="meta")
    assert step_uses_topk(runner, batch)
    batch.logits_indices = torch.empty(9, dtype=torch.int64, device="meta")
    batch.num_draft_tokens = 5
    assert not step_uses_topk(runner, batch)


def test_the_step_predicate_depends_on_the_row_count_alone():
    runner = fake_runner(min_tokens=16)
    # Every TP rank has the same rows: the same side of the threshold.
    assert [step_uses_topk(runner, fake_batch(n)) for n in (1, 15, 16, 17, 256)] == [
        False,
        False,
        True,
        True,
        True,
    ]


def test_the_runner_counts_the_steps_by_reason():
    runner = fake_runner(min_tokens=2)
    for rows in (1, 8, 8, 64):
        step_uses_topk(runner, fake_batch(rows))
    step_uses_topk(runner, fake_batch(8), grammar_output=object())
    runner.sampler.topk_logits_fallback_reason = lambda idx, k: "logprobs"
    step_uses_topk(runner, fake_batch(8))
    stats = runner.topk_logits_stats
    assert stats.taken == 3 and stats.steps == 6
    assert stats.fallbacks == {"few-rows": 1, "grammar": 1, "logprobs": 1}


def test_a_launch_without_the_path_counts_nothing():
    runner = fake_runner(k=0)
    assert not step_uses_topk(runner, fake_batch(8))
    assert runner.topk_logits_stats.steps == 0 and runner.sampler.calls == []


def test_route_stats_count_the_threshold_steps():
    stats = RouteStats(64, clock=lambda: 0.0)
    stats.record(None, threshold=True)
    stats.record(None)
    stats.record(None, threshold=True)
    line = stats.record("logprobs")
    assert stats.taken == 3 and stats.taken_threshold == 2
    assert "3 of 4 sampled steps" in line and "2 of them on FlashInfer's rule" in line


def test_route_stats_log_the_first_step_new_reasons_and_then_every_interval():
    now = [100.0]
    stats = RouteStats(64, interval_s=30.0, clock=lambda: now[0])
    first = stats.record(None)
    assert first is not None and "1 of 1 sampled steps (100.0%)" in first
    assert "top-64" in first and "gathered" not in first
    assert stats.record(None) is None
    # A reason seen for the first time is logged at once, a repeat is not.
    line = stats.record("logprobs")
    assert line is not None and "logprobs: 1" in line and "2 of 3" in line
    assert stats.record("logprobs") is None
    now[0] += 29.0
    assert stats.record(None) is None
    now[0] += 1.0
    line = stats.record("few-rows")  # new reason
    assert line is not None
    now[0] += 31.0
    line = stats.record(None)
    assert line is not None and "few-rows: 1, logprobs: 2" in line
    assert "4 of 7 sampled steps" in line


def resolve(model_k: int = K, **overrides):
    fields = dict(
        model=SimpleNamespace(topk_logits_k=model_k),
        speculative_config=None,
        lora_config=None,
        batch_sharder=None,
        sampler=make_sampler(),
        topk_logits_min_tokens=1,
    )
    runner = SimpleNamespace(**(fields | overrides))
    return GPUModelRunner._resolve_topk_logits_k(runner), runner


def test_the_launch_switch_follows_the_model():
    assert resolve()[0] == K
    assert resolve(model_k=0)[0] == 0
    assert resolve(model_k=16)[0] == 16


@pytest.mark.parametrize(
    "override",
    [
        {"lora_config": object()},
        {"batch_sharder": object()},
        {"sampler": object()},
    ],
    ids=["lora", "batch-sharded", "custom-sampler"],
)
def test_the_launch_switch_turns_off_for_what_needs_the_logits(override):
    assert resolve(**override)[0] == 0


def test_speculative_decoding_leaves_the_launch_switch_on():
    # A step with draft tokens takes the full path by itself (the step predicate);
    # a step without any, such as dynamic speculative decoding at K == 0, may not.
    k, runner = resolve(speculative_config=object())
    assert k == K and runner.topk_logits_stats is not None
    # What else turns the path off still does under speculation.
    assert resolve(speculative_config=object(), lora_config=object())[0] == 0
    assert resolve(speculative_config=object(), batch_sharder=object())[0] == 0
    assert resolve(speculative_config=object(), sampler=object())[0] == 0


def test_a_model_without_the_path_is_left_alone():
    runner = SimpleNamespace(
        model=SimpleNamespace(),
        speculative_config=None,
        lora_config=None,
        batch_sharder=None,
        sampler=make_sampler(),
    )
    assert GPUModelRunner._resolve_topk_logits_k(runner) == 0


def test_the_min_tokens_knob(monkeypatch):
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_TOPK_LOGITS_MIN_TOKENS", "24")
    k, runner = resolve()
    assert k == K and runner.topk_logits_min_tokens == 24
    monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_TOPK_LOGITS_MIN_TOKENS", "0")
    k, runner = resolve()
    assert runner.topk_logits_min_tokens == 1


class _Recorder:
    def __init__(self, events: list, name: str, result=None):
        self.events, self.name, self.result = events, name, result

    def __call__(self, *args, **kwargs):
        self.events.append(self.name)
        return self.result


def sample_runner(events: list, speculative: bool, refuse: str | None = None):
    """The members of the runner that `GPUModelRunner.sample` reads."""
    out = SimpleNamespace(num_sampled="num_sampled", num_rejected="num_rejected")
    sampler = _Recorder(events, "sampler", out)
    sampler.topk_logits_fallback_reason = lambda idx, k: refuse
    sampler.topk_logits_threshold_route = lambda idx: False
    sampler.topk_logits_tie_draw = lambda batch: None
    sampler.sample_topk_logits = _Recorder(events, "sample_topk_logits", out)
    model = SimpleNamespace(
        compute_logits=lambda h: events.append("compute_logits") or h,
        compute_topk_logits=lambda h, ties: events.append("compute_topk_logits") or h,
    )
    runner = SimpleNamespace(
        batch_sharder=None,
        model=model,
        sampler=sampler,
        rejection_sampler=_Recorder(events, "rejection_sampler", out)
        if speculative
        else None,
        speculator=SimpleNamespace(draft_logits=None) if speculative else None,
        topk_logits_k=K,
        topk_logits_min_tokens=1,
        topk_logits_stats=RouteStats(K),
        topk_verify_level=0,
        pcp_manager=None,
        _topk_logits_logged=True,
    )
    runner._topk_logits_fallback_reason = functools.partial(
        GPUModelRunner._topk_logits_fallback_reason, runner
    )
    runner._sample_from_topk_logits = functools.partial(
        GPUModelRunner._sample_from_topk_logits, runner
    )
    return runner


def run_sample(runner, rows: int, reqs: int, drafts: int):
    batch = fake_batch(rows, draft_tokens=drafts, reqs=reqs)
    return GPUModelRunner.sample(runner, torch.zeros(rows, 4), batch, None)


@pytest.mark.parametrize(
    ("speculative", "refuse", "rows", "reqs", "drafts", "expected"),
    [
        # K == 0 under speculation: the plain top-k path, no rejection sampler.
        (True, None, 4, 4, 0, ["compute_topk_logits", "sample_topk_logits"]),
        # The same step needing the logits: the plain sampler, never the rejection one.
        (True, "logprobs", 4, 4, 0, ["compute_logits", "sampler"]),
        # Any draft token: the full logits and the rejection sampler.
        (True, None, 4 + 28, 4, 28, ["compute_logits", "rejection_sampler"]),
        (True, None, 4 + 1, 4, 1, ["compute_logits", "rejection_sampler"]),
        # Without speculation nothing changes.
        (False, None, 4, 4, 0, ["compute_topk_logits", "sample_topk_logits"]),
        (False, "logprobs", 4, 4, 0, ["compute_logits", "sampler"]),
    ],
    ids=[
        "k0-top-k",
        "k0-needs-logits",
        "full-block",
        "one-draft",
        "no-spec-top-k",
        "no-spec-logits",
    ],
)
def test_sample_routes_the_step(speculative, refuse, rows, reqs, drafts, expected):
    events: list = []
    runner = sample_runner(events, speculative, refuse)
    output, num_sampled, num_rejected = run_sample(runner, rows, reqs, drafts)
    assert events == expected
    assert (num_sampled, num_rejected) == ("num_sampled", "num_rejected")
