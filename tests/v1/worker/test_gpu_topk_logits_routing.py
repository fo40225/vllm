# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When a V2 step samples from the top-K logits of each TP rank, and when it does not.

Every trigger that needs more than the top K of the vocabulary, or the logits
themselves, has to send the step down the full path, and every TP rank has to decide
alike: the decision reads host copies of the request state, nothing rank-specific.
"""

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


def fake_runner(k: int = K, min_tokens: int = 1, supports: bool = True, **kwargs):
    sampler = SimpleNamespace(
        calls=[],
        topk_logits_supported=lambda idx, kk: (
            sampler.calls.append((idx.tolist(), kk)) or supports
        ),
    )
    return SimpleNamespace(
        topk_logits_k=k,
        topk_logits_min_tokens=min_tokens,
        pcp_manager=None,
        sampler=sampler,
        **kwargs,
    )


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
        runner.sampler.topk_logits_supported = lambda idx, k: False
    assert not step_uses_topk(runner, batch, grammar)


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
        {"speculative_config": object()},
        {"lora_config": object()},
        {"batch_sharder": object()},
        {"sampler": object()},
    ],
    ids=["speculative", "lora", "batch-sharded", "custom-sampler"],
)
def test_the_launch_switch_turns_off_for_what_needs_the_logits(override):
    assert resolve(**override)[0] == 0


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
