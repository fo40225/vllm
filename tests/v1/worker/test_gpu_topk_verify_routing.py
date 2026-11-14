# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""When a speculative step verifies its drafts from the top-K logits of each TP rank.

The step has to take the full path whenever it needs more than the top K of the
vocabulary, or the logits themselves, and every TP rank has to decide alike: the
decision reads the batch shape and the host copy of the request state, nothing
rank-specific, and no device data.
"""

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for top-k verify routing tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.topk_logits import RouteStats
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler

from .test_gpu_topk_logits_routing import (
    GREEDY,
    T1K64,
    VOCAB_SIZE,
    K,
    fake_batch,
    sampler_with,
)

STEPS = 7


def plan(sampler, rows, sampled: bool, k: int = K):
    return sampler.topk_verify_plan(np.asarray(rows, dtype=np.intp), k, sampled)


# --------------------------------------------------------------------------------
# The sampler: the reason and the candidates per rank.
# --------------------------------------------------------------------------------


def test_a_step_of_greedy_requests_needs_one_candidate_per_rank():
    sampler = sampler_with(GREEDY, GREEDY, SamplingParams(temperature=0.0, top_k=500))
    for sampled in (False, True):
        assert plan(sampler, [0, 1, 2], sampled) == (None, 1)


def test_sampled_rows_need_the_launch_to_allow_them():
    sampler = sampler_with(GREEDY, T1K64)
    assert plan(sampler, [0, 1], sampled=False) == ("sampled-rows", 0)
    assert plan(sampler, [0, 1], sampled=True) == (None, K)
    # A top_k above the candidates of the launch is not covered.
    assert plan(sampler, [1], sampled=True, k=16) == ("top-k-unset-or-above-k", 0)
    # A greedy-only subset of the same sampler is still one candidate per rank.
    assert plan(sampler, [0], sampled=False) == (None, 1)


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        (SamplingParams(temperature=1.0, top_k=64, top_p=0.9), "top-p-or-min-p"),
        (SamplingParams(temperature=1.0, top_k=64, min_p=0.1), "top-p-or-min-p"),
        (SamplingParams(temperature=1.0, top_k=65), "top-k-unset-or-above-k"),
        (SamplingParams(temperature=1.0), "top-k-unset-or-above-k"),
        (SamplingParams(temperature=0.0, logprobs=1), "logprobs"),
        (SamplingParams(temperature=1.0, top_k=64, logprobs=0), "logprobs"),
        (SamplingParams(temperature=0.0, frequency_penalty=0.1), "logits-processing"),
        (
            SamplingParams(temperature=1.0, top_k=64, logit_bias={3: 1.0}),
            "logits-processing",
        ),
    ],
)
def test_one_request_that_needs_the_logits_sends_the_step_to_the_full_path(
    params, reason
):
    sampler = sampler_with(T1K64, params)
    assert plan(sampler, [0, 1], sampled=True) == (reason, 0)
    # The same request alone is just as unsupported; its neighbours do not matter.
    assert plan(sampler, [1], sampled=True)[0] == reason


def test_the_plan_reads_the_rows_of_the_batch_only():
    sampler = sampler_with(T1K64, GREEDY, SamplingParams(temperature=0.0, logprobs=1))
    assert plan(sampler, [0, 1], sampled=True) == (None, K)
    assert plan(sampler, [1], sampled=True) == (None, 1)
    assert plan(sampler, [2], sampled=True) == ("logprobs", 0)


# --------------------------------------------------------------------------------
# The launch: the level and what turns it off.
# --------------------------------------------------------------------------------


def rejection_sampler(**kwargs) -> RejectionSampler:
    rs = object.__new__(RejectionSampler)
    rs.watermark_key = None
    rs.synthetic_conditional_rates = None
    rs.enable_adaptive_verification = False
    rs.fly_window_size = 0
    rs.use_block_verification = True
    rs.num_speculative_steps = STEPS
    for name, value in kwargs.items():
        setattr(rs, name, value)
    return rs


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({}, None),
        ({"watermark_key": 5}, "a watermark is on"),
        (
            {"synthetic_conditional_rates": torch.ones(1)},
            "synthetic acceptance rates are on",
        ),
        ({"enable_adaptive_verification": True}, "adaptive verification is on"),
        ({"fly_window_size": 2}, "FLY verification is on"),
    ],
)
def test_the_launch_reason_of_the_rejection_sampler(kwargs, reason):
    assert rejection_sampler(**kwargs).topk_verify_launch_reason == reason


def resolve(
    monkeypatch, level, k=K, width=VOCAB_SIZE, split_model=True, **sampler_kwargs
):
    if level is None:
        monkeypatch.delenv("VLLM_MUSE_SPARK_SM120_VERIFY_TOPK_LOGITS", raising=False)
    else:
        monkeypatch.setenv("VLLM_MUSE_SPARK_SM120_VERIFY_TOPK_LOGITS", str(level))
    runner = SimpleNamespace(
        topk_logits_k=k,
        vocab_size=VOCAB_SIZE,
        rejection_sampler=rejection_sampler(**sampler_kwargs),
        speculator=SimpleNamespace(
            draft_logits=torch.empty(2, STEPS, width, device="meta")
        ),
        topk_verify_sampled=False,
        topk_verify_stats=None,
        model=(
            SimpleNamespace(compute_split_verify_logits=lambda *a, **kw: None)
            if split_model
            else SimpleNamespace()
        ),
    )
    return GPUModelRunner._resolve_topk_verify_level(runner), runner


@pytest.mark.parametrize("level", [0, 1, 2, 3, 4])
def test_the_level_follows_the_flag(monkeypatch, level):
    got, runner = resolve(monkeypatch, level)
    assert got == level
    assert runner.topk_verify_sampled == (level >= 2)
    assert (runner.topk_verify_stats is not None) == (level > 0)


@pytest.mark.parametrize("level", [3, 4])
def test_the_split_levels_need_a_model_that_splits_the_gather(monkeypatch, level):
    """Without ``compute_split_verify_logits`` the launch keeps level 2: a step with a
    request that needs the logits verifies from the full logits as before."""
    got, runner = resolve(monkeypatch, level, split_model=False)
    assert got == 2 and runner.topk_verify_sampled


def test_the_flag_is_on_by_default(monkeypatch):
    """The default is level 2: greedy and sampled steps are verified from the
    candidates (the accuracy policy: identical for greedy, distribution-exact for
    sampled rows)."""
    got, runner = resolve(monkeypatch, None)
    assert got == 2 and runner.topk_verify_sampled
    assert runner.topk_verify_stats is not None


@pytest.mark.parametrize(
    "case",
    [
        "bad-level",
        "no-topk-path",
        "no-rejection-sampler",
        "watermark",
        "synthetic",
        "adaptive",
        "rejection-sampler-subclass",
        "narrow-draft-vocabulary",
    ],
)
@pytest.mark.parametrize("level", [1, 2])
def test_what_turns_the_path_off(monkeypatch, case, level):
    k, kwargs, width = K, {}, VOCAB_SIZE
    if case == "bad-level":
        level = 5
    elif case == "rejection-sampler-subclass":
        # A subclass may override what the path replaces: keep its full path.
        kwargs = {"__class__": type("Sub", (RejectionSampler,), {})}
    elif case == "narrow-draft-vocabulary":
        # The full path cuts every pass, the greedy argmax too, to the draft's width.
        width = VOCAB_SIZE - 1
    elif case == "no-topk-path":
        k = 0
    elif case == "watermark":
        kwargs = {"watermark_key": 1}
    elif case == "synthetic":
        kwargs = {"synthetic_conditional_rates": torch.ones(1)}
    elif case == "adaptive":
        kwargs = {"enable_adaptive_verification": True}
    got, runner = resolve(monkeypatch, level, k=k, width=width, **kwargs)
    if case == "no-rejection-sampler":
        runner.rejection_sampler = None
        got = GPUModelRunner._resolve_topk_verify_level(runner)
    assert got == 0


@pytest.mark.parametrize(
    "case",
    [
        "standard-method",
        "no-draft-logits",
        "wide-draft-vocabulary",
    ],
)
def test_sampled_rows_need_the_block_method_and_a_draft_of_the_vocabulary_width(
    monkeypatch, case
):
    kwargs, width = {}, VOCAB_SIZE
    if case == "standard-method":
        kwargs = {"use_block_verification": False}
    elif case == "wide-draft-vocabulary":
        # The full path restricts the draft sums to the target's width: not reproduced.
        width = VOCAB_SIZE + 8
    got, runner = resolve(monkeypatch, 2, width=width, **kwargs)
    if case == "no-draft-logits":
        runner.speculator.draft_logits = None
        got = GPUModelRunner._resolve_topk_verify_level(runner)
    # The greedy steps keep the path; sampled requests keep the full one.
    assert got == 2 and not runner.topk_verify_sampled


# --------------------------------------------------------------------------------
# The runner: the step predicate and the routing of `sample`.
# --------------------------------------------------------------------------------


class _Recorder:
    def __init__(self, events: list, name: str, result=None):
        self.events, self.name, self.result = events, name, result
        self.calls: list = []

    def __call__(self, *args, **kwargs):
        self.events.append(self.name)
        self.calls.append((args, kwargs))
        return self.result


def verify_runner(events: list, plan_result=(None, 1), level: int = 2, **kwargs):
    out = SimpleNamespace(num_sampled="num_sampled", num_rejected="num_rejected")
    sampler = _Recorder(events, "sampler", out)
    sampler.topk_verify_plan = lambda idx, k, sampled: (
        events.append(f"plan(k={k}, sampled={sampled})") or plan_result
    )
    sampler.topk_logits_fallback_reason = lambda idx, k: "unexpected"
    sampler.topk_logits_threshold_route = lambda idx: False
    sampler.topk_logits_tie_draw = lambda batch: None
    sampler.sample_topk_logits = _Recorder(events, "sample_topk_logits", out)
    sampler.sampling_states = SimpleNamespace(
        temperature=SimpleNamespace(np=np.asarray([0.0, 1.0] * 32, dtype=np.float32))
    )
    model = SimpleNamespace(
        compute_logits=lambda h: events.append("compute_logits") or h,
        compute_topk_logits=lambda h, ties=None, k=None: events.append(
            f"compute_topk_logits(k={k})"
        )
        or h,
    )
    rejection = _Recorder(events, "rejection_sampler", out)
    rejection.verify_topk = _Recorder(events, "verify_topk", out)
    runner = SimpleNamespace(
        batch_sharder=None,
        model=model,
        sampler=sampler,
        rejection_sampler=rejection,
        speculator=SimpleNamespace(draft_logits="draft_logits"),
        topk_logits_k=K,
        topk_logits_min_tokens=1,
        topk_logits_stats=RouteStats(K),
        topk_verify_level=level,
        topk_verify_sampled=True,
        topk_verify_stats=RouteStats(
            K, label="steps with drafts", threshold_note=False, action="verified from"
        ),
        _topk_verify_logged=True,
        _topk_logits_logged=True,
        adaptive_verification=None,
        pcp_manager=None,
        **kwargs,
    )
    runner._topk_verify_plan = functools.partial(
        GPUModelRunner._topk_verify_plan, runner
    )
    runner._verify_from_topk_logits = functools.partial(
        GPUModelRunner._verify_from_topk_logits, runner
    )
    runner._topk_logits_fallback_reason = functools.partial(
        GPUModelRunner._topk_logits_fallback_reason, runner
    )
    runner._sample_from_topk_logits = functools.partial(
        GPUModelRunner._sample_from_topk_logits, runner
    )
    return runner


def run_sample(runner, rows: int, reqs: int, drafts: int, grammar=None):
    batch = fake_batch(rows, draft_tokens=drafts, reqs=reqs)
    return GPUModelRunner.sample(runner, torch.zeros(rows, 4), batch, grammar)


def test_a_verify_step_takes_the_candidates():
    events: list = []
    runner = verify_runner(events, plan_result=(None, K))
    output, num_sampled, num_rejected = run_sample(runner, 4 + 28, 4, 28)
    assert events == [
        f"plan(k={K}, sampled=True)",
        f"compute_topk_logits(k={K})",
        "verify_topk",
    ]
    ((args, _),) = runner.rejection_sampler.verify_topk.calls
    # The candidates, the batch, the draft logits and "has a sampled row".
    assert args[2] == "draft_logits" and args[3] is True
    assert (num_sampled, num_rejected) == ("num_sampled", "num_rejected")
    assert runner.topk_verify_stats.taken == 1
    # The plain top-k path never saw the step.
    assert runner.topk_logits_stats.steps == 0


def test_a_greedy_verify_step_takes_one_candidate_and_no_draft_statistics():
    events: list = []
    runner = verify_runner(events, plan_result=(None, 1))
    runner.sampler.sampling_states.temperature.np[:] = 0.0
    run_sample(runner, 4 + 28, 4, 28)
    assert events[1:] == ["compute_topk_logits(k=1)", "verify_topk"]
    ((args, _),) = runner.rejection_sampler.verify_topk.calls
    assert args[3] is False


@pytest.mark.parametrize("reason", ["sampled-rows", "logprobs", "top-p-or-min-p"])
def test_a_step_the_sampler_refuses_takes_the_full_path(reason):
    events: list = []
    runner = verify_runner(events, plan_result=(reason, 0))
    run_sample(runner, 4 + 28, 4, 28)
    assert events == [
        f"plan(k={K}, sampled=True)",
        "compute_logits",
        "rejection_sampler",
    ]
    assert runner.topk_verify_stats.fallbacks == {reason: 1}
    # Counted once, by the verify counters.
    assert runner.topk_logits_stats.steps == 0


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("grammar", "grammar"),
        ("context-parallel", "context-parallel"),
        ("adaptive", "adaptive-verification"),
    ],
)
def test_the_step_predicate_falls_back(case, reason):
    events: list = []
    runner = verify_runner(events)
    grammar = None
    if case == "grammar":
        grammar = object()
    elif case == "context-parallel":
        runner.pcp_manager = object()
    else:
        runner.adaptive_verification = object()
    batch = fake_batch(4 + 28, draft_tokens=28, reqs=4)
    got, k = GPUModelRunner._topk_verify_plan(runner, batch, grammar)
    assert (got, k) == (reason, 0)
    # The sampler is not asked: the batch shape and the launch decide alone.
    assert not any(e.startswith("plan(") for e in events)


def test_a_step_without_drafts_keeps_the_plain_top_k_path():
    events: list = []
    runner = verify_runner(events)
    runner.sampler.topk_logits_fallback_reason = lambda idx, k: None
    run_sample(runner, 4, 4, 0)
    assert "verify_topk" not in events and "rejection_sampler" not in events
    assert events == ["compute_topk_logits(k=None)", "sample_topk_logits"]


def test_a_launch_without_the_verify_path_is_unchanged():
    events: list = []
    runner = verify_runner(events, level=0)
    run_sample(runner, 4 + 28, 4, 28)
    assert events == ["compute_logits", "rejection_sampler"]
    assert runner.topk_logits_stats.fallbacks == {"draft-tokens": 1}
    assert runner.topk_verify_stats.steps == 0


def test_the_step_predicate_reads_no_device_data():
    events: list = []
    runner = verify_runner(events, plan_result=(None, 1))
    batch = fake_batch(4 + 28, draft_tokens=28, reqs=4)
    batch.logits_indices = torch.empty(32, dtype=torch.int64, device="meta")
    assert GPUModelRunner._topk_verify_plan(runner, batch, None) == (None, 1)


def test_the_route_stats_of_verified_steps_say_what_they_count():
    stats = RouteStats(
        K, label="steps with drafts", threshold_note=False, action="verified from"
    )
    line = stats.record(None)
    assert (
        line is not None
        and "1 of 1 steps with drafts (100.0%) verified from the top-64" in line
    )
    assert "FlashInfer" not in line
    line = stats.record("sampled-rows")
    assert line is not None and "sampled-rows: 1" in line


def _verify_topk_args(temperature, top_k, k=K, min_p=None, top_p=None):
    rs = rejection_sampler()
    rs.sampler = SimpleNamespace(
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(np=np.asarray(temperature, dtype=np.float32)),
            top_k=SimpleNamespace(np=np.asarray(top_k, dtype=np.int32)),
            min_p=SimpleNamespace(
                np=np.asarray(min_p or [0.0] * len(temperature), dtype=np.float32)
            ),
            top_p=SimpleNamespace(
                np=np.asarray(top_p or [1.0] * len(temperature), dtype=np.float32)
            ),
        )
    )
    batch = SimpleNamespace(idx_mapping_np=np.arange(len(temperature), dtype=np.intp))
    topk = SimpleNamespace(k=k, tp_size=8, shard_size=25256)
    return rs, topk, batch


def test_verify_topk_refuses_a_top_k_above_the_candidates():
    """A misrouted step must fail, not truncate the target distribution silently."""
    rs, topk, batch = _verify_topk_args([0.0, 1.0, 1.0], [VOCAB_SIZE, K, K + 1])
    with pytest.raises(ValueError, match="top_k above the 64 candidates"):
        rs.verify_topk(topk, batch, None, True)


def test_verify_topk_accepts_what_the_candidates_cover():
    """A greedy row ignores its top_k; a sampled row at k is covered. The check passes
    and the stub batch then fails at the first device access (the gather)."""
    rs, topk, batch = _verify_topk_args([0.0, 1.0], [VOCAB_SIZE, K])
    with pytest.raises(AttributeError):
        rs.verify_topk(topk, batch, None, True)
    # A batch without sampled rows needs no check at all.
    rs, topk, batch = _verify_topk_args([0.0, 0.0], [VOCAB_SIZE, VOCAB_SIZE], k=1)
    with pytest.raises(AttributeError):
        rs.verify_topk(topk, batch, None, False)


def test_verify_topk_refuses_a_min_p_or_top_p_the_call_does_not_apply():
    """A step routed without the kernel variant must fail, not drop the parameter."""
    rs, topk, batch = _verify_topk_args([1.0, 1.0], [K, K], min_p=[0.0, 0.1])
    with pytest.raises(ValueError, match="min_p the call does not apply"):
        rs.verify_topk(topk, batch, None, True)
    rs, topk, batch = _verify_topk_args([1.0, 1.0], [K, K], top_p=[1.0, 0.9])
    with pytest.raises(ValueError, match="top_p the call does not apply"):
        rs.verify_topk(topk, batch, None, True)
    # Applied: the checks pass and the stub batch fails at the first device access.
    rs, topk, batch = _verify_topk_args(
        [1.0, 1.0], [K, K], min_p=[0.0, 0.1], top_p=[1.0, 0.9]
    )
    with pytest.raises(AttributeError):
        rs.verify_topk(topk, batch, None, True, use_min_p=True, use_top_p=True)
    # A greedy row ignores both.
    rs, topk, batch = _verify_topk_args([0.0, 1.0], [K, K], min_p=[0.2, 0.0])
    with pytest.raises(AttributeError):
        rs.verify_topk(topk, batch, None, True)


def test_verify_topk_refuses_a_top_k_the_min_p_emulation_does_not_cover():
    """The full path's outlier pivot has the sigma for top_k / vocab * 200 < 1."""
    rs, topk, batch = _verify_topk_args([1.0], [K], min_p=[0.1])
    topk.shard_size = 100  # a vocabulary of 800: 200 * 64 > 800
    with pytest.raises(ValueError, match="top_k too large for min_p"):
        rs.verify_topk(topk, batch, None, True, use_min_p=True)
