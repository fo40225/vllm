# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The memory profile runs the verify step of a full speculative decode batch.

A speculative step passes ``decode_query_len`` rows of every request through the
rejection sampler, 8 times the rows of the plain dummy sampler run, with the hidden
states and the aux taps of the forward still alive. The profile peak, and with it
``--kv-cache-memory-bytes`` sizing, has to cover that step.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu import model_runner as model_runner_module
from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test

DEVICE = torch.device("cpu")


class _RecordingRejectionSampler:
    def __init__(self, watermark_key=None, fail=False):
        self.watermark_key = watermark_key
        self.enable_adaptive_verification = True
        self.fail = fail
        self.output_factory = None
        self.calls: list[dict] = []

    def __call__(self, logits, batch, draft_logits, invalid_drafts):
        self.calls.append(
            dict(
                logits=logits,
                batch=batch,
                draft_logits=draft_logits,
                invalid_drafts=invalid_drafts,
                adaptive=self.enable_adaptive_verification,
            )
        )
        if self.fail:
            raise RuntimeError("out of memory")
        if self.output_factory is not None:
            return self.output_factory(batch)


def _runner(rejection_sampler, *, max_num_reqs=4, query_len=4, max_tokens=64):
    draft_logits = torch.zeros(max_num_reqs, query_len - 1, 8)
    model = SimpleNamespace(
        compute_logits=lambda hidden: torch.zeros(hidden.shape[0], 8)
    )
    return SimpleNamespace(
        device=DEVICE,
        max_num_reqs=max_num_reqs,
        decode_query_len=query_len,
        input_buffers=InputBuffers(max_num_reqs, max_tokens, DEVICE),
        rejection_sampler=rejection_sampler,
        speculator=SimpleNamespace(draft_logits=draft_logits),
        model=model,
        topk_verify_level=0,
        topk_verify_sampled=False,
        topk_logits_k=64,
    )


def test_the_dummy_verify_step_has_max_num_reqs_times_the_query_length_rows():
    sampler = _RecordingRejectionSampler()
    runner = _runner(sampler)
    hidden = torch.zeros(64, 5)

    GPUModelRunner._dummy_rejection_sampler_run(runner, hidden)

    (call,) = sampler.calls
    batch = call["batch"]
    assert call["logits"].shape == (16, 8)
    assert call["draft_logits"] is runner.speculator.draft_logits
    assert call["invalid_drafts"] is None
    assert batch.num_reqs == 4
    assert batch.num_draft_tokens == 12
    assert batch.num_draft_tokens_per_req.tolist() == [3, 3, 3, 3]
    assert batch.cu_num_logits_np.tolist() == [0, 4, 8, 12, 16]
    assert batch.cu_num_logits.tolist() == [0, 4, 8, 12, 16]
    assert batch.logits_indices.tolist() == list(range(16))
    assert batch.expanded_idx_mapping.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4
    assert batch.expanded_local_pos.tolist() == [0, 1, 2, 3] * 4


def test_the_rows_never_exceed_the_hidden_states_of_the_forward():
    sampler = _RecordingRejectionSampler()
    runner = _runner(sampler, max_num_reqs=16, query_len=4)

    GPUModelRunner._dummy_rejection_sampler_run(runner, torch.zeros(10, 5))

    (call,) = sampler.calls
    assert call["batch"].num_reqs == 2  # 10 rows // 4
    assert call["logits"].shape[0] == 8


def test_adaptive_verification_is_off_during_the_call_and_restored():
    sampler = _RecordingRejectionSampler()
    runner = _runner(sampler)

    GPUModelRunner._dummy_rejection_sampler_run(runner, torch.zeros(64, 5))

    assert sampler.calls[0]["adaptive"] is False
    assert sampler.enable_adaptive_verification is True


def test_adaptive_verification_is_restored_when_the_call_fails():
    sampler = _RecordingRejectionSampler(fail=True)
    runner = _runner(sampler)

    with pytest.raises(RuntimeError, match="out of memory"):
        GPUModelRunner._dummy_rejection_sampler_run(runner, torch.zeros(64, 5))

    assert sampler.enable_adaptive_verification is True


@pytest.mark.parametrize(
    ("sampled", "shapes"),
    [(False, [(1, False)]), (True, [(1, False), (64, True)])],
    ids=["greedy-steps-only", "with-sampled-rows"],
)
def test_the_dummy_verify_step_compiles_the_candidate_path_after_the_full_one(
    sampled, shapes
):
    """The full path runs first (fallback steps need its memory, so the memory profile
    must cover it), then the verification from the top-k logits once per shape."""
    events: list = []
    sampler = _RecordingRejectionSampler()
    sampler.verify_topk = lambda topk, batch, draft_logits, has_sampled, **kwargs: (
        events.append(("verify_topk", topk, has_sampled))
    )
    runner = _runner(sampler)
    runner.topk_verify_level = 2 if sampled else 1
    runner.topk_verify_sampled = sampled
    runner.model.compute_topk_logits = lambda hidden, k=None: ("topk", k)

    GPUModelRunner._dummy_rejection_sampler_run(runner, torch.zeros(64, 5))

    assert len(sampler.calls) == 1  # the full path, once
    assert events == [("verify_topk", ("topk", k), has) for k, has in shapes]


def test_a_launch_without_the_verify_path_compiles_nothing_extra():
    sampler = _RecordingRejectionSampler()
    sampler.verify_topk = lambda *args: pytest.fail("the candidate path ran")
    runner = _runner(sampler)

    GPUModelRunner._dummy_rejection_sampler_run(runner, torch.zeros(64, 5))

    assert len(sampler.calls) == 1


def test_a_watermarked_run_is_skipped():
    sampler = _RecordingRejectionSampler(watermark_key=7)
    runner = _runner(sampler)

    GPUModelRunner._dummy_rejection_sampler_run(runner, torch.zeros(64, 5))

    assert sampler.calls == []


def _profile_runner(events, *, speculative: bool):
    """A runner whose profile_run only records the order of its steps."""
    hidden = torch.zeros(64, 5)
    taps = [torch.zeros(64, 5)]

    def dummy_run(num_tokens, **kwargs):
        events.append(("forward", kwargs.get("keep_aux_hidden_states", False)))
        if kwargs.get("keep_aux_hidden_states"):
            runner._profile_aux_hidden_states = taps
        return hidden, hidden[:4]

    def verify(hidden_states):
        events.append(("verify", runner._profile_aux_hidden_states is taps))

    runner = SimpleNamespace(
        supports_mm_inputs=False,
        max_num_tokens=64,
        is_last_pp_rank=True,
        pooling_runner=None,
        speculator=object() if speculative else None,
        rejection_sampler=object() if speculative else None,
        _profile_aux_hidden_states=None,
        _dummy_run=dummy_run,
        _dummy_sampler_run=lambda h: events.append(("sample", None)),
        _dummy_rejection_sampler_run=verify,
        _dummy_pooler_run=lambda h: events.append(("pool", None)),
        _profile_microbatched_step=lambda randomize: events.append(("micro", None)),
        reset_encoder_cache=lambda: None,
    )
    return runner


def test_profile_run_verifies_while_the_aux_taps_are_alive():
    events: list = []
    runner = _profile_runner(events, speculative=True)

    GPUModelRunner.profile_run(runner)

    assert events == [
        ("forward", True),
        ("sample", None),
        ("verify", True),
        ("micro", None),
    ]
    # The taps are not kept past the profile (the microbatched step must not hold them).
    assert runner._profile_aux_hidden_states is None


def test_profile_run_without_speculative_decoding_is_unchanged():
    events: list = []
    runner = _profile_runner(events, speculative=False)

    GPUModelRunner.profile_run(runner)

    assert [e[0] for e in events] == ["forward", "sample", "micro"]
    assert runner._profile_aux_hidden_states is None


def _split_output(num_reqs: int):
    return SimpleNamespace(
        sampled_token_ids=torch.zeros(num_reqs, 8, dtype=torch.int64),
        logprobs_tensors=None,
        num_nans=None,
        num_sampled=torch.ones(num_reqs, dtype=torch.int32),
        num_rejected=torch.zeros(num_reqs, dtype=torch.int32),
        sampling_mask_tensors=None,
    )


@pytest.mark.parametrize(
    ("level", "variants"),
    [
        (2, [(1, False, False, False), (64, True, False, False)]),
        (
            3,
            [
                (1, False, False, False),
                (64, True, False, False),
                (64, True, True, False),
            ],
        ),
        (
            4,
            [
                (1, False, False, False),
                (64, True, False, False),
                (64, True, True, False),
                (64, True, False, True),
                (64, True, True, True),
            ],
        ),
    ],
)
def test_the_dummy_verify_step_compiles_every_variant_the_level_can_launch(
    level, variants, monkeypatch
):
    """The kernel variants (sampled branch, min_p, top_p) a level can launch compile
    before the first request, and from level 3 one split verification (the fewest
    candidate requests the split allows, the others from the logits) runs on the real
    dummy batch: its kernels compile and its gathers are allocated once."""
    # The dummy batch of this test holds 4 requests: a minimum of 2 candidate requests.
    monkeypatch.setattr(model_runner_module, "SPLIT_MIN_CANDIDATE_REQS", 2)
    events: list = []
    sampler = _RecordingRejectionSampler()
    kinds: list = []
    sampler.verify_topk = lambda topk, batch, draft_logits, has_sampled, **kw: (
        events.append(
            (
                "verify_topk",
                topk,
                has_sampled,
                kw.get("use_min_p", False),
                kw.get("use_top_p", False),
                batch.num_reqs,
            )
        )
        or _split_output(batch.num_reqs)
    )
    calls = sampler.calls
    runner = _runner(sampler)
    runner.topk_verify_level = level
    runner.topk_verify_sampled = True
    runner.model.compute_topk_logits = lambda hidden, k=None: ("topk", k)
    runner._dummy_split_verify = functools.partial(
        GPUModelRunner._dummy_split_verify, runner
    )
    runner.model.compute_split_verify_logits = lambda hidden, k, cand, full: (
        kinds.append((k, cand.tolist(), full.tolist())) or ("topk", "logits")
    )
    # The full path of the split step returns an output too.
    sampler.output_factory = lambda batch: _split_output(batch.num_reqs)

    GPUModelRunner._dummy_rejection_sampler_run(runner, torch.zeros(64, 5))

    whole = [e for e in events if e[5] == 4]
    assert [(e[1][1], e[2], e[3], e[4]) for e in whole] == variants
    if level < 3:
        assert len(events) == len(variants) and not kinds
        return
    # One split verification: requests 0 and 1 (the candidate minimum) from the
    # candidates, the others from the logits.
    (k, cand_rows, full_rows) = kinds[0]
    assert k == 64
    assert cand_rows == [0, 1, 2, 3, 4, 5, 6, 7]
    assert full_rows == [8, 9, 10, 11, 12, 13, 14, 15]
    (split,) = [e for e in events if e[5] == 2]
    assert split[1] == "topk"
    # The full path ran once on the whole batch and once on the view of two requests.
    assert [c["batch"].num_reqs for c in calls] == [4, 2]
    assert calls[1]["logits"] == "logits"


def test_the_dummy_verify_step_runs_no_split_below_the_candidate_minimum():
    """A dummy batch of 4 requests cannot hold the 5 candidate requests of the real
    minimum and a full request: the dummy step runs the whole-batch variants only."""
    events: list = []
    sampler = _RecordingRejectionSampler()
    sampler.verify_topk = lambda topk, batch, draft_logits, has_sampled, **kw: (
        events.append(batch.num_reqs) or _split_output(batch.num_reqs)
    )
    runner = _runner(sampler)
    runner.topk_verify_level = 3
    runner.topk_verify_sampled = True
    runner.model.compute_topk_logits = lambda hidden, k=None: ("topk", k)
    runner._dummy_split_verify = functools.partial(
        GPUModelRunner._dummy_split_verify, runner
    )
    GPUModelRunner._dummy_rejection_sampler_run(runner, torch.zeros(64, 5))
    assert events == [4, 4, 4] and [c["batch"].num_reqs for c in sampler.calls] == [4]
