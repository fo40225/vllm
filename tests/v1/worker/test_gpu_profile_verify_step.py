# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The memory profile runs the verify step of a full speculative decode batch.

A speculative step passes ``decode_query_len`` rows of every request through the
rejection sampler, 8 times the rows of the plain dummy sampler run, with the hidden
states and the aux taps of the forward still alive. The profile peak, and with it
``--kv-cache-memory-bytes`` sizing, has to cover that step.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.worker.gpu.input_batch import InputBuffers
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

pytestmark = pytest.mark.cpu_test

DEVICE = torch.device("cpu")


class _RecordingRejectionSampler:
    def __init__(self, watermark_key=None, fail=False):
        self.watermark_key = watermark_key
        self.enable_adaptive_verification = True
        self.fail = fail
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
