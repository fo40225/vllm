# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A DFlash-family draft step at K == 0 only stores the context K/V.

Dynamic speculative decoding schedules K == 0 for large batches: the next step then
verifies no draft token, so the draft step is dead weight. ``propose`` must still store
the context K/V of the step (the draft's KV cache stays complete for a later step with
K > 0) and nothing else: no attention metadata, no CUDA graph dispatch, no draft
forward. The runner cuts the returned drafts to K == 0 itself.

The K == 0 branch of ``propose`` is upstream's (the V2 runner skips drafting where no
request can use the drafts, #62784); it runs for dummy runs and data-parallel launches
too, because the DP ranks agree on it through ``DPSyncState.skip_drafts`` and dynamic
speculative decoding is disabled with data parallelism.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm.v1.worker.gpu.spec_decode.dflash.speculator as dflash_speculator
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import DFlash2Speculator
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator
from vllm.v1.worker.gpu.spec_decode.lilicorr.speculator import LiLiCorrSpeculator

pytestmark = pytest.mark.cpu_test

NUM_STEPS = 7
NUM_REQS = 3
NUM_TARGET_TOKENS = 10  # one prefill chunk and two decodes
MAX_TOKENS = 64
HIDDEN = 16
NUM_GROUPS = 2

FAMILY = [DFlashSpeculator, DFlash2Speculator, DSparkSpeculator, LiLiCorrSpeculator]


class _Drafting(Exception):
    """Raised by the patched graph dispatch: the draft step was started."""


class _Calls:
    def __init__(self):
        self.prepare_inputs = 0
        self.precompute: list[tuple] = []
        self.anchor = 0
        self.eplb = 0
        self.dispatch = 0
        self.generate = 0


class _DraftModel:
    def __init__(self, calls: _Calls):
        self.calls = calls

    def precompute_and_store_context_kv(self, states, positions, slots=None):
        self.calls.precompute.append((states.clone(), positions.clone(), slots))


def _setup(
    monkeypatch, cls, *, dp_size: int = 1, enable_adaptive_verification: bool = False
):
    calls = _Calls()

    def prepare_dflash_inputs(*args, **kwargs):
        calls.prepare_inputs += 1

    def dispatch(*args, **kwargs):
        calls.dispatch += 1
        raise _Drafting

    monkeypatch.setattr(
        dflash_speculator, "prepare_dflash_inputs", prepare_dflash_inputs
    )
    monkeypatch.setattr(dflash_speculator, "dispatch_cg_and_sync_dp", dispatch)

    spec = object.__new__(cls)
    spec.num_speculative_steps = NUM_STEPS
    spec.num_query_per_req = NUM_STEPS + 1
    spec.max_model_len = 4096
    spec.max_num_reqs = 8
    spec.max_num_tokens = MAX_TOKENS
    spec.dp_size = dp_size
    spec.dp_rank = 0
    spec.enable_adaptive_verification = enable_adaptive_verification
    spec.dcp_size = 1
    spec.pcp_manager = None
    spec.sample_from_anchor = False
    spec.parallel_drafting_token_id = 1
    spec.hidden_states = torch.zeros(MAX_TOKENS, HIDDEN)
    spec.context_positions = torch.arange(MAX_TOKENS, dtype=torch.int64)
    spec._context_slot_mappings = torch.arange(
        NUM_GROUPS * MAX_TOKENS, dtype=torch.int64
    ).view(NUM_GROUPS, MAX_TOKENS)
    spec._layer_group_idx = [0, 1, 0, 1, 0]
    spec.draft_kv_cache_group_ids = list(range(NUM_GROUPS))
    spec.draft_kv_cache_group_id = 0
    spec.block_tables = SimpleNamespace(
        slot_mappings=[None] * NUM_GROUPS,
        input_block_tables=[None] * NUM_GROUPS,
        block_sizes=[16] * NUM_GROUPS,
        cp_rank=0,
        cp_interleave=1,
    )
    for name in (
        "input_buffers",
        "sample_indices",
        "sample_pos",
        "sample_idx_mapping",
        "temperature",
        "seeds",
        "query_cudagraph_manager",
    ):
        setattr(spec, name, None)
    spec.draft_tokens = torch.full((8, NUM_STEPS), 5, dtype=torch.int64)
    spec.model = _DraftModel(calls)

    def count(name):
        def fn(*args, **kwargs):
            setattr(calls, name, getattr(calls, name) + 1)

        return fn

    spec.prepare_context_anchor = count("anchor")
    spec._prepare_eplb_forward = count("eplb")
    spec._generate_draft = count("generate")
    return spec, calls


def _propose(spec, num_speculative_tokens, **kwargs):
    input_batch = SimpleNamespace(
        num_reqs=NUM_REQS,
        num_tokens=NUM_TARGET_TOKENS,
        seq_lens_cpu_upper_bound=torch.tensor([40, 21, 30], dtype=torch.int32),
    )
    hidden = torch.arange(NUM_TARGET_TOKENS * HIDDEN, dtype=torch.float32).view(
        NUM_TARGET_TOKENS, HIDDEN
    )
    drafts = spec.propose(
        input_batch,
        {},
        {},
        hidden,
        None,
        torch.ones(NUM_REQS, dtype=torch.int32),
        torch.zeros(NUM_REQS, dtype=torch.int32),
        None,
        None,
        None,
        None,
        num_speculative_tokens=num_speculative_tokens,
        **kwargs,
    )
    return drafts, hidden


@pytest.mark.parametrize("cls", FAMILY)
def test_the_family_shares_one_propose(cls):
    # The K == 0 branch lives in one place and every drafter of the family has it.
    assert cls.propose is DFlashSpeculator.propose


@pytest.mark.parametrize("cls", FAMILY)
def test_k0_stores_the_context_and_runs_no_draft_step(monkeypatch, cls):
    spec, calls = _setup(monkeypatch, cls)

    drafts, hidden = _propose(spec, 0)

    # The context of every row of the step, for every layer's cache group.
    assert calls.prepare_inputs == NUM_GROUPS
    assert len(calls.precompute) == 1
    states, positions, slots = calls.precompute[0]
    assert torch.equal(states, hidden)
    assert torch.equal(positions, spec.context_positions[:NUM_TARGET_TOKENS])
    assert slots is not None and len(slots) == len(spec._layer_group_idx)
    for slot, gidx in zip(slots, spec._layer_group_idx):
        assert torch.equal(slot, spec._context_slot_mappings[gidx][:NUM_TARGET_TOKENS])
    # Nothing of the draft step.
    assert calls.dispatch == 0
    assert calls.generate == 0
    assert calls.eplb == 0
    assert calls.anchor == 0
    # The buffer the runner reads is returned untouched: the runner sets the rows to -1
    # for the K of the step.
    assert drafts.shape == (NUM_REQS, NUM_STEPS)
    assert drafts.data_ptr() == spec.draft_tokens.data_ptr()
    assert (spec.draft_tokens == 5).all()


@pytest.mark.parametrize("num_speculative_tokens", [None, NUM_STEPS, 3, 1])
def test_a_step_with_drafts_starts_the_draft_step(monkeypatch, num_speculative_tokens):
    # K in 1 .. N - 1 drafts the whole block too (the runner cuts it to K).
    spec, calls = _setup(monkeypatch, DSparkSpeculator)

    with pytest.raises(_Drafting):
        _propose(spec, num_speculative_tokens)

    assert calls.dispatch == 1
    assert calls.precompute == []
    assert (spec.draft_tokens == 5).all()


def test_a_dummy_run_at_k0_stores_no_context_slots(monkeypatch):
    # A dummy run has no slot mapping to store into: the context K/V is computed only.
    spec, calls = _setup(monkeypatch, DFlashSpeculator)

    _propose(spec, 0, dummy_run=True)

    assert calls.dispatch == 0
    assert len(calls.precompute) == 1
    assert calls.precompute[0][2] is None


def test_a_data_parallel_launch_skips_the_draft_step_at_k0(monkeypatch):
    # Every DP rank reaches K == 0 together (DPSyncState.skip_drafts), so no rank
    # waits in the collective of the dispatch.
    spec, calls = _setup(monkeypatch, DFlashSpeculator, dp_size=2)

    _propose(spec, 0)

    assert calls.dispatch == 0
    assert len(calls.precompute) == 1


def test_adaptive_verification_always_drafts(monkeypatch):
    # The runner records the draft confidences after every propose and the acceptance
    # estimator reads the draft step's features: a skipped draft step would leave both
    # with the values of an older step.
    spec, calls = _setup(
        monkeypatch, DSparkSpeculator, enable_adaptive_verification=True
    )

    with pytest.raises(_Drafting):
        _propose(spec, 0)

    assert calls.dispatch == 1
    assert calls.precompute == []
