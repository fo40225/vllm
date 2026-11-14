# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer plan() fence: the staging of one plan() is not rewritten before its copy.

plan() stages its schedule in a persistent page-locked workspace and enqueues the
host-to-device copy without waiting. With GPU work queued in front of the copy (the
draft of the previous step under speculation), the next plan() of the same wrapper
rewrites the staging before the copy has run unless something orders them. The tests
queue a long `torch.cuda._sleep` in front of the copy to make that deterministic.
"""

import pytest
import torch

from vllm.v1.attention.backends.flashinfer import PlanFence

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

SLEEP_CYCLES = 200_000_000  # about 0.1 s at the SM clock; the copy waits behind it


def _race_on_a_staging_buffer(fence_on: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Two plans on one 'wrapper' (a pinned staging buffer and its device copy)."""
    dev = torch.zeros(4096, dtype=torch.int32, device="cuda")
    staging = torch.zeros(4096, dtype=torch.int32).pin_memory()
    fence, wrapper = PlanFence(), object()
    torch.cuda.synchronize()

    def plan(value: int) -> None:
        if fence_on:
            fence.wait(wrapper)
        staging.fill_(value)  # the host writes the schedule ...
        dev.copy_(staging, non_blocking=True)  # ... and enqueues its copy
        fence.record(wrapper)

    torch.cuda._sleep(SLEEP_CYCLES)  # the draft of the previous step
    plan(1)
    seen_by_step_1 = dev.clone()  # what the kernels of step 1 read
    plan(2)
    torch.cuda.synchronize()
    return seen_by_step_1, dev


def test_unfenced_second_plan_corrupts_the_first():
    """The hazard the fence closes (also checks that the test detects it)."""
    seen, final = _race_on_a_staging_buffer(fence_on=False)
    assert (seen == 2).all(), "step 1 should have read the schedule of step 2"
    assert (final == 2).all()


def test_fenced_second_plan_waits_for_the_first_copy():
    seen, final = _race_on_a_staging_buffer(fence_on=True)
    assert (seen == 1).all()
    assert (final == 2).all()


def test_fence_counts_and_does_not_wait_when_the_copy_has_run():
    dev = torch.zeros(16, dtype=torch.int32, device="cuda")
    staging = torch.zeros(16, dtype=torch.int32).pin_memory()
    fence, wrapper = PlanFence(), object()
    fence.wait(wrapper)  # no event yet
    assert fence.waits == 0
    dev.copy_(staging, non_blocking=True)
    fence.record(wrapper)
    torch.cuda.synchronize()
    fence.wait(wrapper)
    assert (fence.waits, fence.blocked) == (1, 0)
    torch.cuda._sleep(SLEEP_CYCLES)
    dev.copy_(staging, non_blocking=True)
    fence.record(wrapper)
    fence.wait(wrapper)  # blocks behind the sleep
    assert (fence.waits, fence.blocked) == (2, 1)
    torch.cuda.synchronize()


def test_fence_is_per_wrapper():
    fence = PlanFence()
    a, b = object(), object()
    torch.cuda._sleep(SLEEP_CYCLES)
    fence.record(a)
    fence.wait(b)  # b has no event: no wait
    assert (fence.waits, fence.blocked) == (0, 0)
    fence.wait(a)
    assert (fence.waits, fence.blocked) == (1, 1)


def test_fence_is_inert_during_graph_capture():
    fence, wrapper = PlanFence(), object()
    x = torch.zeros(8, device="cuda")
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fence.record(wrapper)  # would be a captured event record otherwise
        fence.wait(wrapper)
        x += 1
    assert (fence.waits, fence.blocked) == (0, 0)
    assert id(wrapper) not in fence._events


def _plan_args(q_lens, kv_lens, page_size=16):
    q = torch.tensor(q_lens, dtype=torch.int32)
    kv = torch.tensor(kv_lens, dtype=torch.int32)
    pages = (kv + page_size - 1) // page_size
    qo_indptr = torch.zeros(len(q_lens) + 1, dtype=torch.int32)
    qo_indptr[1:] = q.cumsum(0)
    kv_indptr = torch.zeros(len(q_lens) + 1, dtype=torch.int32)
    kv_indptr[1:] = pages.cumsum(0)
    last = ((kv - 1) % page_size + 1).to(torch.int32)
    indices = torch.arange(int(pages.sum()), dtype=torch.int32, device="cuda")
    # Fresh pinned staging per call, like the builder.
    return dict(
        qo_indptr=qo_indptr.pin_memory(),
        paged_kv_indptr=kv_indptr.pin_memory(),
        paged_kv_indices=indices,
        paged_kv_last_page_len=last.pin_memory(),
        seq_lens=kv.pin_memory(),
    )


def _real_wrapper_race(fence_on: bool):
    """Two plans of different shapes on one real FlashInfer prefill wrapper."""
    flashinfer = pytest.importorskip("flashinfer")
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD", backend="fa2")
    common = dict(
        num_qo_heads=16,
        num_kv_heads=2,
        head_dim_qk=64,
        page_size=16,
        causal=True,
        sm_scale=0.125,
        window_left=-1,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        o_data_type=torch.bfloat16,
    )
    shape_a = ([700, 2900, 1500], [9000, 3100, 1500])
    shape_b = ([5000], [5000])

    def plan(shape):
        wrapper.plan(**_plan_args(*shape), **common)

    int_ws = wrapper._int_workspace_buffer
    # References, each drained after every plan: A alone, and A then B (the workspace
    # keeps the bytes of A beyond the part B rewrites).
    int_ws.zero_()
    torch.cuda.synchronize()
    plan(shape_a)
    torch.cuda.synchronize()
    ref_a = int_ws.clone()
    plan(shape_b)
    torch.cuda.synchronize()
    ref_ab = int_ws.clone()
    assert not torch.equal(ref_a, ref_ab), "the two shapes must plan differently"

    fence = PlanFence()
    int_ws.zero_()
    torch.cuda._sleep(SLEEP_CYCLES)  # the draft of the previous step
    if fence_on:
        fence.wait(wrapper)
    plan(shape_a)
    fence.record(wrapper)
    seen_by_a = int_ws.clone()  # what step A's kernels read
    if fence_on:
        fence.wait(wrapper)
    plan(shape_b)
    fence.record(wrapper)
    torch.cuda.synchronize()
    return seen_by_a, ref_a, int_ws.clone(), ref_ab


def test_real_wrapper_unfenced_second_plan_reaches_the_first_step():
    seen_by_a, ref_a, _, _ = _real_wrapper_race(fence_on=False)
    assert not torch.equal(seen_by_a, ref_a)


def test_real_wrapper_fenced_plans_keep_their_schedules():
    seen_by_a, ref_a, final, ref_ab = _real_wrapper_race(fence_on=True)
    assert torch.equal(seen_by_a, ref_a)
    assert torch.equal(final, ref_ab)
