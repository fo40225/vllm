# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests of the microbatch hand-off protocol of vllm/v1/worker/ubatching.py.

The microbatch threads alternate at every collective of the model. They run with
fake streams and events here, so the protocol is tested without a GPU: which thread
runs when, and that the step ends whatever number of collectives each thread issues.
"""

import threading

import pytest
import torch

import vllm.v1.worker.ubatching as ub


class _FakeStream:
    def wait_event(self, event) -> None:
        pass


class _FakeEvent:
    def record(self, stream=None) -> None:
        pass


class _Result:
    def record_stream(self, stream) -> None:
        pass


@pytest.fixture(autouse=True)
def fake_streams(monkeypatch):
    stream_of_thread = threading.local()
    monkeypatch.setattr(
        ub, "current_stream", lambda: getattr(stream_of_thread, "stream", None)
    )
    monkeypatch.setattr(
        torch.cuda, "set_stream", lambda s: setattr(stream_of_thread, "stream", s)
    )
    monkeypatch.setattr(torch, "Event", _FakeEvent)
    yield stream_of_thread
    # No test may leave a registration behind.
    ub._THREAD_ID_TO_CONTEXT.clear()
    ub._CURRENT_CONTEXTS[:] = []
    ub._ABORTED.clear()


def _run_step(
    fake_streams,
    collectives: list[int],
    fail_at: dict[int, int] | None = None,
    timeout: float = 10.0,
):
    """Run one thread per entry of ``collectives``, each issuing that many collectives.

    ``fail_at`` maps a thread to the collective it raises before.
    Returns (finished, errors, trace, hung): the threads that completed, the
    exception of each thread that raised, the order of the collectives as
    (thread, index) and whether a thread was still running after ``timeout``.
    """
    fail_at = fail_at or {}
    n = len(collectives)
    compute, comm = _FakeStream(), _FakeStream()
    barrier = threading.Barrier(n + 1)
    contexts = ub.make_ubatch_contexts(
        n, compute, comm, [object() for _ in range(n)], barrier, overlap_collectives=True
    )
    finished: list[int] = []
    errors: dict[int, BaseException] = {}
    trace: list[tuple[int, int]] = []

    def body(ctx: ub.UBatchContext, count: int) -> None:
        fake_streams.stream = compute
        try:
            with ctx:
                for k in range(count):
                    if fail_at.get(ctx.id) == k:
                        raise ValueError("boom")
                    ctx.run_collective(
                        lambda t, k=k: (trace.append((ctx.id, k)), _Result())[1],
                        object(),
                    )
            finished.append(ctx.id)
        except BaseException as e:  # noqa: BLE001
            errors[ctx.id] = e

    threads = [
        threading.Thread(target=body, args=(c, k), daemon=True)
        for c, k in zip(contexts, collectives)
    ]
    for t in threads:
        t.start()
    barrier.wait()
    contexts[0].cpu_wait_event.set()
    for t in threads:
        t.join(timeout)
    hung = any(t.is_alive() for t in threads)
    if hung:
        # Release the stuck threads so that the test process can go on.
        ub.dbo_abort()
        for t in threads:
            t.join(2.0)
    return finished, errors, trace, hung


@pytest.mark.parametrize(
    "collectives",
    [
        [3, 3],
        [4, 3],
        [3, 4],
        [5, 3],
        [3, 5],
        [0, 0],
        [0, 3],
        [3, 0],
        [2, 5, 3],
        [5, 2, 3],
        [3, 3, 3, 3],
        [1, 6, 0, 2],
    ],
)
def test_step_ends_whatever_the_collective_counts_are(fake_streams, collectives):
    """The microbatches of a step may issue different numbers of collectives.

    A model chooses its path from the token count of the microbatch it runs, so one
    half of a split batch can issue more collectives than the other. The thread
    left alone must run on instead of waiting for a handoff nobody can make.
    """
    finished, errors, trace, hung = _run_step(fake_streams, collectives)

    assert not hung, f"a microbatch parked forever with collectives={collectives}"
    assert not errors
    assert sorted(finished) == list(range(len(collectives)))
    assert sorted(trace) == sorted(
        (i, k) for i, count in enumerate(collectives) for k in range(count)
    )
    assert not ub.dbo_enabled()
    assert all(ctx is None for ctx in ub._CURRENT_CONTEXTS)


def test_equal_counts_alternate_in_ring_order(fake_streams):
    """With equal counts the handoff is the one of the stock protocol."""
    _, _, trace, hung = _run_step(fake_streams, [2, 2, 2])

    assert not hung
    assert trace == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]


def test_the_longer_microbatch_runs_the_rest_alone(fake_streams):
    _, _, trace, hung = _run_step(fake_streams, [1, 3])

    assert not hung
    assert trace == [(0, 0), (1, 0), (1, 1), (1, 2)]


@pytest.mark.parametrize("fail_at", [0, 1])
def test_failing_microbatch_unwinds_its_sibling_and_leaves_no_registration(
    fake_streams, fail_at
):
    """A failure before the first handoff (the sibling is still starting up) and
    after one (the sibling is parked at a handoff) both end the step cleanly."""
    finished, errors, _, hung = _run_step(
        fake_streams, [3, 3], fail_at={0: fail_at}, timeout=10.0
    )

    assert not hung
    assert not finished
    assert isinstance(errors[0], ValueError)
    assert isinstance(errors[1], ub.UBatchAborted)
    # The next step must not see the dead one: no thread registered, nothing in
    # the context list (dbo_enabled() and dbo_current_ubatch_id() read them).
    assert not ub.dbo_enabled()
    assert all(ctx is None for ctx in ub._CURRENT_CONTEXTS)
    assert ub.dbo_current_ubatch_id() == 0


def test_registry_of_a_failed_step_does_not_leak_into_the_next(fake_streams):
    _run_step(fake_streams, [3, 3], fail_at={0: 0})
    finished, errors, trace, hung = _run_step(fake_streams, [2, 2])

    assert not hung
    assert not errors
    assert sorted(finished) == [0, 1]
    assert trace == [(0, 0), (1, 0), (0, 1), (1, 1)]
