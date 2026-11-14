# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
from collections.abc import Callable

import torch

from vllm import forward_context
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)

_THREAD_ID_TO_CONTEXT: dict = {}
# Here we hardcode the number of microbatches to 2 for default.
_NUM_UBATCHES: int = 2
_CURRENT_CONTEXTS: list["UBatchContext | None"] = []
# Set when one microbatch thread fails, so that the others do not wait for a handoff
# that never comes.
_ABORTED = threading.Event()


class UBatchAborted(RuntimeError):
    """Raised in a microbatch thread that another microbatch's failure woke up."""


class UBatchContext:
    """Context manager for micro-batching synchronization using threading events."""

    def __init__(
        self,
        id: int,
        comm_stream: torch.cuda.Stream,
        compute_stream: torch.cuda.Stream,
        forward_context: ForwardContext,
        ready_barrier: threading.Barrier,
        cpu_wait_event: threading.Event,
        cpu_signal_event: threading.Event,
        gpu_comm_done_event: torch.Event,
        gpu_compute_done_event: torch.Event,
        schedule: str = "default",
        token_offset: int = 0,
        overlap_collectives: bool = False,
    ):
        self.id = id
        self.comm_stream = comm_stream
        self.compute_stream = compute_stream
        self.forward_context = forward_context
        self.ready_barrier = ready_barrier
        self.cpu_wait_event = cpu_wait_event
        self.cpu_signal_event = cpu_signal_event
        self.current_stream = compute_stream
        self.gpu_comm_done_event = gpu_comm_done_event
        self.gpu_compute_done_event = gpu_compute_done_event
        self.schedule = schedule
        self.token_offset = token_offset
        # Run the tensor-parallel collectives of this microbatch on the comm stream
        # (see run_collective); the expert all-to-all of DeepEP does its own handoff.
        self.overlap_collectives = overlap_collectives
        # Set while a collective runs on the comm stream: the compute stream the
        # communicator should allocate its result on (see dbo_collective_output_stream).
        self.collective_output_stream: torch.cuda.Stream | None = None
        self.recv_hook = None

    def __enter__(self):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _THREAD_ID_TO_CONTEXT[threading.get_ident()] = self.id
        _CURRENT_CONTEXTS[self.id] = self
        # _NUM_UBATCHES is set in make_ubatch_contexts
        self.ready_barrier.wait()

        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._check_not_aborted()
        self._restore_context()
        # Assume we want to start on the compute stream
        self.update_stream(self.compute_stream)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CURRENT_CONTEXTS, _THREAD_ID_TO_CONTEXT
        _CURRENT_CONTEXTS[self.id] = None
        del _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        self.maybe_run_recv_hook()
        self.cpu_signal_event.set()
        self.cpu_wait_event.clear()
        return False

    def _restore_context(self):
        forward_context._forward_context = self.forward_context

    def update_stream(self, stream):
        self.current_stream = stream
        if current_stream() != self.current_stream:
            torch.cuda.set_stream(self.current_stream)

    def _signal_comm_done(self):
        self.gpu_comm_done_event.record(self.comm_stream)

    def _signal_compute_done(self):
        self.gpu_compute_done_event.record(self.compute_stream)

    def _wait_compute_done(self):
        self.comm_stream.wait_event(self.gpu_compute_done_event)

    def _wait_comm_done(self):
        self.compute_stream.wait_event(self.gpu_comm_done_event)

    def _cpu_yield(self):
        # It is critical for correctness that only one thread is running
        # at a time. These asserts just make sure that this is the only
        # thread running before waking the other one up and going to sleep
        assert forward_context._forward_context == self.forward_context
        assert current_stream() == self.current_stream
        assert not self.cpu_wait_event.is_set()

        self.cpu_signal_event.set()
        self.cpu_wait_event.wait()
        self.cpu_wait_event.clear()
        self._check_not_aborted()
        self._restore_context()

    @staticmethod
    def _check_not_aborted():
        if _ABORTED.is_set():
            raise UBatchAborted("Microbatch aborted: another microbatch failed")

    def switch_to_comm(self):
        self.update_stream(self.comm_stream)

    def switch_to_compute(self):
        self.update_stream(self.compute_stream)

    def switch_to_comm_sync(self):
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def switch_to_compute_sync(self):
        self._signal_comm_done()
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    def maybe_run_recv_hook(self):
        if self.recv_hook is not None:
            self.recv_hook()
            self.recv_hook = None

    def yield_(self):
        self.current_stream = current_stream()
        self._cpu_yield()
        self.update_stream(self.current_stream)

    def yield_and_switch_from_compute_to_comm(self):
        assert current_stream() == self.compute_stream
        self._signal_compute_done()
        self._cpu_yield()
        assert self.current_stream == self.compute_stream
        self.update_stream(self.comm_stream)
        self._wait_compute_done()

    def yield_and_switch_from_comm_to_compute(self):
        assert current_stream() == self.comm_stream
        self._signal_comm_done()
        self._cpu_yield()
        assert self.current_stream == self.comm_stream
        self.update_stream(self.compute_stream)
        self._wait_comm_done()

    def run_collective(
        self, collective: Callable[[torch.Tensor], torch.Tensor], input_: torch.Tensor
    ) -> torch.Tensor:
        """Run ``collective(input_)`` on the comm stream, beside the sibling's compute.

        The collective is enqueued on the comm stream as soon as the compute stream
        has produced its input. The thread then hands off, so the other microbatch
        enqueues its compute behind what this one has queued (and its own collective
        behind this one), and this microbatch continues once the collective is
        done. Every collective of a step goes on the one comm stream in the order
        the threads hand off, which is the same on every rank.

        The caching allocator frees a block for reuse once the stream that
        allocated it is done with it, so a result allocated on the comm stream while
        the compute stream reads it is recorded on the compute stream. The input
        needs no such record: it is allocated on the compute stream, and that
        stream waits for the collective before any later kernel that could reuse
        its block (a record would hold the blocks of every collective the CPU has
        enqueued ahead of the GPU, hundreds of MiB, until the GPU gets to them).
        """
        compute_stream = self.compute_stream
        assert current_stream() == compute_stream
        self._signal_compute_done()
        self.update_stream(self.comm_stream)
        self._wait_compute_done()
        self.collective_output_stream = compute_stream
        try:
            output = collective(input_)
        finally:
            self.collective_output_stream = None
        output.record_stream(compute_stream)
        self._signal_comm_done()
        self.update_stream(compute_stream)
        self._cpu_yield()
        self._wait_comm_done()
        return output


def dbo_abort() -> None:
    """Wake the parked microbatch threads so that they unwind with an error."""
    _ABORTED.set()
    for ctx in _CURRENT_CONTEXTS:
        if ctx is not None:
            ctx.cpu_wait_event.set()


def dbo_enabled() -> bool:
    return len(_THREAD_ID_TO_CONTEXT) > 0


def dbo_overlapping_context() -> "UBatchContext | None":
    """The calling thread's microbatch context if it overlaps its TP collectives."""
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return None
    ctx_idx = _THREAD_ID_TO_CONTEXT.get(threading.get_ident())
    if ctx_idx is None:
        return None
    ctx = _CURRENT_CONTEXTS[ctx_idx]
    if ctx is None or not ctx.overlap_collectives:
        return None
    # An aux-stream branch of the model stays where it is.
    return ctx if current_stream() == ctx.compute_stream else None


def dbo_collective_output_stream() -> "torch.cuda.Stream | None":
    """Where a collective that runs on the comm stream should allocate its result.

    The compute stream, inside ``UBatchContext.run_collective``, else None. A block
    belongs to the pool of the stream that allocated it, so a result allocated on
    the comm stream sits in a pool of its own that the compute stream cannot reuse
    (and that holds its blocks until the compute stream is done with them): the
    two pools together outgrow a GPU that is nearly full, and the allocator then
    frees and re-allocates them over and over. A result allocated on the compute
    stream and written by the comm stream needs neither: the compute stream only
    reads it after the collective, and reuses it only after those reads.
    """
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return None
    ctx_idx = _THREAD_ID_TO_CONTEXT.get(threading.get_ident())
    if ctx_idx is None:
        return None
    ctx = _CURRENT_CONTEXTS[ctx_idx]
    return None if ctx is None else ctx.collective_output_stream


def dbo_current_ubatch_id() -> int:
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    return _THREAD_ID_TO_CONTEXT[threading.get_ident()]


def dbo_current_ubatch_token_offset() -> int:
    """Row offset of the calling ubatch inside the parent batch.

    Returns 0 when no DBO context is active on this thread.
    """
    if len(_THREAD_ID_TO_CONTEXT) == 0:
        return 0
    ctx = _CURRENT_CONTEXTS[_THREAD_ID_TO_CONTEXT[threading.get_ident()]]
    return ctx.token_offset


def _register_ubatch_function(func):
    def wrapper(*args, **kwargs):
        if len(_THREAD_ID_TO_CONTEXT) > 0:
            ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
            ctx = _CURRENT_CONTEXTS[ctx_idx]
            func(ctx, *args, **kwargs)

    return wrapper


dbo_maybe_run_recv_hook = _register_ubatch_function(UBatchContext.maybe_run_recv_hook)
dbo_yield = _register_ubatch_function(UBatchContext.yield_)
dbo_yield_and_switch_from_compute_to_comm = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_compute_to_comm
)
dbo_yield_and_switch_from_comm_to_compute = _register_ubatch_function(
    UBatchContext.yield_and_switch_from_comm_to_compute
)
dbo_switch_to_comm = _register_ubatch_function(UBatchContext.switch_to_comm)
dbo_switch_to_compute = _register_ubatch_function(UBatchContext.switch_to_compute)
dbo_switch_to_comm_sync = _register_ubatch_function(UBatchContext.switch_to_comm_sync)
dbo_switch_to_compute_sync = _register_ubatch_function(
    UBatchContext.switch_to_compute_sync
)


def dbo_register_recv_hook(recv_hook):
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        next_ctx = _CURRENT_CONTEXTS[(ctx_idx + 1) % _NUM_UBATCHES]
        next_ctx.recv_hook = recv_hook


def dbo_get_previous_event(func, *args, **kwargs):
    if len(_THREAD_ID_TO_CONTEXT) > 0:
        ctx_idx = _THREAD_ID_TO_CONTEXT[threading.get_ident()]
        ctx = _CURRENT_CONTEXTS[ctx_idx]
        # execute callable on the ubatch compute stream to record/wait events there
        with torch.cuda.stream(ctx.compute_stream):
            return func(*args, **kwargs)


def make_ubatch_contexts(
    num_micro_batches: int,
    compute_stream: torch.cuda.Stream,
    comm_stream: torch.cuda.Stream,
    forward_contexts: list[ForwardContext],
    ready_barrier: threading.Barrier,
    schedule: str = "default",
    token_offsets: list[int] | None = None,
    overlap_collectives: bool = False,
) -> list[UBatchContext]:
    global _NUM_UBATCHES, _CURRENT_CONTEXTS
    assert num_micro_batches > 1, "num_micro_batches must be greater than 1"
    _ABORTED.clear()

    _NUM_UBATCHES = num_micro_batches
    # Ensure the global context list is large enough
    if len(_CURRENT_CONTEXTS) < num_micro_batches:
        _CURRENT_CONTEXTS.extend([None] * (num_micro_batches - len(_CURRENT_CONTEXTS)))

    """
    Create a context manager for micro-batching synchronization.
    """
    cpu_events = [threading.Event() for _ in range(num_micro_batches)]
    gpu_comm_done_events = [torch.Event() for _ in range(num_micro_batches)]
    gpu_compute_done_events = [torch.Event() for _ in range(num_micro_batches)]

    ctxs = []
    for i in range(num_micro_batches):
        ctx = UBatchContext(
            id=i,
            compute_stream=compute_stream,
            comm_stream=comm_stream,
            forward_context=forward_contexts[i],
            ready_barrier=ready_barrier,
            cpu_wait_event=cpu_events[i],
            cpu_signal_event=cpu_events[(i + 1) % num_micro_batches],
            gpu_comm_done_event=gpu_comm_done_events[i],
            gpu_compute_done_event=gpu_compute_done_events[i],
            schedule=schedule,
            token_offset=token_offsets[i] if token_offsets else 0,
            overlap_collectives=overlap_collectives,
        )
        ctxs.append(ctx)

    return ctxs
