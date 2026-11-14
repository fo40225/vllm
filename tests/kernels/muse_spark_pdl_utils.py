# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A slow producer for the programmatic dependent launch (PDL) tests of Muse Spark.

A kernel launched as a programmatic dependent may start before its predecessor has
finished, so it must wait (``gdc_wait``) before it reads anything a kernel of the
step writes. ``slow_copy`` is the predecessor of such a test: it lets its
dependents launch at once and writes every output only after a long spin, over a
poison value, so a dependent that reads one of them before its wait sees the poison.
"""

import contextlib
import os
import sys
import time

import torch

from vllm.triton_utils import tl, triton

# About 300 us of dependent FMAs on an SM120: much longer than the host needs to
# enqueue the next launch, so that an eager launch overlaps the producer too.
SPIN = 100_000
BLOCK = 1024
MAX_PAIRS = 3
# Fewer programs than SMs: all of them are resident at once, so that the dependents
# launch while every program still spins (the trigger waits for the last program).
MAX_PROGRAMS = 96


@triton.jit
def _copy_chunk(dst_ptr, src_ptr, n, chunk, acc, BLOCK: tl.constexpr):
    offs = chunk * BLOCK + tl.arange(0, BLOCK)
    # acc stays below 1e3, so the second mask term is always true; it keeps the
    # spin from being removed.
    mask = (offs < n) & (acc < 1e30)
    tl.store(dst_ptr + offs, tl.load(src_ptr + offs, mask=mask), mask=mask)


@triton.jit
def _slow_copy_kernel(
    dst0,
    src0,
    n0,
    chunks0,
    dst1,
    src1,
    n1,
    chunks1,
    dst2,
    src2,
    n2,
    total_chunks,
    SPIN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # Let the dependents of this kernel launch right now ...
    tl.extra.cuda.gdc_launch_dependents()
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for _ in range(SPIN):
        acc = acc * 0.999 + 1.0
    # ... and write all the data only after the spin.
    for chunk in range(tl.program_id(0), total_chunks, tl.num_programs(0)):
        if chunk < chunks0:
            _copy_chunk(dst0, src0, n0, chunk, acc, BLOCK)
        elif chunk < chunks0 + chunks1:
            _copy_chunk(dst1, src1, n1, chunk - chunks0, acc, BLOCK)
        else:
            _copy_chunk(dst2, src2, n2, chunk - chunks0 - chunks1, acc, BLOCK)


def slow_copy(*pairs: tuple[torch.Tensor, torch.Tensor]) -> None:
    """``dst.copy_(src)`` for every ``(dst, src)`` pair (at most three), in ONE
    kernel that finishes late but triggers its dependents early.

    One kernel, so that the writes of all the pairs land after the same spin and a
    dependent can race with each of them: a chain of slow kernels would finish one
    producer before the next one starts.
    """
    assert 1 <= len(pairs) <= MAX_PAIRS
    args: list = []
    total_chunks = 0
    for dst, src in pairs:
        assert dst.is_contiguous() and src.is_contiguous() and dst.dtype == src.dtype
        chunks = triton.cdiv(src.numel(), BLOCK)
        total_chunks += chunks
        args += [dst.view(-1), src.view(-1), src.numel(), chunks]
    while len(args) < 4 * MAX_PAIRS:  # unused slots: a pair that has no chunk
        args += [args[0], args[1], 0, 0]
    args[-1] = total_chunks  # the last slot has no chunk count of its own
    grid = (min(total_chunks, MAX_PROGRAMS),)
    _slow_copy_kernel[grid](*args, SPIN=SPIN, BLOCK=BLOCK)


def synchronize_or_exit(what: str, capture=None, timeout_s: float = 90.0) -> None:
    """``torch.cuda.synchronize()`` that gives up on a hung kernel.

    A launch that skips a wait it needs can hang for good (a spin on a counter
    that never completes), and the test would then hold the shared GPU forever.
    The hung kernel cannot be cancelled and would stall every later launch, so the
    process exits (status 3) instead of failing the single test. ``capture`` is the
    ``capfd`` fixture of the calling test: without it pytest's output capture would
    swallow the message of the exiting process.
    """
    event = torch.cuda.Event()
    event.record()
    deadline = time.monotonic() + timeout_s
    while not event.query():
        if time.monotonic() > deadline:
            with capture.disabled() if capture else contextlib.nullcontext():
                print(
                    f"\nFATAL: {what} did not finish within {timeout_s:.0f} s; "
                    "a kernel is hung (a missing programmatic launch wait?). "
                    "Exiting the test process.",
                    file=sys.stderr,
                    flush=True,
                )
            os._exit(3)
        time.sleep(0.001)
    torch.cuda.synchronize()
