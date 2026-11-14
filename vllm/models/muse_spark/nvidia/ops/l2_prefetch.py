# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Side-stream L2 prefetch of the weights of the next decode ops (SM120).

A decode step is bound by HBM in the QKV and O projections (bf16 weights, tens
of MB per layer and rank, read once per token batch) and, to a smaller degree, in
the pooler, router and unpooler GEMVs, while HBM idles inside the collectives and
during the straggler wait of the MoE all-reduce. This forks a graph branch that
pulls those weights into the 128 MB L2 (``_muse_C::l2_prefetch``, asynchronous
bulk prefetches issued by a few CTAs) ahead of their use, so the GEMMs read from
L2 instead of HBM. It only changes where the bytes come from: no tensor is
written and no kernel output changes.

The branch exists only inside a CUDA graph capture. Its join sits at the end of
the model forward: the GEMMs need no ordering after the prefetch (it is a hint),
and a join in front of them would stall the rank that finishes its MoE last. The
prefetch is deliberately slow (4 CTAs, about 100 us for the 55 MB of the
attention weights): issued as a burst (8 to 16 CTAs) it competes with the latency
bound NCCL kernels for HBM and loses a part of the gain (16 CTAs: half of it).
"""

from collections.abc import Sequence

import torch

from vllm.logger import init_logger
from vllm.models.muse_spark.nvidia.ops import muse_c

logger = init_logger(__name__)

# Padded decode batches above this keep the plain path: the projections stop
# being bound by the weight stream.
L2_PREFETCH_MAX_TOKENS = 256
# What one ``_muse_C::l2_prefetch`` launch takes.
_MAX_TENSORS = 4


class L2Prefetcher:
    """Launches ``_muse_C::l2_prefetch`` on a side stream inside a graph capture."""

    def __init__(
        self, num_ctas: int = 4, chunk_kib: int = 64, evict_last: bool = False
    ) -> None:
        # Never build inside a forward pass or a capture.
        muse_c.load()
        self.num_ctas = num_ctas
        self.chunk_kib = chunk_kib
        self.evict_last = evict_last
        self._stream = torch.cuda.Stream()
        self._fork = torch.cuda.Event()
        self._join = torch.cuda.Event()
        self._forked = False

    @staticmethod
    def _active() -> bool:
        if not torch.cuda.is_current_stream_capturing():
            return False
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        return not BreakableCUDAGraphCapture.is_active()

    def launch(self, tensors: Sequence[torch.Tensor]) -> None:
        """Prefetch ``tensors`` after the work queued on the current stream."""
        if not self._active():
            return
        tensors = [
            t
            for t in tensors
            if t.is_cuda and t.is_contiguous() and t.numel() and t.data_ptr() % 16 == 0
        ][:_MAX_TENSORS]
        if not tensors:
            return
        self._fork.record()
        with torch.cuda.stream(self._stream):
            self._fork.wait()
            torch.ops._muse_C.l2_prefetch(
                tensors, self.num_ctas, self.chunk_kib, self.evict_last
            )
        self._forked = True
        logger.info_once("MuseSpark: L2 weight prefetch captured into a CUDA graph.")

    def join(self) -> None:
        """Rejoin the side stream; a capture must end with it joined."""
        if not self._forked:
            return
        self._forked = False
        self._join.record(self._stream)
        self._join.wait()
