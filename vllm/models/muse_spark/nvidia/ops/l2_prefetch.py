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
prefetch is throttled to a few CTAs (4 by default, 8 for the 62 MB of the next
attention weights where the collectives are fast, see
``VLLM_MUSE_SPARK_SM120_L2_PREFETCH_AFTER_EXPERTS_CTAS``): issued as a burst (12 to
16 CTAs) it competes with the latency bound collectives for HBM and loses the gain,
but it must also be done before its readers. The launches share one stream, so a
prefetch still running when the next one is due delays that one: with the host
collectives (a layer of 160 us instead of 296 us) the pooler and router weights,
queued behind the attention weights, arrived after their GEMVs in half of the layers
when the launch took 113 us (4 CTAs).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.models.muse_spark.nvidia.ops import muse_c

logger = init_logger(__name__)

# Padded decode batches above this keep the plain path: the projections stop
# being bound by the weight stream.
L2_PREFETCH_MAX_TOKENS = 256
# What one ``_muse_C::l2_prefetch`` launch takes.
_MAX_TENSORS = 4
L2_PREFETCH_MAX_CTAS = 1024
# The most that one launch pulls into L2, as a share of the cache. The weights share
# it with the activations, the KV blocks of the attention in flight, the NCCL buffers
# and the weights of the op that is running, and a prefetch of more than the cache
# holds evicts its own first lines before the GEMMs read them. The model at TP8 takes
# about 63 MB (half of a 128 MB L2) per launch; a smaller TP has larger shards of QKV
# and O and would thrash without a limit.
L2_BUDGET_FRACTION = 0.75
# The L2 size to assume when the driver does not report one.
_FALLBACK_L2_BYTES = 64 << 20

# The values of ``VLLM_MUSE_SPARK_SM120_L2_PREFETCH_ORDER`` (see ``L2PrefetchOrder``).
L2_PREFETCH_ORDERS = (0, 1, 2, 3, 4)


@dataclass(frozen=True, slots=True)
class L2PrefetchOrder:
    """What one value of ``VLLM_MUSE_SPARK_SM120_L2_PREFETCH_ORDER`` changes.

    Order 0 is the shipped one: behind the experts of a layer the unpooler shards,
    then the QKV and O weights of the next layer; behind the QKV projection the
    pooler shards and the router.
    """

    # The QKV weight leads the launch behind the experts (the launch pulls the tensors
    # in list order, so the first one is the first to be warm).
    qkv_first: bool = False
    # No unpooler shards where the communicator runs the unpooler GEMV, which loads
    # its weights by itself.
    skip_fused_unpooler: bool = False
    # The O weight of a layer goes with the launch behind that layer's own QKV
    # projection (in front of the pooler and router) instead of the launch behind
    # the experts of the layer before.
    o_after_qkv: bool = False


_ORDERS = {
    0: L2PrefetchOrder(),
    1: L2PrefetchOrder(qkv_first=True),
    2: L2PrefetchOrder(skip_fused_unpooler=True),
    3: L2PrefetchOrder(o_after_qkv=True),
    4: L2PrefetchOrder(skip_fused_unpooler=True, o_after_qkv=True),
}


def l2_prefetch_order(order: int) -> L2PrefetchOrder:
    """The lists of ``VLLM_MUSE_SPARK_SM120_L2_PREFETCH_ORDER=order``."""
    if order not in _ORDERS:
        raise ValueError(
            "VLLM_MUSE_SPARK_SM120_L2_PREFETCH_ORDER must be one of "
            f"{L2_PREFETCH_ORDERS}, got {order}."
        )
    return _ORDERS[order]


def l2_budget_bytes(device: torch.device | int | None = None) -> int:
    """What one launch may prefetch on ``device``: ``L2_BUDGET_FRACTION`` of its L2."""
    props = torch.cuda.get_device_properties(device)
    l2_bytes = getattr(props, "L2_cache_size", 0) or _FALLBACK_L2_BYTES
    return int(l2_bytes * L2_BUDGET_FRACTION)


class L2Prefetcher:
    """Launches ``_muse_C::l2_prefetch`` on a side stream inside a graph capture."""

    def __init__(
        self,
        num_ctas: int = 4,
        chunk_kib: int = 64,
        evict_last: bool = False,
        budget_bytes: int | None = None,
    ) -> None:
        # Never build inside a forward pass or a capture.
        muse_c.load()
        self.num_ctas = num_ctas
        self.chunk_kib = chunk_kib
        self.evict_last = evict_last
        self.budget_bytes = l2_budget_bytes() if budget_bytes is None else budget_bytes
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

    def select(self, tensors: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """The tensors of ``tensors`` that one launch takes: the usable ones, in the
        order given, as long as they fit the byte budget (a tensor that does not fit
        is skipped, a smaller one after it may still be taken)."""
        chosen: list[torch.Tensor] = []
        total = 0
        for t in tensors:
            if not (
                t.is_cuda
                and t.is_contiguous()
                and t.numel()
                and t.data_ptr() % 16 == 0
            ):
                continue
            nbytes = t.numel() * t.element_size()
            if total + nbytes > self.budget_bytes:
                logger.info_once(
                    "MuseSpark: the L2 prefetch skips weights beyond its budget of %d "
                    "MiB per launch (a %d MiB tensor does not fit after %d MiB).",
                    self.budget_bytes >> 20,
                    nbytes >> 20,
                    total >> 20,
                )
                continue
            chosen.append(t)
            total += nbytes
            if len(chosen) == _MAX_TENSORS:
                break
        return chosen

    def launch(
        self, tensors: Sequence[torch.Tensor], num_ctas: int | None = None
    ) -> None:
        """Prefetch ``tensors`` after the work queued on the current stream, with
        ``num_ctas`` CTAs (default: the prefetcher's)."""
        if not self._active():
            return
        tensors = self.select(tensors)
        if not tensors:
            return
        self._fork.record()
        with torch.cuda.stream(self._stream):
            self._fork.wait()
            torch.ops._muse_C.l2_prefetch(
                tensors, num_ctas or self.num_ctas, self.chunk_kib, self.evict_last
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
