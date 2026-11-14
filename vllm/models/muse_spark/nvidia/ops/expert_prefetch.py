# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routing-aware L2 prefetch of the experts of a decode layer (SM120).

The muse decode kernel streams the NVFP4 weights of the experts that the routing
picked for this rank (27 MB each) from HBM, and a launch spends about 5 us (plan,
first DRAM latency) before its first weights arrive. The routing kernel runs on the
router's aux stream beside the pooler chain, and its ids exist about when the kernel
starts. This queues ``_muse_C::expert_prefetch`` on that stream behind the routing
kernel: it pulls the head of the experts' weights into the 128 MB L2 in the order the
kernel will consume them, so the first tiles of the kernel hit L2. It only moves cache
lines (no tensor is written, no result changes).

How much it can win depends on its lead over the kernel's first loads: with the L2
cold and the HBM idle, a prefetch that starts L microseconds ahead takes about L - 1.2
microseconds off the kernel, whatever the number of experts (a stand-alone replay of
the kernel on a flushed L2, graph, real shapes; 27 MB of one expert in L2 take it from
25.9 to 15.5 us, three experts from 60 to 27 us). In a decode layer the routing chain
(router GEMV, reduce, top-k) is about as long as the pooler chain it runs beside, so
without help the ids come about 1 us before the kernel starts and the prefetch has no
lead. The boundary kernel writes the router's FP32 input when the prefetch is on
(``wants_cast_router_input``), which removes the cast kernel from the routing chain
and moves the ids 3 us earlier. And the prefetch must be small: it competes with the
kernel's own loads and with the weight prefetches of ``l2_prefetch.py`` for HBM, and
one that pulls every expert of the rank (96 MiB) made the step slower; the head of the
first expert (``EXPERT_PREFETCH_BYTES``) takes 3 us off the kernel of a rank with two
or more experts, the ones that the all-reduce waits for.

The launch stays on the aux stream, after the event that hands the routing to the
main stream, so the experts never wait for it. The next layer's router chain on
that stream is queued behind it, and the prefetch ends long before then. A capture
must end with the stream joined; the next layer's join covers it, and the model
calls ``join`` at the end of its forward in case no layer follows.

Like the weight prefetch of ``l2_prefetch.py`` it exists only inside a full CUDA
graph capture.
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# What one prefetch pulls into L2, in the order the kernel consumes it: the head of the
# first expert. That is about what HBM moves between the end of the routing and the
# kernel's first loads (a few microseconds at 1.5 TB/s, see the module docstring);
# anything beyond it only competes with the kernel's own loads: 96 MiB (every expert
# of the rank, up to 3/4 of the L2) made the decode step 1.2% slower than none (10.01
# against 9.88 ms), 6 to 16 MiB are all alike (9.73, 9.75, 9.73, 9.73 ms for 6, 8, 12,
# 16 MiB; TPOT at batch 1 on 8 RTX PRO 6000 with real weights and the 8 CTA launch of
# ``l2_prefetch.py``). The model reads VLLM_MUSE_SPARK_SM120_EXPERT_PREFETCH_BYTES,
# whose default is this value (envs.py cannot import it; a test keeps the two equal).
EXPERT_PREFETCH_BYTES = 12 << 20
# CTAs of the prefetch, one thread each issues the requests: 8 and 16 are alike, a CTA
# per SM (188) is slower.
EXPERT_PREFETCH_CTAS = 16


class ExpertPrefetch:
    """Queues the expert prefetch of the decode layers on the router's aux stream."""

    def __init__(
        self, stream: torch.cuda.Stream, budget_bytes: int = EXPERT_PREFETCH_BYTES
    ) -> None:
        self.stream = stream
        self.budget_bytes = budget_bytes
        self._join = torch.cuda.Event()
        self._pending = False

    @staticmethod
    def _active() -> bool:
        if not torch.cuda.is_current_stream_capturing():
            return False
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        return not BreakableCUDAGraphCapture.is_active()

    def launch(self, runner, topk_ids: torch.Tensor) -> None:
        """Prefetch the experts that ``runner`` will read for ``topk_ids``.

        Call it right after the routing joined the main stream (the routing kernel
        is then the last kernel on the aux stream).
        """
        if not self._active():
            return
        with torch.cuda.stream(self.stream):
            queued = runner.prefetch_routed_experts(topk_ids, self.budget_bytes)
        if queued:
            self._pending = True
            logger.info_once(
                "MuseSpark: expert L2 prefetch captured into a CUDA graph."
            )

    def join(self) -> None:
        """Rejoin the aux stream; a capture must end with it joined."""
        if not self._pending:
            return
        self._pending = False
        self._join.record(self.stream)
        self._join.wait()
