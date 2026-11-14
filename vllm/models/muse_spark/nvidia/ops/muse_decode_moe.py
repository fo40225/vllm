# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The NVFP4 (W4A4) decode MoE kernel of ``_muse_C`` for a few tokens on SM120.

``_muse_C::nvfp4_decode_moe`` runs routing, the activation quantization, both
expert GEMMs and the SiLU * up requantization of a padded batch of up to 8
tokens in one persistent launch. It reads the packed weights and the swizzled
block scales of a layer in place (no copy) and writes the unweighted GEMM2 rows
route-major, plus the permute map the post-GEMM2 hook needs: ``perm_idx[r]`` is
``r`` for a route whose expert is local to this rank and -1 otherwise (the rows
of the latter are not written).

Every shape is static: the launch parameters depend on the padded token count
only (1, 2, 4 or 8; a batch of 3 runs as 4), the routing is data on the device.
The output buffers and the scratch are static tensors shared by all layers,
which run one after another on one stream; the rows stay valid until the next
call. The kernel is deterministic (fixed reduction order, no float atomics), so
the engine is also valid in batch-invariant mode.

The engine is optional: ``muse_unusable_reason`` and ``muse_shape_mismatch`` say
whether the host and a layer can use it, and callers that cannot stay on their
other engine.
"""

from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.models.muse_spark.common.ops import muse_pdl_enabled
from vllm.models.muse_spark.nvidia.ops import muse_c

logger = init_logger(__name__)

# Padded token counts 1..MUSE_MAX_TOKENS are served (as 1, 2, 4 or 8).
MUSE_MAX_TOKENS = 8
_T_PADS = (1, 2, 4, 8)
# The fixed problem of the kernel, apart from the number of local experts.
MUSE_HIDDEN = 4096
MUSE_INTERMEDIATE = 4096
MUSE_TOP_K = 8
# Flags of the op, see muse_c_ops.h.
_FLAG_ZERO_NON_LOCAL = 4
_FLAG_NO_PDL = 8

# Set once an engine could not be set up: the other layers would fail alike.
_setup_failure: str | None = None


def muse_unusable_reason() -> str | None:
    """Why the _muse_C extension cannot be built here, or None if it can."""
    return _setup_failure or muse_c.preflight()


def mark_muse_setup_failed(reason: str) -> None:
    global _setup_failure
    _setup_failure = reason


def muse_shape_mismatch(
    hidden: int, intermediate: int, top_k: int, num_local_experts: int
) -> str | None:
    """Why a layer of these sizes is outside the kernel's problem, or None.

    ``intermediate`` is the size per rank (one half of w13).
    """
    if (hidden, intermediate, top_k) != (MUSE_HIDDEN, MUSE_INTERMEDIATE, MUSE_TOP_K):
        return (
            f"the kernel is built for hidden={MUSE_HIDDEN}, "
            f"intermediate={MUSE_INTERMEDIATE} and top-k {MUSE_TOP_K} per rank; "
            f"the layer has hidden={hidden}, intermediate={intermediate}, "
            f"top-k {top_k}"
        )
    if num_local_experts < 1:
        return "the rank holds no experts"
    return None


@dataclass
class _Scratch:
    """Static scratch of one device, shared by every layer and shape."""

    hq_data: torch.Tensor
    hq_sf: torch.Tensor
    plan: torch.Tensor
    # Counters of the persistent kernel: zero once, every launch leaves them zero.
    ctl: torch.Tensor


@dataclass
class _Outputs:
    """Static outputs of one padded token count, shared by every layer."""

    rows: torch.Tensor
    perm_idx: torch.Tensor


_scratch: dict[torch.device, _Scratch] = {}
_outputs: dict[tuple[torch.device, int], _Outputs] = {}


def _get_scratch(device: torch.device) -> _Scratch:
    scratch = _scratch.get(device)
    if scratch is None:
        half = MUSE_INTERMEDIATE // 2
        scratch = _scratch[device] = _Scratch(
            hq_data=torch.zeros(64, half, dtype=torch.uint8, device=device),
            hq_sf=torch.zeros(64, half // 8, dtype=torch.uint8, device=device),
            plan=torch.zeros(1024, dtype=torch.int32, device=device),
            ctl=torch.zeros(128, dtype=torch.int32, device=device),
        )
    return scratch


def _get_outputs(device: torch.device, t_pad: int) -> _Outputs:
    key = (device, t_pad)
    outputs = _outputs.get(key)
    if outputs is None:
        outputs = _outputs[key] = _Outputs(
            rows=torch.zeros(
                t_pad * MUSE_TOP_K, MUSE_HIDDEN, dtype=torch.bfloat16, device=device
            ),
            perm_idx=torch.full(
                (t_pad * MUSE_TOP_K,), -1, dtype=torch.int32, device=device
            ),
        )
    return outputs


def _padded_tokens(num_tokens: int) -> int:
    for t_pad in _T_PADS:
        if num_tokens <= t_pad:
            return t_pad
    raise RuntimeError(
        f"The muse decode engine serves 1..{MUSE_MAX_TOKENS} tokens, got {num_tokens}."
    )


class MuseDecodeMoE:
    """The muse decode engine of one layer, over the weights the CUTLASS path reads.

    ``w1`` [E, 2n, K / 2] is stored ``[gate; up]`` (the checkpoint's order) unless
    ``up_first``; ``w1_scale`` / ``w2_scale`` are the swizzled block scales. ``prm``
    [E, 4] and ``gs`` [2] are the kernel's scales (``decode_kernel_params``). The
    engine reads the weights where they are (see ``reads``) and never copies
    them; a caller whose weights move builds a new one. With ``zero_non_local``
    the rows of non-local routes are zeroed as well, for consumers that reduce
    the rows across ranks before they look at the permute map.
    """

    def __init__(
        self,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        *,
        prm: torch.Tensor,
        gs: torch.Tensor,
        up_first: bool = False,
        zero_non_local: bool = False,
    ) -> None:
        if w2.dim() != 3:
            raise ValueError(f"w2 must be [E, K, N / 2], got {tuple(w2.shape)}")
        num_local, hidden, packed = w2.shape
        mismatch = muse_shape_mismatch(hidden, packed * 2, MUSE_TOP_K, num_local)
        if mismatch is not None:
            raise ValueError(mismatch)
        self.num_local_experts = num_local
        self.device = w1.device
        self._w1, self._w1_scale = w1, w1_scale
        self._w2, self._w2_scale = w2, w2_scale
        self._prm = prm.float().contiguous()
        self._gs = gs.float().contiguous()
        self._up_first = up_first
        self._flags = _FLAG_ZERO_NON_LOCAL if zero_non_local else 0
        self._sources = self._pointers(w1, w1_scale, w2, w2_scale)
        # The first launch of each shape sets up its kernel (the shared memory
        # attribute), which must not happen during a CUDA graph capture.
        muse_c.load()
        self._scratch = _get_scratch(self.device)
        for t_pad in _T_PADS:
            _get_outputs(self.device, t_pad)
        self._warm_up()

    @staticmethod
    def _pointers(*tensors: torch.Tensor) -> tuple[int, ...]:
        return tuple(t.data_ptr() for t in tensors)

    def reads(
        self,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> bool:
        """Whether the engine was prepared on exactly these tensors."""
        return self._sources == self._pointers(w1, w1_scale, w2, w2_scale)

    def _warm_up(self) -> None:
        """Launch every padded shape once, with and without an expert map."""
        expert_map = torch.arange(
            self.num_local_experts, dtype=torch.int32, device=self.device
        )
        for t_pad in _T_PADS:
            hidden = torch.zeros(
                t_pad, MUSE_HIDDEN, dtype=torch.bfloat16, device=self.device
            )
            ids = (
                torch.arange(t_pad * MUSE_TOP_K, dtype=torch.int32, device=self.device)
                % self.num_local_experts
            ).view(t_pad, MUSE_TOP_K)
            for ids_map in (None, expert_map):
                self.run(hidden, ids, ids_map)
        torch.accelerator.synchronize()
        if bool(self._scratch.ctl.any()):
            raise RuntimeError("The muse decode kernel left its counters dirty.")

    def run(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rows, perm_idx)`` of ``hidden_states`` [T, hidden]; T is 1 to 8.

        The rows [T * 8, hidden] are the unweighted GEMM2 output, route-major:
        row ``t * 8 + k`` belongs to route ``(t, k)``, and ``perm_idx`` [T * 8]
        holds that index, or -1 if the expert of the route is not local (the row
        is then stale, or zero with ``zero_non_local``). ``topk_ids`` [T, 8] are
        global expert ids and ``expert_map`` is the global-to-local map (None:
        all are local). Both results belong to the shared buffers and stay valid
        until the next call.
        """
        num_tokens = hidden_states.shape[0]
        t_pad = _padded_tokens(num_tokens)
        outputs = _get_outputs(self.device, t_pad)
        scratch = self._scratch
        torch.ops._muse_C.nvfp4_decode_moe(
            hidden_states.contiguous(),
            topk_ids.to(torch.int32).contiguous(),
            None
            if expert_map is None
            else expert_map.to(torch.int32).contiguous(),
            self._w1,
            self._w1_scale,
            self._w2,
            self._w2_scale,
            self._prm,
            self._gs,
            self._up_first,
            outputs.rows,
            outputs.perm_idx,
            scratch.hq_data,
            scratch.hq_sf,
            scratch.plan,
            scratch.ctl,
            None,
            # Launched as a programmatic dependent of the kernel before it in the
            # stream (the routing kernel, or the pre-expert norm when the routing
            # runs on the aux stream and joins by an event). It waits for it before
            # it reads the ids and the activations. Plain launch if PDL is off.
            self._flags | (0 if muse_pdl_enabled() else _FLAG_NO_PDL),
            None,
            None,
        )
        num_routes = num_tokens * MUSE_TOP_K
        return outputs.rows[:num_routes], outputs.perm_idx[:num_routes]
