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

Batches of 9 to 128 tokens (the CUDA graph sizes of larger decode batches and the
verification batches of speculative decoding) run ``nvfp4_decode_moe_large``: a
small preparation launch quantizes the activations into global scratch and builds
the plan, then the same persistent kernel reads the activations of every stage
from that scratch instead of keeping them in shared memory. The rows are the
same bit for bit as those of the small-batch kernel for every route. Against the
CUTLASS grouped GEMMs (which the batches above 8 tokens used before) the
quantization points, scales and roundings are the same, and the rows are equal bit
for bit in the unit probes (random layers and real expert weights with gaussian,
per-channel scaled and outlier activations); the fp32 accumulation order over K
differs (the kernel adds four partial sums, CUTLASS runs one chain), and end to end
on the real model the two engines differ at the rounding level, reproducibly, as
they already do for 8 tokens or fewer. The cause has not been isolated.

Every shape is static: the launch parameters depend on the padded token count
only (1, 2, 4 or 8; a batch of 3 runs as 4; the large-batch kernel takes the
count as is), the routing is data on the device. The output buffers and the
scratch are static tensors shared by all layers, which run one after another on
one stream; the rows stay valid until the next call. The kernel is deterministic
(fixed reduction order, no float atomics), so the engine is also valid in
batch-invariant mode.

The engine is optional: ``muse_unusable_reason`` and ``muse_shape_mismatch`` say
whether the host and a layer can use it, and callers that cannot stay on their
other engine.
"""

from dataclasses import dataclass

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.models.muse_spark.common.ops import muse_pdl_enabled
from vllm.models.muse_spark.nvidia.ops import muse_c

logger = init_logger(__name__)

# Token counts 1..MUSE_MAX_TOKENS are served: up to MUSE_SMALL_MAX_TOKENS by the
# small-batch kernel (padded to 1, 2, 4 or 8), the rest by the large-batch one.
MUSE_MAX_TOKENS = 128
MUSE_SMALL_MAX_TOKENS = 8
_T_PADS = (1, 2, 4, 8)
# The expert prefetch maps the routes of up to this many tokens (64 routes) in one CTA.
EXPERT_PREFETCH_MAX_TOKENS = 8
# The large-batch kernel keeps its plan in 384 entries (an expert with up to 8
# routes is one entry): 256 local experts plus the experts split over entries.
MUSE_LARGE_MAX_LOCAL_EXPERTS = 256
# The fixed problem of the kernel, apart from the number of local experts.
MUSE_HIDDEN = 4096
MUSE_INTERMEDIATE = 4096
MUSE_TOP_K = 8
# Flags of the ops, see muse_c_ops.h.
_FLAG_ZERO_NON_LOCAL = 4
_FLAG_NO_PDL = 8
_FLAG_LARGE_BULK_LOADER = 16  # only the large-batch op reads it
_FLAG_LARGE_EXTERNAL_PLAN = 32  # the plan was built by build_plan (large-batch op)
_FLAG_LARGE_EXTERNAL_XQ = 64  # the quantized activations are in the scratch (idem)
# Scratch of the large-batch kernel: one row per route of the largest batch.
_MAX_ROUTES = MUSE_MAX_TOKENS * MUSE_TOP_K

# Set once an engine could not be set up: the other layers would fail alike.
_setup_failure: str | None = None


def muse_max_tokens(num_local_experts: int | None = None) -> int:
    """The largest token count the muse engine serves.

    ``VLLM_MUSE_SPARK_SM120_DECODE_MAX_TOKENS`` lowers it (the CUTLASS grouped
    GEMMs serve the larger batches), but not below the small-batch kernel's 8
    tokens: a smaller value is raised to 8 with a warning (the engine is turned off
    with ``VLLM_MUSE_SPARK_SM120_DECODE_ENGINE=cutlass``). A rank with more local
    experts than the large-batch kernel's plan holds is limited to the small-batch
    kernel.
    """
    requested = envs.VLLM_MUSE_SPARK_SM120_DECODE_MAX_TOKENS
    if requested < MUSE_SMALL_MAX_TOKENS:
        logger.warning_once(
            "VLLM_MUSE_SPARK_SM120_DECODE_MAX_TOKENS=%d is below the %d tokens of "
            "the muse small-batch kernel; using %d. To turn the muse decode engine "
            "off set VLLM_MUSE_SPARK_SM120_DECODE_ENGINE=cutlass.",
            requested,
            MUSE_SMALL_MAX_TOKENS,
            MUSE_SMALL_MAX_TOKENS,
        )
    limit = min(MUSE_MAX_TOKENS, max(requested, MUSE_SMALL_MAX_TOKENS))
    if (
        num_local_experts is not None
        and num_local_experts > MUSE_LARGE_MAX_LOCAL_EXPERTS
    ):
        limit = min(limit, MUSE_SMALL_MAX_TOKENS)
    return limit


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
    # Large batches: the quantized activations (codes, linear scales).
    xq_data: torch.Tensor
    xq_sf: torch.Tensor


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
            hq_data=torch.zeros(_MAX_ROUTES, half, dtype=torch.uint8, device=device),
            hq_sf=torch.zeros(_MAX_ROUTES, half // 8, dtype=torch.uint8, device=device),
            plan=torch.zeros(8192, dtype=torch.int32, device=device),
            ctl=torch.zeros(512, dtype=torch.int32, device=device),
            xq_data=torch.zeros(
                MUSE_MAX_TOKENS, half, dtype=torch.uint8, device=device
            ),
            xq_sf=torch.zeros(
                MUSE_MAX_TOKENS, half // 8, dtype=torch.uint8, device=device
            ),
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


def _get_large_outputs(device: torch.device) -> _Outputs:
    """The outputs of the large-batch kernel: a batch uses their leading rows."""
    key = (device, 0)
    outputs = _outputs.get(key)
    if outputs is None:
        outputs = _outputs[key] = _Outputs(
            rows=torch.zeros(
                _MAX_ROUTES, MUSE_HIDDEN, dtype=torch.bfloat16, device=device
            ),
            perm_idx=torch.full((_MAX_ROUTES,), -1, dtype=torch.int32, device=device),
        )
    return outputs


def _padded_tokens(num_tokens: int) -> int:
    for t_pad in _T_PADS:
        if num_tokens <= t_pad:
            return t_pad
    raise RuntimeError(
        f"The small-batch muse decode kernel serves 1..{MUSE_SMALL_MAX_TOKENS} "
        f"tokens, got {num_tokens}."
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
        self.max_tokens = muse_max_tokens(num_local)
        self.device = w1.device
        self._w1, self._w1_scale = w1, w1_scale
        self._w2, self._w2_scale = w2, w2_scale
        self._prm = prm.float().contiguous()
        self._gs = gs.float().contiguous()
        self._up_first = up_first
        self._flags = _FLAG_ZERO_NON_LOCAL if zero_non_local else 0
        bulk_loader = envs.VLLM_MUSE_SPARK_SM120_LARGE_BULK_LOADER
        self._large_flags = _FLAG_LARGE_BULK_LOADER if bulk_loader else 0
        self._sources = self._pointers(w1, w1_scale, w2, w2_scale)
        # The ids a ``build_plan`` queued the plan of, and the activations that were
        # quantized into the scratch (``quantize_input``, or an external producer that
        # calls ``mark_input_quantized``), until the next ``run`` takes them.
        self._plan_key: tuple | None = None
        self._xq_key: tuple | None = None
        # The first launch of each shape sets up its kernel (the shared memory
        # attribute), which must not happen during a CUDA graph capture.
        muse_c.load()
        self._scratch = _get_scratch(self.device)
        for t_pad in _T_PADS:
            _get_outputs(self.device, t_pad)
        if self.max_tokens > MUSE_SMALL_MAX_TOKENS:
            _get_large_outputs(self.device)
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
        norm_weight = torch.ones(MUSE_HIDDEN, dtype=torch.bfloat16, device=self.device)
        warm_up = list(_T_PADS)
        if self.max_tokens > MUSE_SMALL_MAX_TOKENS:
            # One launch sets up the large-batch kernel for every token count.
            warm_up.append(MUSE_SMALL_MAX_TOKENS + 1)
        for t_pad in warm_up:
            hidden = torch.zeros(
                t_pad, MUSE_HIDDEN, dtype=torch.bfloat16, device=self.device
            )
            ids = (
                torch.arange(t_pad * MUSE_TOP_K, dtype=torch.int32, device=self.device)
                % self.num_local_experts
            ).view(t_pad, MUSE_TOP_K)
            for ids_map in (None, expert_map):
                self.run(hidden, ids, ids_map)
                if t_pad > MUSE_SMALL_MAX_TOKENS:
                    self.build_plan(ids, ids_map)
                    self.run(hidden, ids, ids_map)
                    # The launches that quantize ahead of the large-batch kernel.
                    self.quantize_input(hidden)
                    self.run(hidden, ids, ids_map)
                    self.rmsnorm_quantize(hidden, norm_weight, 1e-6, 0.0)
                    self.discard_input()
        torch.accelerator.synchronize()
        if bool(self._scratch.ctl.any()):
            raise RuntimeError("The muse decode kernel left its counters dirty.")

    def prefetch(
        self,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
        *,
        num_ctas: int,
        budget_bytes: int,
        pdl: bool,
        dump: torch.Tensor | None = None,
    ) -> None:
        """Queue an L2 prefetch of the experts that ``run`` will read for ``topk_ids``.

        The same ids and map as ``run``. The kernel maps the routes to this rank's
        distinct local experts and prefetches their weights and scales in the order
        the decode kernel consumes them (GEMM1 of every expert, then GEMM2), up to
        ``budget_bytes``. It reads the ids and writes nothing, so it can run on a
        side stream behind the routing kernel. With ``pdl`` it is launched as a
        programmatic dependent of the kernel before it in the stream. ``dump`` (int64,
        tests only) receives the (address, size) of the requests in place of the
        prefetches.
        """
        torch.ops._muse_C.expert_prefetch(
            topk_ids.to(torch.int32).contiguous(),
            None if expert_map is None else expert_map.to(torch.int32).contiguous(),
            self._w1,
            self._w1_scale,
            self._w2,
            self._w2_scale,
            self._up_first,
            num_ctas,
            budget_bytes,
            pdl,
            dump,
        )

    @staticmethod
    def _plan_key_of(topk_ids: torch.Tensor, expert_map: torch.Tensor | None) -> tuple:
        return (
            topk_ids.data_ptr(),
            tuple(topk_ids.shape),
            None if expert_map is None else expert_map.data_ptr(),
        )

    def build_plan(
        self, topk_ids: torch.Tensor, expert_map: torch.Tensor | None
    ) -> None:
        """Queue the plan of the large-batch kernel (9 to ``max_tokens`` tokens) for ``topk_ids``.

        The plan and the route -> row map depend on the ids and the expert map alone,
        so the launch can run on another stream than ``run`` (the router's aux stream,
        right behind the top-k): ``run`` then finds it prepared for the same ids and
        skips building it. The stream of ``run`` must be ordered after this launch
        (an event, or the join of the aux stream in a CUDA graph capture), and ``run``
        for these ids must be the next call: the plan lives in the shared scratch.
        """
        num_tokens = topk_ids.shape[0]
        if not MUSE_SMALL_MAX_TOKENS < num_tokens <= self.max_tokens:
            raise RuntimeError(
                f"The large-batch plan serves {MUSE_SMALL_MAX_TOKENS + 1}.."
                f"{self.max_tokens} tokens, got {num_tokens}."
            )
        outputs = _get_large_outputs(self.device)
        torch.ops._muse_C.nvfp4_decode_moe_plan(
            topk_ids.to(torch.int32).contiguous(),
            None if expert_map is None else expert_map.to(torch.int32).contiguous(),
            self.num_local_experts,
            outputs.perm_idx[: num_tokens * MUSE_TOP_K],
            self._scratch.plan,
            0 if muse_pdl_enabled() else _FLAG_NO_PDL,
        )
        self._plan_key = self._plan_key_of(topk_ids, expert_map)

    def discard_plan(self) -> None:
        """Forget a plan ``build_plan`` queued that no ``run`` took."""
        self._plan_key = None

    @property
    def input_scale(self) -> torch.Tensor:
        """The kernel's ``gs`` [2]; ``gs[0]`` is the global scale of the input
        quantization (what ``quantize_input`` and an external producer must use)."""
        return self._gs

    def input_scratch(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``(codes [T, 2048] u8, scales [T, 256] u8)`` scratch of the large-batch
        kernel's quantized input, for a producer that quantizes the activations itself
        (``mark_input_quantized`` then tells ``run``)."""
        if not MUSE_SMALL_MAX_TOKENS < num_tokens <= self.max_tokens:
            raise RuntimeError(
                f"The quantized input of the large-batch kernel serves "
                f"{MUSE_SMALL_MAX_TOKENS + 1}..{self.max_tokens} tokens, got "
                f"{num_tokens}."
            )
        return self._scratch.xq_data[:num_tokens], self._scratch.xq_sf[:num_tokens]

    def quantize_input(self, hidden_states: torch.Tensor) -> None:
        """Quantize ``hidden_states`` [T, 4096] bf16 (9 to ``max_tokens`` tokens) into the
        scratch ahead of ``run``, which then skips its own quantization for these very
        activations. The bytes are those of the preparation launch of ``run``."""
        data, sf = self.input_scratch(hidden_states.shape[0])
        torch.ops._muse_C.nvfp4_decode_act_quant(
            hidden_states.contiguous(), self._gs, data, sf, muse_pdl_enabled()
        )
        self.mark_input_quantized(hidden_states)

    def rmsnorm_quantize(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
        gain_center: float,
    ) -> torch.Tensor:
        """The pre-expert RMSNorm of ``x`` [T, 4096] and the quantization of its output, in
        one launch (9 to ``max_tokens`` tokens).

        Returns the normalized rows, bit for bit what the Triton RMSNorm of the model
        returns (weight after the normalization, ``f32(weight) + gain_center``); the NVFP4
        codes and scales of those rows are written to the scratch, and ``run`` on these
        very rows skips its own quantization.
        """
        data, sf = self.input_scratch(x.shape[0])
        out = torch.empty_like(x)
        torch.ops._muse_C.rmsnorm_nvfp4_quant(
            x, weight, eps, gain_center, self._gs, out, data, sf, muse_pdl_enabled()
        )
        self.mark_input_quantized(out)
        return out

    def mark_input_quantized(self, hidden_states: torch.Tensor) -> None:
        """Say that ``input_scratch`` holds the quantization of ``hidden_states`` with
        ``input_scale[0]`` (written by the caller's own kernel, ordered before ``run``)."""
        self._xq_key = (hidden_states.data_ptr(), tuple(hidden_states.shape))

    def discard_input(self) -> None:
        self._xq_key = None

    def run(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_map: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(rows, perm_idx)`` of ``hidden_states`` [T, hidden], T = 1..``max_tokens``.

        The rows [T * 8, hidden] are the unweighted GEMM2 output, route-major:
        row ``t * 8 + k`` belongs to route ``(t, k)``, and ``perm_idx`` [T * 8]
        holds that index, or -1 if the expert of the route is not local (the row
        is then stale, or zero with ``zero_non_local``). ``topk_ids`` [T, 8] are
        global expert ids and ``expert_map`` is the global-to-local map (None:
        all are local). Both results belong to the shared buffers and stay valid
        until the next call.
        """
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_tokens:
            raise RuntimeError(
                f"The muse decode engine serves 1..{self.max_tokens} tokens, "
                f"got {num_tokens}."
            )
        # Launched as a programmatic dependent of the kernel before it in the
        # stream (the routing kernel, or the pre-expert norm when the routing
        # runs on the aux stream and joins by an event). It waits for it before
        # it reads the ids and the activations. Plain launch if PDL is off.
        flags = self._flags | (0 if muse_pdl_enabled() else _FLAG_NO_PDL)
        scratch = self._scratch
        num_routes = num_tokens * MUSE_TOP_K
        # A plan queued by ``build_plan`` is good for this call only, and only for the
        # very ids and map it was built for.
        plan_key, self._plan_key = self._plan_key, None
        xq_key, self._xq_key = self._xq_key, None
        if num_tokens > MUSE_SMALL_MAX_TOKENS:
            capturing = torch.cuda.is_current_stream_capturing()
            if xq_key is not None:
                if xq_key == (hidden_states.data_ptr(), tuple(hidden_states.shape)):
                    flags |= _FLAG_LARGE_EXTERNAL_XQ
                    if capturing:
                        logger.info_once(
                            "MuseSpark MoE: the large-batch muse kernel takes its "
                            "quantized input from the producer of the pre-expert norm "
                            "output (the norm + quantization launch, or the TP "
                            "communicator's gather + norm epilogue; "
                            "VLLM_MUSE_SPARK_SM120_FUSED_NORM_QUANT=1)."
                        )
                else:
                    logger.warning_once(
                        "MuseSpark MoE: the activations quantized ahead of the "
                        "large-batch muse kernel were other ones than it ran on; the "
                        "kernel quantizes its own."
                    )
            if plan_key is not None:
                if plan_key == self._plan_key_of(topk_ids, expert_map):
                    flags |= _FLAG_LARGE_EXTERNAL_PLAN
                    if capturing:
                        logger.info_once(
                            "MuseSpark MoE: the plan of the large-batch muse kernel "
                            "is built on the aux stream behind the top-k "
                            "(VLLM_MUSE_SPARK_SM120_AUX_PLAN=1)."
                        )
                else:
                    logger.warning_once(
                        "MuseSpark MoE: the plan built on the aux stream was for other "
                        "ids than the experts ran on; the kernel builds its own."
                    )
            outputs = _get_large_outputs(self.device)
            rows = outputs.rows[:num_routes]
            perm_idx = outputs.perm_idx[:num_routes]
            torch.ops._muse_C.nvfp4_decode_moe_large(
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
                rows,
                perm_idx,
                scratch.hq_data,
                scratch.hq_sf,
                scratch.xq_data,
                scratch.xq_sf,
                scratch.plan,
                scratch.ctl,
                flags | self._large_flags,
            )
            return rows, perm_idx
        t_pad = _padded_tokens(num_tokens)
        outputs = _get_outputs(self.device, t_pad)
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
            flags,
            None,
            None,
        )
        return outputs.rows[:num_routes], outputs.perm_idx[:num_routes]
