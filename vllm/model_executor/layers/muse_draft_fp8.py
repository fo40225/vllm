# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 (e4m3) weights for the Muse Spark DFlash2 draft on SM120.

At decode the draft's GEMMs are weight-streaming bound (1.2-1.4 TB/s of the 1.54 TB/s
DRAM ceiling at 8 rows), so e4m3 weights halve their time. The weights are quantized
online, at load, from the bf16 checkpoint: one fp32 scale per output channel (per rank
shard: a row-parallel layer scales the partial product of its own K shard, which the
all-reduce then sums). Two kernels run them (VLLM_MUSE_SPARK_SM120_DRAFT_FP8_KERNEL):

* ``marlin``: weight-only FP8 (W8A16, vLLM's Marlin FP8): bf16 activations, the weight
  is repacked to the Marlin layout; the lowest error (no activation quantization).
* ``cutlass``: W8A8: dynamic per-token e4m3 activations (``scaled_fp8_quant``) and
  ``cutlass_scaled_mm`` with per-token x per-channel scales; also uses the FP8 tensor
  cores, so it keeps its gain above 128 rows where the draft turns compute bound.

Both are plain device ops of a fixed schedule (no atomics: run-to-run deterministic)
and capture into CUDA graphs.
"""

from types import SimpleNamespace

import torch
from torch import nn

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.utils import replace_parameter

logger = init_logger(__name__)

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = 448.0
# Floor of the channel scale (an all-zero row): the one vLLM's online FP8 uses.
_MIN_SCALE = 1.0 / (FP8_MAX * 512)
# Rows are quantized in chunks so the fp32 temporaries stay near 64 MB.
_QUANT_CHUNK_ELEMS = 16 * 1024 * 1024
# CUTLASS needs N % 16 == 0 (vLLM otherwise falls back to a Triton kernel).
_CUTLASS_ROW_ALIGN = 16

KERNELS = ("marlin", "cutlass")


def quantize_rows(
    weight: torch.Tensor, pad_rows_to: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] bf16/fp16 -> ([N', K] e4m3, [N'] fp32 scale) with one scale per row.

    ``weight ~= wq * scale[:, None]``; N' is N rounded up to ``pad_rows_to`` (the extra
    rows are zero with scale 1).
    """
    n, k = weight.shape
    n_padded = -(-n // pad_rows_to) * pad_rows_to
    wq = torch.zeros(n_padded, k, dtype=FP8_DTYPE, device=weight.device)
    scale = torch.ones(n_padded, dtype=torch.float32, device=weight.device)
    step = max(1, _QUANT_CHUNK_ELEMS // k)
    for start in range(0, n, step):
        end = min(start + step, n)
        rows = weight[start:end].float()
        row_scale = (rows.abs().amax(dim=1) / FP8_MAX).clamp_min(_MIN_SCALE)
        wq[start:end] = (
            (rows / row_scale[:, None]).clamp_(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
        )
        scale[start:end] = row_scale
    return wq, scale


def cutlass_fp8_linear(
    x: torch.Tensor, wq: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """x [..., K] bf16 @ (wq [N', K] e4m3 * scale [N'])^T -> [..., N'] bf16 (W8A8)."""
    x2 = x.reshape(-1, x.shape[-1])
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    xq, x_scale = ops.scaled_fp8_quant(x2, None, None, use_per_token_if_dynamic=True)
    out = ops.cutlass_scaled_mm(
        xq, wq.t(), scale_a=x_scale, scale_b=scale.view(-1, 1), out_dtype=x.dtype
    )
    return out.view(*x.shape[:-1], wq.shape[0])


class DraftFp8LinearMethod(LinearMethodBase):
    """Installed on a draft linear layer after its bf16 weight is quantized in place."""

    supports_pre_processed_weights = True
    requires_device_loading = False

    def __init__(self, kernel: str, out_features: int, in_features: int) -> None:
        if kernel not in KERNELS:
            raise ValueError(
                f"VLLM_MUSE_SPARK_SM120_DRAFT_FP8_KERNEL={kernel!r}: expected one of "
                f"{KERNELS}."
            )
        self.kernel = kernel
        self.out_features = out_features
        self.in_features = in_features

    def create_weights(self, layer: nn.Module, *args, **kwargs) -> None:
        raise NotImplementedError(
            "DraftFp8LinearMethod replaces the method of a loaded layer"
        )

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        return  # the weight is already quantized

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("the FP8 draft layers have no bias")
        if self.kernel == "marlin":
            from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
                apply_fp8_marlin_linear,
            )

            return apply_fp8_marlin_linear(
                input=x,
                weight=layer.weight,
                weight_scale=layer.weight_scale,
                workspace=None,
                size_n=self.out_features,
                size_k=self.in_features,
                bias=None,
            )
        return cutlass_fp8_linear(x, layer.weight, layer.weight_scale)


def quantize_draft_linear(layer: LinearBase, kernel: str) -> int:
    """Quantize a loaded bf16 draft linear layer in place; returns the bytes freed.

    Layers that are not plain unquantized bf16/fp16 (a quantized draft checkpoint) are
    left alone and count as 0.
    """
    if not isinstance(layer.quant_method, UnquantizedLinearMethod):
        return 0
    weight = layer.weight.data
    if weight.dtype not in (torch.bfloat16, torch.float16) or weight.ndim != 2:
        return 0
    if getattr(layer, "bias", None) is not None:
        return 0  # the FP8 GEMMs take no bias (the draft has none)
    before = weight.numel() * weight.element_size()
    out_features, in_features = weight.shape
    wq, scale = quantize_rows(weight)
    if kernel == "marlin":
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp8 import (
            prepare_fp8_layer_for_marlin,
        )

        # prepare_fp8_layer_for_marlin takes the weight as [K, N] and rewrites
        # weight / weight_scale (the scale as bf16, permuted) in place.
        layer.orig_dtype = weight.dtype
        replace_parameter(layer, "weight", wq.t().contiguous())
        layer.register_parameter(
            "weight_scale", nn.Parameter(scale.view(1, -1), requires_grad=False)
        )
        # A replicated layer has no per-partition sizes; the Marlin repack reads them.
        layer.output_size_per_partition = out_features
        layer.input_size_per_partition = in_features
        prepare_fp8_layer_for_marlin(layer, size_k_first=True)
    else:
        replace_parameter(layer, "weight", wq)
        layer.register_parameter(
            "weight_scale", nn.Parameter(scale, requires_grad=False)
        )
    layer.quant_method = DraftFp8LinearMethod(kernel, out_features, in_features)
    after = sum(
        p.numel() * p.element_size() for p in (layer.weight, layer.weight_scale)
    )
    return before - after


class _HeadMethod:
    """The ``quant_method`` of ``DraftFp8Head`` as ``LogitsProcessor._apply_head`` calls it."""

    def apply(
        self, head: "DraftFp8Head", x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("the FP8 draft head has no bias")
        return cutlass_fp8_linear(x, head.weight_q, head.weight_scale)


class DraftFp8Head(nn.Module):
    """An e4m3 copy of a vocab-parallel lm_head shard for the draft's candidate top-k.

    The draft shares the target's bf16 lm_head (which the target needs as it is), so the
    FP8 head is an extra copy (half the bytes of the shard) that the draft reads instead.
    It stands in for the head in ``LogitsProcessor.get_top_k_tokens``: the rows are padded
    to a multiple of 16 for CUTLASS and the logits come back as wide as the padded
    shard, the padding counted in ``shard_indices.num_org_vocab_padding`` so that the
    top-k masks those columns to -inf as it masks the vocabulary padding.
    """

    def __init__(self, lm_head: nn.Module) -> None:
        super().__init__()
        weight = lm_head.weight.data
        wq, scale = quantize_rows(weight, _CUTLASS_ROW_ALIGN)
        self.register_buffer("weight_q", wq)
        self.register_buffer("weight_scale", scale)
        self.quant_method = _HeadMethod()
        self.tp_size = lm_head.tp_size
        shard = lm_head.shard_indices
        self.shard_indices = SimpleNamespace(
            org_vocab_start_index=shard.org_vocab_start_index,
            num_org_vocab_padding=shard.num_org_vocab_padding
            + (wq.shape[0] - weight.shape[0]),
        )

    @property
    def nbytes(self) -> int:
        return self.weight_q.numel() + self.weight_scale.numel() * 4
