# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 weights of the DFlash2 draft (VLLM_MUSE_SPARK_SM120_DRAFT_FP8).

CPU tests: the per-channel quantizer, the model hook's gating (flag values, what is and
is not quantized). GPU tests, on the real per-rank shapes of the TP8 draft (hidden 8192,
intermediate 28672, 64 / 8 heads of 128, vocabulary 202048): the error of each kernel
against an fp32 reference stays inside a bound next to the bf16 GEMM's own, capture and
replay of a CUDA graph give the eager bits, a repeated call gives the same bits, the FP8
candidate head returns the bf16 head's top-k (up to ties) and never a padded column.
"""

import glob
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm import envs
from vllm.distributed import parallel_state
from vllm.model_executor.layers.linear import (
    ReplicatedLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.muse_draft_fp8 import (
    FP8_DTYPE,
    DraftFp8Head,
    DraftFp8LinearMethod,
    cutlass_fp8_linear,
    quantize_draft_linear,
    quantize_rows,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
)
from vllm.model_executor.models import qwen3_dflash2
from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM

FLAG = "VLLM_MUSE_SPARK_SM120_DRAFT_FP8"
KERNEL_FLAG = "VLLM_MUSE_SPARK_SM120_DRAFT_FP8_KERNEL"
KERNELS = ["cutlass", "marlin"]

# (output rows, input columns) of the draft's GEMMs on one rank of TP8.
SHAPES = {
    "gate_up": (7168, 8192),
    "down": (8192, 3584),
    "qkv": (1280, 8192),
    "o": (8192, 1024),
    "kernel_projection": (2048, 8192),
}
ROWS = [8, 16, 64, 128, 256, 512]

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def one_rank(monkeypatch):
    """Linear layers read the TP group when they are built: a group of one rank."""
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=1)
    )


def realistic_weight(n: int, k: int, seed: int, dtype=torch.bfloat16, device="cpu"):
    """Gaussian weights with a spread of per-channel magnitudes (a trained matrix's rows
    differ in scale, which is what the per-channel scales are for)."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(n, k, generator=generator) * 0.02
    weight *= torch.exp(0.3 * torch.randn(n, 1, generator=generator))
    return weight.to(dtype).to(device)


def linear_with(weight: torch.Tensor) -> ReplicatedLinear:
    n, k = weight.shape
    layer = ReplicatedLinear(
        k, n, bias=False, params_dtype=weight.dtype, return_bias=False
    )
    layer.weight.data = weight.clone()
    return layer


def errors(y: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(max error / max |ref|, rms error / rms ref)."""
    diff = y.float() - ref.float()
    return (
        (diff.abs().max() / ref.abs().max()).item(),
        (diff.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()).item(),
    )


# --------------------------------------------------------------------------------
# The quantizer (CPU).
# --------------------------------------------------------------------------------


def test_quantize_rows_per_channel_scale_and_bound():
    weight = realistic_weight(300, 512, seed=1)
    wq, scale = quantize_rows(weight)
    assert wq.dtype == FP8_DTYPE and wq.shape == weight.shape
    assert scale.dtype == torch.float32 and scale.shape == (300,)
    # Every row uses the full e4m3 range: its largest element maps to +-448.
    row_max = wq.float().abs().amax(dim=1)
    assert torch.all(row_max == 448.0)
    assert torch.allclose(scale, weight.float().abs().amax(dim=1) / 448.0)
    # e4m3 has 3 mantissa bits: half a step is at most 1/16 of the value (above the
    # subnormal range), plus the subnormal spacing of the smallest elements.
    dequant = wq.float() * scale[:, None]
    abs_err = (dequant - weight.float()).abs()
    bound = weight.float().abs() / 16 + scale[:, None] * 2**-10
    assert torch.all(abs_err <= bound * 1.0001)
    rel_rms = abs_err.pow(2).mean().sqrt() / weight.float().pow(2).mean().sqrt()
    assert rel_rms < 0.04


def test_quantize_rows_pads_and_keeps_zero_rows_finite():
    weight = realistic_weight(25, 64, seed=2)
    weight[3] = 0
    wq, scale = quantize_rows(weight, pad_rows_to=16)
    assert wq.shape == (32, 64) and scale.shape == (32,)
    assert torch.all(wq[25:].float() == 0) and torch.all(scale[25:] == 1.0)
    assert torch.all(wq[3].float() == 0)
    assert torch.isfinite(scale).all() and scale[3] > 0
    assert torch.equal(wq[:25], quantize_rows(weight)[0])


def test_quantize_rows_chunks_agree(monkeypatch):
    import vllm.model_executor.layers.muse_draft_fp8 as module

    weight = realistic_weight(97, 128, seed=3)
    whole = quantize_rows(weight)
    monkeypatch.setattr(module, "_QUANT_CHUNK_ELEMS", 128 * 10)
    chunked = quantize_rows(weight)
    assert torch.equal(whole[0], chunked[0]) and torch.equal(whole[1], chunked[1])


# --------------------------------------------------------------------------------
# The model hook (CPU).
# --------------------------------------------------------------------------------


def fake_draft(num_layers: int = 2, k: int = 64, n: int = 32):
    """The attributes ``process_weights_after_loading`` walks, with tiny real layers."""

    def lin(rows, cols, seed):
        return linear_with(realistic_weight(rows, cols, seed))

    layers = []
    for i in range(num_layers):
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    qkv_proj=lin(n, k, 10 * i), o_proj=lin(k, n, 10 * i + 1)
                ),
                mlp=SimpleNamespace(
                    gate_up_proj=lin(2 * n, k, 10 * i + 2),
                    down_proj=lin(k, n, 10 * i + 3),
                ),
                attention_conv=SimpleNamespace(kernel_projection=lin(n, k, 10 * i + 4)),
                mlp_conv=SimpleNamespace(kernel_projection=lin(n, k, 10 * i + 5)),
            )
        )
    fc = lin(k, 5 * k, 99)
    fake = SimpleNamespace(model=SimpleNamespace(layers=layers, fc=fc))
    # What ``load_weights`` builds before the hook runs (the bf16 fused context K/V).
    fake.model._fused_kv_weight = torch.zeros(4, k, dtype=torch.bfloat16)
    fake.fp8_head = None
    fake.candidate_logits_processor = SimpleNamespace(head_dtype=None)
    # The hook asks for its layer list through the real method.
    fake._fp8_linears = lambda: DFlash2Qwen3ForCausalLM._fp8_linears(fake)
    return fake


class LogRecorder:
    """Stands in for the module logger: keeps the formatted lines per level."""

    def __init__(self, monkeypatch):
        self.lines = {"info": [], "warning": []}
        for level in self.lines:
            monkeypatch.setattr(
                qwen3_dflash2.logger, level, self._recorder(level), raising=False
            )

    def _recorder(self, level):
        def record(msg, *args, **kwargs):
            self.lines[level].append(msg % args)

        return record


def draft_linears(fake):
    for layer in fake.model.layers:
        yield layer.self_attn.qkv_proj
        yield layer.self_attn.o_proj
        yield layer.mlp.gate_up_proj
        yield layer.mlp.down_proj
        yield layer.attention_conv.kernel_projection
        yield layer.mlp_conv.kernel_projection


def test_the_flag_defaults_to_the_head_copy_and_cutlass(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(KERNEL_FLAG, raising=False)
    assert envs.VLLM_MUSE_SPARK_SM120_DRAFT_FP8 == 2
    assert envs.VLLM_MUSE_SPARK_SM120_DRAFT_FP8_KERNEL == "cutlass"


def test_hook_is_a_noop_when_the_flag_is_off(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    fake = fake_draft()
    before = [lin.weight.data.clone() for lin in draft_linears(fake)]
    DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    for lin, w in zip(draft_linears(fake), before, strict=True):
        assert isinstance(lin.quant_method, UnquantizedLinearMethod)
        assert torch.equal(lin.weight.data, w)


@pytest.mark.parametrize("flag", ["-1", "3"])
def test_hook_rejects_a_bad_flag(monkeypatch, flag):
    monkeypatch.setenv(FLAG, flag)
    with pytest.raises(ValueError, match="expected 0, 1 or 2"):
        DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake_draft())


@pytest.mark.parametrize("flag", ["1", "2"])
@pytest.mark.parametrize("kernel", KERNELS)
def test_hook_quantizes_the_draft_linears_and_leaves_the_fc(monkeypatch, flag, kernel):
    if kernel == "marlin" and not torch.cuda.is_available():
        pytest.skip("the Marlin repack needs a GPU")
    monkeypatch.setenv(FLAG, flag)
    monkeypatch.setenv(KERNEL_FLAG, kernel)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fake = fake_draft()
    for lin in draft_linears(fake):
        lin.to(device)
    fc_weight = fake.model.fc.weight.data.clone()
    fused_kv = torch.cat(
        [layer.self_attn.qkv_proj.weight.data[:8] for layer in fake.model.layers]
    )
    DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    for lin in draft_linears(fake):
        assert isinstance(lin.quant_method, DraftFp8LinearMethod)
        assert lin.quant_method.kernel == kernel
        assert lin.weight.dtype in (FP8_DTYPE, torch.int32)
        assert lin.quant_method.out_features == lin.output_size
        assert lin.quant_method.in_features == lin.input_size
    # The fc keeps its bf16 weight and method: FC_SHARD=2 slices it by rows.
    assert isinstance(fake.model.fc.quant_method, UnquantizedLinearMethod)
    assert torch.equal(fake.model.fc.weight.data.cpu(), fc_weight.cpu())
    assert fused_kv.dtype == torch.bfloat16


def test_hook_skips_layers_that_are_not_plain_bf16(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(KERNEL_FLAG, "cutlass")
    log = LogRecorder(monkeypatch)
    fake = fake_draft(num_layers=1)
    quantized = fake.model.layers[0].mlp.down_proj
    quantized.quant_method = DraftFp8LinearMethod("cutlass", 64, 32)  # not unquantized
    other = fake.model.layers[0].mlp.gate_up_proj
    other.weight.data = other.weight.data.float()  # not bf16/fp16
    DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    assert quantized.quant_method.kernel == "cutlass"
    assert isinstance(other.quant_method, UnquantizedLinearMethod)
    assert other.weight.dtype == torch.float32
    # 5 of 6 layers run FP8 (one was already): the float32 one stays, a warning and
    # no success line.
    assert not log.lines["info"]
    assert len(log.lines["warning"]) == 1
    assert "does not apply in full: 5 of 6" in log.lines["warning"][0]


@pytest.mark.parametrize("kernel", KERNELS)
def test_hook_logs_the_success_line_the_launch_check_reads(monkeypatch, kernel):
    if kernel == "marlin" and not torch.cuda.is_available():
        pytest.skip("the Marlin repack needs a GPU")
    monkeypatch.setenv(FLAG, "2")
    monkeypatch.setenv(KERNEL_FLAG, kernel)
    log = LogRecorder(monkeypatch)
    fake = fake_draft(num_layers=5)
    if torch.cuda.is_available():
        for lin in draft_linears(fake):
            lin.to("cuda")
    DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    assert not log.lines["warning"]
    (line,) = log.lines["info"]
    assert line.startswith(
        f"DFlash2 draft weights in FP8 (VLLM_MUSE_SPARK_SM120_DRAFT_FP8=2, kernel "
        f"{kernel}): 30 linear layers quantized, "
    )
    # A second call finds everything quantized and says nothing.
    DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    assert len(log.lines["info"]) == 1 and not log.lines["warning"]


def test_a_bad_kernel_name_is_rejected():
    with pytest.raises(ValueError, match="expected one of"):
        DraftFp8LinearMethod("triton", 8, 8)


def test_a_bad_kernel_name_is_rejected_before_any_weight_changes(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(KERNEL_FLAG, "triton")
    fake = fake_draft(num_layers=1)
    before = [lin.weight.data.clone() for lin in draft_linears(fake)]
    with pytest.raises(ValueError, match="expected one of"):
        DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    for lin, w in zip(draft_linears(fake), before, strict=True):
        assert isinstance(lin.quant_method, UnquantizedLinearMethod)
        assert torch.equal(lin.weight.data, w)
    with pytest.raises(ValueError, match="expected one of"):
        quantize_draft_linear(next(draft_linears(fake)), "triton")


def test_hook_refuses_to_quantize_without_the_fused_context_kv_weight(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(KERNEL_FLAG, "cutlass")
    fake = fake_draft(num_layers=1)
    fake.model._fused_kv_weight = None  # a build that came after the quantization
    with pytest.raises(RuntimeError, match="fused context K/V weight is missing"):
        DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    assert all(
        isinstance(lin.quant_method, UnquantizedLinearMethod)
        for lin in draft_linears(fake)
    )


def test_hook_builds_the_fused_context_kv_weight_when_load_weights_did_not(
    monkeypatch,
):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(KERNEL_FLAG, "cutlass")
    fake = fake_draft(num_layers=1)
    del fake.model._fused_kv_weight
    built = []

    def build():
        built.append(
            sum(
                isinstance(lin.quant_method, DraftFp8LinearMethod)
                for lin in draft_linears(fake)
            )
        )
        fake.model._fused_kv_weight = torch.zeros(4, 64, dtype=torch.bfloat16)

    fake.model._build_fused_kv_buffers = build
    DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    assert built == [0]  # before the first layer was quantized
    assert all(
        isinstance(lin.quant_method, DraftFp8LinearMethod)
        for lin in draft_linears(fake)
    )


def test_a_layer_cutlass_cannot_take_stays_bf16_and_is_reported(monkeypatch):
    # N or K not a multiple of 16: cutlass_scaled_mm would run a Triton kernel.
    layer = linear_with(realistic_weight(24, 64, seed=20))  # N = 24
    assert quantize_draft_linear(layer, "cutlass") == 0
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    layer = linear_with(realistic_weight(32, 72, seed=21))  # K = 72
    assert quantize_draft_linear(layer, "cutlass") == 0
    assert isinstance(layer.quant_method, UnquantizedLinearMethod)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(KERNEL_FLAG, "cutlass")
    log = LogRecorder(monkeypatch)
    fake = fake_draft(num_layers=1, k=64, n=24)  # N = 24 in the layers with N rows
    DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    assert not log.lines["info"]
    assert "does not apply in full" in log.lines["warning"][0]


def test_quantize_rows_rounds_the_scale_to_bf16_when_asked():
    weight = realistic_weight(64, 128, seed=22)
    wq, scale = quantize_rows(weight, scale_in_bf16=True)
    assert torch.equal(scale, scale.bfloat16().float())
    assert torch.all(wq.float().abs() <= 448.0)
    # The weights are quantized against that scale: the error stays at the e4m3 level.
    dequant = wq.float() * scale[:, None]
    rel = (dequant - weight.float()).pow(2).mean().sqrt()
    assert rel / weight.float().pow(2).mean().sqrt() < 0.04
    plain = quantize_rows(weight)[1]
    assert not torch.equal(plain, scale)  # the fp32 scales are not all bf16 values


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_quantize_rows_refuses_a_non_finite_weight(bad):
    weight = realistic_weight(8, 32, seed=23)
    weight[5, 7] = bad
    with pytest.raises(ValueError, match="non-finite"):
        quantize_rows(weight)


def test_attach_fp8_head_gates_on_the_flag(monkeypatch):
    fake = fake_draft(num_layers=1)
    head = SimpleNamespace(
        weight=torch.zeros(4, 4, dtype=torch.bfloat16), quant_method=object()
    )
    monkeypatch.setenv(FLAG, "1")
    DFlash2Qwen3ForCausalLM.attach_fp8_head(fake, head)  # not mode 2: nothing
    assert fake.fp8_head is None
    monkeypatch.setenv(FLAG, "2")
    log = LogRecorder(monkeypatch)
    DFlash2Qwen3ForCausalLM.attach_fp8_head(fake, head)  # not a plain matrix: warns
    assert fake.fp8_head is None
    assert "DRAFT_FP8=2 does not apply to the head" in log.lines["warning"][0]


def cpu_head(rows: int = 40, hidden: int = 64):
    return SimpleNamespace(
        weight=nn.Parameter(
            realistic_weight(rows, hidden, seed=24), requires_grad=False
        ),
        quant_method=UnquantizedEmbeddingMethod(),
        tp_size=1,
        shard_indices=SimpleNamespace(org_vocab_start_index=0, num_org_vocab_padding=0),
    )


def test_attach_fp8_head_copies_a_plain_head(monkeypatch):
    monkeypatch.setenv(FLAG, "2")
    log = LogRecorder(monkeypatch)
    fake = fake_draft(num_layers=1)
    DFlash2Qwen3ForCausalLM.attach_fp8_head(fake, cpu_head())
    assert isinstance(fake.fp8_head, DraftFp8Head)
    assert fake.fp8_head.weight_q.shape == (48, 64)  # 40 rows padded to 16
    assert (
        "DFlash2 candidate head in FP8 (VLLM_MUSE_SPARK_SM120_DRAFT_FP8=2)"
        in (log.lines["info"][0])
    )
    assert not log.lines["warning"]


def test_attach_fp8_head_skips_a_head_dtype_that_differs_from_the_model(monkeypatch):
    monkeypatch.setenv(FLAG, "2")
    log = LogRecorder(monkeypatch)
    fake = fake_draft(num_layers=1)
    # --hf-overrides head_dtype: LogitsProcessor._apply_head would reject the FP8 head
    # at the first draft step; the copy is skipped at load instead.
    fake.candidate_logits_processor = SimpleNamespace(head_dtype=torch.float32)
    DFlash2Qwen3ForCausalLM.attach_fp8_head(fake, cpu_head())
    assert fake.fp8_head is None
    assert "head_dtype" in log.lines["warning"][0]
    assert "does not apply to the head" in log.lines["warning"][0]
    fake.candidate_logits_processor = SimpleNamespace(head_dtype=torch.bfloat16)
    DFlash2Qwen3ForCausalLM.attach_fp8_head(fake, cpu_head())
    assert fake.fp8_head is not None  # the same dtype as the model: fine


# --------------------------------------------------------------------------------
# The kernels on the real shapes (GPU).
# --------------------------------------------------------------------------------

# Error bounds on random data next to what the bf16 GEMM itself shows (about 0.5% of the
# max element): e4m3 weights give about 2.5% of the max element at the largest outputs
# (Marlin) and 3.7% with e4m3 activations too (cutlass); the bounds leave 50% margin.
BOUND_MAX = {"cutlass": 0.06, "marlin": 0.04}
BOUND_RMS = {"cutlass": 0.065, "marlin": 0.045}
# Three chained quantized GEMMs with the multiplicative conv coefficients and
# SiLU-and-mul between them (the MLP half of a draft layer): the errors compound
# (measured relative RMS 9.1% cutlass, 6.5% marlin, at 8 / 128 / 256 rows alike).
BLOCK_BOUND_RMS = {"cutlass": 0.11, "marlin": 0.08}


@needs_cuda
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_error_against_fp32_on_the_real_shapes(shape, kernel):
    n, k = SHAPES[shape]
    weight = realistic_weight(n, k, seed=4, device="cuda")
    layer = linear_with(weight)
    base = layer.weight.data.clone()
    assert quantize_draft_linear(layer, kernel) > 0
    for m in ROWS:
        generator = torch.Generator(device="cpu").manual_seed(m)
        x = torch.randn(m, k, generator=generator).to("cuda", torch.bfloat16)
        ref = x.float() @ base.float().t()
        y = layer(x)
        assert y.shape == (m, n) and y.dtype == torch.bfloat16
        bf16 = errors(F.linear(x, base), ref)
        got = errors(y, ref)
        assert bf16[0] < 0.01
        assert got[0] < BOUND_MAX[kernel], (shape, kernel, m, got)
        assert got[1] < BOUND_RMS[kernel], (shape, kernel, m, got)


@needs_cuda
@pytest.mark.parametrize("kernel", KERNELS)
def test_zero_and_padded_rows_stay_finite(kernel):
    """CUDA graph padding rows can hold zeros: the per-token scale must not divide
    by 0."""
    n, k = SHAPES["qkv"]
    layer = linear_with(realistic_weight(n, k, seed=5, device="cuda"))
    quantize_draft_linear(layer, kernel)
    x = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    x[5:9] = 0
    y = layer(x)
    assert torch.isfinite(y).all()
    assert torch.all(y[5:9] == 0)


@needs_cuda
def test_zero_rows_give_an_empty_result():
    """F.linear accepts M = 0; the per-token quant kernel launches one block per row."""
    n, k = SHAPES["qkv"]
    layer = linear_with(realistic_weight(n, k, seed=25, device="cuda"))
    quantize_draft_linear(layer, "cutlass")
    y = layer(torch.empty(0, k, device="cuda", dtype=torch.bfloat16))
    assert y.shape == (0, n) and y.dtype == torch.bfloat16
    wq, scale = quantize_rows(realistic_weight(n, k, seed=25, device="cuda"))
    empty = cutlass_fp8_linear(
        torch.empty(2, 0, k, device="cuda", dtype=torch.bfloat16), wq, scale
    )
    assert empty.shape == (2, 0, n)
    torch.cuda.synchronize()


@needs_cuda
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("shape", ["gate_up", "down", "kernel_projection"])
def test_repeat_is_bitwise_and_rows_are_independent(shape, kernel):
    n, k = SHAPES[shape]
    layer = linear_with(realistic_weight(n, k, seed=6, device="cuda"))
    quantize_draft_linear(layer, kernel)
    x = torch.randn(256, k, device="cuda", dtype=torch.bfloat16)
    first = layer(x)
    torch.matmul(
        torch.randn(2048, 2048, device="cuda"), torch.randn(2048, 2048, device="cuda")
    )
    assert torch.equal(first, layer(x))
    # Per-token activation scales: a row's output does not depend on the other rows
    # (CUTLASS) -- Marlin, a weight-only kernel, has no activation scales at all. The
    # GEMM tiling may differ with the row count, so compare within rounding.
    small = layer(x[:8])
    assert errors(small, first[:8])[0] < 0.02


@needs_cuda
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("m", [8, 128, 256])
def test_cuda_graph_replay_equals_eager(kernel, m):
    layers = {}
    for shape in ("gate_up", "down", "qkv", "o", "kernel_projection"):
        n, k = SHAPES[shape]
        layers[shape] = linear_with(realistic_weight(n, k, seed=7, device="cuda"))
        quantize_draft_linear(layers[shape], kernel)
    x_static = torch.randn(m, SHAPES["gate_up"][1], device="cuda", dtype=torch.bfloat16)

    def run(x):
        a = layers["gate_up"](x)
        b = layers["down"](a[:, : SHAPES["down"][1]].contiguous())
        c = layers["qkv"](x)
        d = layers["o"](c[:, : SHAPES["o"][1]].contiguous())
        e = layers["kernel_projection"](x)
        return torch.cat([b, d, e], dim=-1)

    run(x_static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y_static = run(x_static)
    for seed in (1, 2):
        x_new = torch.randn(
            m, x_static.shape[1], generator=torch.Generator().manual_seed(seed)
        ).to("cuda", torch.bfloat16)
        x_static.copy_(x_new)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(y_static, run(x_new))


@needs_cuda
@pytest.mark.parametrize("kernel", KERNELS)
def test_draft_mlp_block_forward_error(kernel, default_vllm_config):
    """The draft layer's MLP half end to end, with the real classes at the draft's
    hidden / intermediate sizes (8192 / 28672, TP 1): kernel projection and grouped
    conv, gate_up, SiLU-and-mul, down, conv finish. FP8 against the bf16 block."""
    from vllm.model_executor.models.qwen3 import Qwen3MLP
    from vllm.model_executor.models.qwen3_dflash2 import DFlashGroupedConv

    hidden, inter = 8192, 28672
    conv = DFlashGroupedConv(
        hidden,
        taps=2,
        group_size=16,
        block_size=8,
        params_dtype=torch.bfloat16,
        prefix="conv",
    ).to("cuda")
    mlp = Qwen3MLP(hidden, inter, "silu", quant_config=None, prefix="mlp").to("cuda")
    conv.base_kernel.data = 0.5 * torch.randn(
        2, 2, hidden, device="cuda", dtype=torch.bfloat16
    )
    conv.kernel_projection.weight.data = realistic_weight(
        2048, hidden, seed=11, device="cuda"
    )
    mlp.gate_up_proj.weight.data = realistic_weight(
        2 * inter, hidden, seed=12, device="cuda"
    )
    mlp.down_proj.weight.data = realistic_weight(hidden, inter, seed=13, device="cuda")

    def block(h: torch.Tensor) -> torch.Tensor:
        convolved, coefficients = conv.prepare(h)
        return conv.finish(mlp(convolved), coefficients)

    reference = {}
    inputs = {}
    for m in (8, 128, 256):
        inputs[m] = torch.randn(
            m, hidden, generator=torch.Generator().manual_seed(m)
        ).to("cuda", torch.bfloat16)
        reference[m] = block(inputs[m]).float()
    for module in (conv.kernel_projection, mlp.gate_up_proj, mlp.down_proj):
        assert quantize_draft_linear(module, kernel) > 0
    for m in (8, 128, 256):
        got = errors(block(inputs[m]), reference[m])
        assert got[1] < BLOCK_BOUND_RMS[kernel], (kernel, m, got)
    # CUDA graph replay equals eager through the whole block.
    static = inputs[8].clone()
    block(static)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_static = block(static)
    static.copy_(inputs[128][:8])
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out_static, block(inputs[128][:8].contiguous()))


# --------------------------------------------------------------------------------
# The candidate head (GPU).
# --------------------------------------------------------------------------------

VOCAB_SHARD = 25256  # 202048 / 8
HIDDEN = 8192


def fake_lm_head(vocab_start: int = 3 * VOCAB_SHARD, padding: int = 0):
    weight = realistic_weight(VOCAB_SHARD, HIDDEN, seed=8, device="cuda")
    return SimpleNamespace(
        weight=nn.Parameter(weight, requires_grad=False),
        quant_method=SimpleNamespace(
            apply=lambda head, x, bias=None: F.linear(x, head.weight)
        ),
        tp_size=1,
        shard_indices=SimpleNamespace(
            org_vocab_start_index=vocab_start, num_org_vocab_padding=padding
        ),
    )


@needs_cuda
def test_the_fp8_head_returns_the_bf16_top_k_and_no_padded_column(default_vllm_config):
    k = 16
    lm_head = fake_lm_head()
    fp8_head = DraftFp8Head(lm_head)
    assert fp8_head.weight_q.shape == (VOCAB_SHARD + 8, HIDDEN)  # 25264, CUTLASS rows
    assert fp8_head.shard_indices.num_org_vocab_padding == 8
    assert fp8_head.nbytes < lm_head.weight.numel() * 2 * 0.51
    processor = LogitsProcessor(202048, scale=0.1767766952966369, soft_cap=20.0)
    generator = torch.Generator().manual_seed(9)
    overlap = []
    for m in (8, 128, 256):
        hidden = (torch.randn(m, HIDDEN, generator=generator) * 3).to(
            "cuda", torch.bfloat16
        )
        ref_ids, ref_values = processor.get_top_k_tokens(lm_head, hidden, k)
        ids, values = processor.get_top_k_tokens(fp8_head, hidden, k)
        assert ids.shape == (m, k) and values.shape == (m, k)
        start = lm_head.shard_indices.org_vocab_start_index
        assert ids.min() >= start and ids.max() < start + VOCAB_SHARD
        overlap += [
            len(set(a.tolist()) & set(b.tolist())) / k
            for a, b in zip(ids.cpu(), ref_ids.cpu(), strict=True)
        ]
        # The rank-aligned values agree within the FP8 error of a logit (e4m3 weights
        # and activations: about 3.5% of the largest element, as in the layer test).
        assert (values - ref_values).abs().max() < 0.05 * values.abs().max()
    # Random Gaussian logits have close ranks: the FP8 head swaps some tied entries.
    assert sum(overlap) / len(overlap) > 0.7, sum(overlap) / len(overlap)


@needs_cuda
def test_the_fp8_head_masks_the_vocabulary_padding_too(default_vllm_config):
    lm_head = fake_lm_head(padding=5)
    fp8_head = DraftFp8Head(lm_head)
    assert fp8_head.shard_indices.num_org_vocab_padding == 5 + 8
    processor = LogitsProcessor(202048)
    # Make the padding rows the best candidates: they must still be masked out.
    lm_head.weight.data[-5:] = 1.0
    fp8_head = DraftFp8Head(lm_head)
    hidden = torch.ones(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids, _ = processor.get_top_k_tokens(fp8_head, hidden, 16)
    start = lm_head.shard_indices.org_vocab_start_index
    assert ids.max() < start + VOCAB_SHARD - 5


@needs_cuda
def test_the_head_graph_replay_equals_eager(default_vllm_config):
    lm_head = fake_lm_head()
    fp8_head = DraftFp8Head(lm_head)
    processor = LogitsProcessor(202048, scale=0.1767766952966369, soft_cap=20.0)
    hidden = torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16)
    processor.get_top_k_tokens(fp8_head, hidden, 16)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids_static, values_static = processor.get_top_k_tokens(fp8_head, hidden, 16)
    hidden.copy_(torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    ids, values = processor.get_top_k_tokens(fp8_head, hidden, 16)
    assert torch.equal(ids_static, ids) and torch.equal(values_static, values)


# --------------------------------------------------------------------------------
# TP8 (8 GPUs): row-parallel partials with per-rank scales, rank-identical replicated
# layers, the FP8 head under the cross-rank top-k gather.
# --------------------------------------------------------------------------------

TP_ROWS = (8, 128)
TP_QUERIES = 16


def _fp8_tp_worker(rank: int, world: int, port: int, out_path: str) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
    from vllm.distributed.parallel_state import graph_capture
    from vllm.model_executor.layers.linear import RowParallelLinear

    torch.cuda.set_device(rank)
    if os.environ.get("MUSE_TOPO_ENABLE") == "1":
        # The host plugin patches the TP communicator (the collectives of production).
        from vllm.plugins import load_general_plugins

        load_general_plugins()
    config = VllmConfig(parallel_config=ParallelConfig(tensor_parallel_size=world))
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=world,
            rank=rank,
            local_rank=rank,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
            backend="nccl",
        )
        initialize_model_parallel(tensor_model_parallel_size=world)

    def identical_on_all_ranks(t: torch.Tensor) -> bool:
        everyone = tensor_model_parallel_all_gather(t.reshape(1, -1), 0)
        return bool((everyone == everyone[0]).all())

    def captured_equals_eager(fn, eager: torch.Tensor) -> bool:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with (
            graph_capture(device=torch.device("cuda", rank)) as capture,
            torch.cuda.graph(graph, stream=capture.stream),
        ):
            out = fn()
        graph.replay()
        torch.cuda.synchronize()
        return bool(torch.equal(out, eager))

    results = {}
    g = torch.Generator(device="cpu").manual_seed(31)
    with set_current_vllm_config(config), torch.inference_mode():
        # Row parallel: o_proj (64 heads x 128 = 8192 -> 8192) and down (28672 -> 8192).
        # Every rank holds a K / 8 slice of the weight and scales its own partial.
        for name, k_full in (("o", 8192), ("down", 28672)):
            full = realistic_weight(8192, k_full, seed=40 + len(name), device="cuda")
            k_rank = k_full // world
            with torch.device("cuda"):
                layer = RowParallelLinear(
                    k_full,
                    8192,
                    bias=False,
                    input_is_parallel=True,
                    params_dtype=torch.bfloat16,
                    return_bias=False,
                    prefix=f"{name}_proj",
                )
            layer.weight.data = full[
                :, rank * k_rank : (rank + 1) * k_rank
            ].contiguous()
            assert quantize_draft_linear(layer, "cutlass") > 0
            for m in TP_ROWS:
                x_full = torch.randn(m, k_full, generator=g).to("cuda", torch.bfloat16)
                reference = x_full.float() @ full.float().t()
                x_rank = x_full[:, rank * k_rank : (rank + 1) * k_rank].contiguous()
                y = layer(x_rank)
                results[f"{name}_{m}"] = dict(
                    error=errors(y, reference),
                    same_on_all_ranks=identical_on_all_ranks(y),
                    graph_equal=captured_equals_eager(
                        lambda lin=layer, a=x_rank: lin(a), y.clone()
                    ),
                )
        # Replicated kernel projection: the same weights and inputs on every rank give
        # bitwise the same output.
        kp = realistic_weight(2048, 8192, seed=50, device="cuda")
        with torch.device("cuda"):
            kproj = ReplicatedLinear(
                8192, 2048, bias=False, params_dtype=torch.bfloat16, return_bias=False
            )
        kproj.weight.data = kp.clone()
        assert quantize_draft_linear(kproj, "cutlass") > 0
        for m in TP_ROWS:
            x = torch.randn(m, 8192, generator=g).to("cuda", torch.bfloat16)
            y = kproj(x)
            results[f"kproj_{m}"] = dict(
                error=errors(y, x.float() @ kp.float().t()),
                same_on_all_ranks=identical_on_all_ranks(y),
                graph_equal=captured_equals_eager(lambda a=x: kproj(a), y.clone()),
            )
        # The candidate head: a vocabulary shard per rank (202048 / 8), queries aimed
        # at tokens of every shard (the hidden state is the token's own row, so the
        # top-1 is decisive), the cross-rank top-k gather over the FP8 values.
        shard = VOCAB_SHARD
        weight = realistic_weight(shard, HIDDEN, seed=60 + rank, device="cuda")
        head = SimpleNamespace(
            weight=nn.Parameter(weight, requires_grad=False),
            quant_method=SimpleNamespace(
                apply=lambda h, x, bias=None: F.linear(x, h.weight)
            ),
            tp_size=world,
            shard_indices=SimpleNamespace(
                org_vocab_start_index=rank * shard, num_org_vocab_padding=0
            ),
        )
        fp8_head = DraftFp8Head(head)
        picks = torch.randint(0, shard, (TP_QUERIES // world,), generator=g).cuda()
        aimed = weight[picks] * (0.2 / 0.02)
        queries = tensor_model_parallel_all_gather(aimed, dim=0)  # [16, 8192]
        queries = torch.cat(
            [queries, torch.randn(48, HIDDEN, generator=g).to("cuda", torch.bfloat16)]
        )
        processor = LogitsProcessor(
            world * shard, scale=0.1767766952966369, soft_cap=20.0
        )
        ref_ids, ref_values = processor.get_top_k_tokens(head, queries, 16)
        ids, values = processor.get_top_k_tokens(fp8_head, queries, 16)
        aimed_rows = slice(0, TP_QUERIES)
        overlap = [
            len(set(a.tolist()) & set(b.tolist())) / 16
            for a, b in zip(ids.cpu(), ref_ids.cpu(), strict=True)
        ]
        results["head"] = dict(
            top1_equal=bool((ids[aimed_rows, 0] == ref_ids[aimed_rows, 0]).all()),
            value_error=float((values - ref_values).abs().max() / values.abs().max()),
            overlap=sum(overlap) / len(overlap),
            shards_hit=len({int(i) // shard for i in ids[aimed_rows, 0].tolist()}),
            in_range=bool((ids >= 0).all() and (ids < world * shard).all()),
            same_on_all_ranks=identical_on_all_ranks(ids)
            and identical_on_all_ranks(values),
            graph_equal=captured_equals_eager(
                lambda: processor.get_top_k_tokens(fp8_head, queries, 16)[1], values
            ),
        )
        torch.distributed.barrier()
    if rank == 0:
        topo = getattr(get_tp_group().device_communicator, "topo_comm", None)
        results["_collectives"] = "muse_topo" if topo is not None else "nccl"
        with open(out_path, "w") as f:
            json.dump(results, f)
    torch.distributed.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires 8 GPUs")
def test_the_fp8_layers_and_head_on_8_ranks(tmp_path):
    """Real per-rank shapes and collectives (the host plugin's when MUSE_TOPO_ENABLE=1):
    row-parallel o_proj / down with a per-rank weight scale and a per-token activation
    scale, summed by the all-reduce, stay within the single-rank error bound of the
    fp32 reference and are identical on all ranks; the replicated kernel projection is
    bitwise the same on every rank; the FP8 head's cross-rank top-k gather returns the
    aimed tokens of every shard, identical on all ranks; CUDA graph replay equals
    eager."""
    import torch.multiprocessing as mp

    out_path = str(tmp_path / "results.json")
    port = 29200 + os.getpid() % 1000
    mp.spawn(_fp8_tp_worker, args=(8, port, out_path), nprocs=8)
    with open(out_path) as f:
        results = json.load(f)
    print("collectives:", results.pop("_collectives"))
    for name, result in results.items():
        print(name, result)
    for name, result in results.items():
        assert result["same_on_all_ranks"], (name, result)
        assert result["graph_equal"], (name, result)
        if name == "head":
            assert result["top1_equal"], result
            assert result["in_range"], result
            assert result["shards_hit"] >= 3, result  # the queries span the shards
            assert result["value_error"] < 0.05, result
            assert result["overlap"] > 0.6, result
        else:
            assert result["error"][0] < BOUND_MAX["cutlass"], (name, result)
            assert result["error"][1] < BOUND_RMS["cutlass"], (name, result)


# --------------------------------------------------------------------------------
# The real checkpoint (GPU, skipped where the draft is not cached).
# --------------------------------------------------------------------------------

DRAFT_GLOB = (
    "/raid/.cache/huggingface/hub/models--RadixArk--Muse-Spark-Open-DFlash2/"
    "snapshots/*/model.safetensors"
)


@needs_cuda
@pytest.mark.skipif(not glob.glob(DRAFT_GLOB), reason="the DFlash2 draft is not cached")
@pytest.mark.parametrize("kernel", KERNELS)
def test_error_on_the_real_draft_weights(kernel):
    from safetensors import safe_open

    path = glob.glob(DRAFT_GLOB)[0]
    tp, rank = 8, 3
    names = {
        "gate_up": ["midlayer.mlp.gate_proj.weight", "midlayer.mlp.up_proj.weight"],
        "down": ["midlayer.mlp.down_proj.weight"],
        "qkv": [
            "midlayer.self_attn.q_proj.weight",
            "midlayer.self_attn.k_proj.weight",
            "midlayer.self_attn.v_proj.weight",
        ],
        "o": ["midlayer.self_attn.o_proj.weight"],
    }
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        probe = [n for names_ in names.values() for n in names_]
        # The draft ships one layer ("midlayer") or numbered layers.
        if not all(n in keys for n in probe):
            names = {
                shape: [n.replace("midlayer", "layers.0") for n in ns]
                for shape, ns in names.items()
            }
        if not all(n in keys for names_ in names.values() for n in names_):
            pytest.skip("unexpected draft tensor names")
        for shape, ns in names.items():
            shards = []
            for name in ns:
                w = f.get_tensor(name)
                if shape in ("down", "o"):  # row parallel: shard the input dim
                    cols = w.shape[1] // tp
                    shards.append(w[:, rank * cols : (rank + 1) * cols])
                else:  # column parallel: shard the output dim
                    rows = w.shape[0] // tp if w.shape[0] >= tp * 128 else w.shape[0]
                    shards.append(w[rank * rows : (rank + 1) * rows])
            weight = torch.cat(shards, dim=0).to("cuda", torch.bfloat16)
            layer = linear_with(weight)
            base = layer.weight.data.clone()
            quantize_draft_linear(layer, kernel)
            for m in (8, 128, 256):
                x = torch.randn(m, weight.shape[1], device="cuda", dtype=torch.bfloat16)
                ref = x.float() @ base.float().t()
                got = errors(layer(x), ref)
                assert got[1] < BOUND_RMS[kernel], (shape, kernel, m, got)
