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
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

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
    quantize_draft_linear,
    quantize_rows,
)
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
    fake.fp8_head = None
    return fake


def draft_linears(fake):
    for layer in fake.model.layers:
        yield layer.self_attn.qkv_proj
        yield layer.self_attn.o_proj
        yield layer.mlp.gate_up_proj
        yield layer.mlp.down_proj
        yield layer.attention_conv.kernel_projection
        yield layer.mlp_conv.kernel_projection


def test_hook_is_a_noop_when_the_flag_is_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
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
def test_hook_quantizes_the_draft_linears_and_leaves_the_fc(
    monkeypatch, flag, kernel
):
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
    fake = fake_draft(num_layers=1)
    quantized = fake.model.layers[0].mlp.down_proj
    quantized.quant_method = DraftFp8LinearMethod("cutlass", 64, 32)  # not unquantized
    other = fake.model.layers[0].mlp.gate_up_proj
    other.weight.data = other.weight.data.float()  # not bf16/fp16
    DFlash2Qwen3ForCausalLM.process_weights_after_loading(fake)
    assert quantized.quant_method.kernel == "cutlass"
    assert isinstance(other.quant_method, UnquantizedLinearMethod)
    assert other.weight.dtype == torch.float32


def test_a_bad_kernel_name_is_rejected():
    with pytest.raises(ValueError, match="expected one of"):
        DraftFp8LinearMethod("triton", 8, 8)


def test_attach_fp8_head_gates_on_the_flag(monkeypatch):
    fake = fake_draft(num_layers=1)
    head = SimpleNamespace(
        weight=torch.zeros(4, 4, dtype=torch.bfloat16), quant_method=object()
    )
    monkeypatch.setenv(FLAG, "1")
    DFlash2Qwen3ForCausalLM.attach_fp8_head(fake, head)  # not mode 2: nothing
    assert fake.fp8_head is None
    monkeypatch.setenv(FLAG, "2")
    DFlash2Qwen3ForCausalLM.attach_fp8_head(fake, head)  # not a plain matrix: warns
    assert fake.fp8_head is None


# --------------------------------------------------------------------------------
# The kernels on the real shapes (GPU).
# --------------------------------------------------------------------------------

# Error bounds on random data next to what the bf16 GEMM itself shows (about 0.5% of the
# max element): e4m3 weights give about 2.5% of the max element at the largest outputs
# (Marlin) and 3.7% with e4m3 activations too (cutlass); the bounds leave 50% margin.
BOUND_MAX = {"cutlass": 0.06, "marlin": 0.04}
BOUND_RMS = {"cutlass": 0.065, "marlin": 0.045}


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
    """CUDA graph padding rows can hold zeros: the per-token scale must not divide by 0."""
    n, k = SHAPES["qkv"]
    layer = linear_with(realistic_weight(n, k, seed=5, device="cuda"))
    quantize_draft_linear(layer, kernel)
    x = torch.randn(16, k, device="cuda", dtype=torch.bfloat16)
    x[5:9] = 0
    y = layer(x)
    assert torch.isfinite(y).all()
    assert torch.all(y[5:9] == 0)


@needs_cuda
@pytest.mark.parametrize("kernel", KERNELS)
@pytest.mark.parametrize("shape", ["gate_up", "down", "kernel_projection"])
def test_repeat_is_bitwise_and_rows_are_independent(shape, kernel):
    n, k = SHAPES[shape]
    layer = linear_with(realistic_weight(n, k, seed=6, device="cuda"))
    quantize_draft_linear(layer, kernel)
    x = torch.randn(256, k, device="cuda", dtype=torch.bfloat16)
    first = layer(x)
    torch.matmul(torch.randn(2048, 2048, device="cuda"), torch.randn(2048, 2048, device="cuda"))
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
        # The rank-aligned values agree within the FP8 error of a logit (e4m3 weights and
        # activations: about 3.5% of the largest element, as in the layer test).
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
