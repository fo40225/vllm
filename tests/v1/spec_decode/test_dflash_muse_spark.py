# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash/DFlash2 drafters on a Muse Spark target.

Muse Spark keeps its residual stream in FP32 and hands the five aux taps to the
drafter in that dtype, while the DFlash fc is a BF16 linear. These tests cover the
cast and, on the meta device at the production TP=8 layout, that the two released
DFlash2 checkpoints (nvidia and RadixArk) build and load without a config change.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

NUM_TAPS = 5
HIDDEN = 64


class _RecordingModel:
    """Stands in for the draft model: records what combine_hidden_states sees."""

    def __init__(self):
        self.seen: torch.Tensor | None = None

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.seen = hidden_states
        return hidden_states


def _speculator(dtype: torch.dtype) -> tuple[DFlashSpeculator, _RecordingModel]:
    speculator = object.__new__(DFlashSpeculator)
    speculator.dtype = dtype
    speculator.model = _RecordingModel()
    return speculator, speculator.model


def _taps(num_tokens: int, dtype: torch.dtype) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return [
        torch.randn(num_tokens, HIDDEN, generator=generator).to(dtype)
        for _ in range(NUM_TAPS)
    ]


def test_aux_taps_are_cast_before_the_concat():
    speculator, model = _speculator(torch.bfloat16)
    taps = _taps(6, torch.float32)

    speculator.combine_aux_hidden_states(taps)

    assert model.seen is not None
    # The concat itself is built in the draft dtype: no FP32 [T, 5H] intermediate.
    assert model.seen.dtype == torch.bfloat16
    assert model.seen.shape == (6, NUM_TAPS * HIDDEN)
    expected = torch.cat([t.to(torch.bfloat16) for t in taps], dim=-1)
    assert torch.equal(model.seen, expected)


def test_aux_taps_already_in_the_draft_dtype_are_untouched():
    speculator, model = _speculator(torch.bfloat16)
    taps = _taps(3, torch.bfloat16)

    speculator.combine_aux_hidden_states(taps)

    assert model.seen is not None
    assert torch.equal(model.seen, torch.cat(taps, dim=-1))


class _Fc(nn.Module):
    """A BF16 fc with the attributes combine_hidden_states reads."""

    def __init__(self, in_features: int, out_features: int, weight_dtype=None):
        super().__init__()
        self.input_size = in_features
        self.params_dtype = torch.bfloat16
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.linear.to(torch.bfloat16)
        # A quantized fc stores packed weights, so its weight dtype is not the
        # activation dtype; params_dtype is.
        self.weight = (
            self.linear.weight
            if weight_dtype is None
            else torch.zeros(out_features, in_features, dtype=weight_dtype)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_input_dtype = x.dtype
        return self.linear(x)


def _draft(fc: _Fc) -> SimpleNamespace:
    return SimpleNamespace(model=SimpleNamespace(use_aux_hidden_state=True, fc=fc))


def test_fp32_fc_input_fails_without_the_cast():
    """The failure this change fixes: a BF16 fc fed the FP32 Muse taps."""
    fc = _Fc(NUM_TAPS * HIDDEN, HIDDEN)
    with pytest.raises(RuntimeError, match="dtype"):
        fc(torch.randn(2, NUM_TAPS * HIDDEN))


@pytest.mark.parametrize("weight_dtype", [None, torch.uint8])
def test_combine_hidden_states_casts_fp32_taps_to_the_fc_dtype(weight_dtype):
    fc = _Fc(NUM_TAPS * HIDDEN, HIDDEN, weight_dtype)
    draft = _draft(fc)
    features = torch.cat(_taps(4, torch.float32), dim=-1)

    out = DFlashQwen3ForCausalLM.combine_hidden_states(draft, features)

    assert fc.last_input_dtype == torch.bfloat16
    assert out.dtype == torch.bfloat16
    expected = fc.linear(features.to(torch.bfloat16))
    assert torch.equal(out, expected)


def test_combine_hidden_states_single_row_fp32():
    fc = _Fc(NUM_TAPS * HIDDEN, HIDDEN)
    features = torch.cat(_taps(1, torch.float32), dim=-1)[0]

    out = DFlashQwen3ForCausalLM.combine_hidden_states(_draft(fc), features)

    assert out.shape == (HIDDEN,)
    assert out.dtype == torch.bfloat16


def test_propose_path_end_to_end_with_fp32_taps():
    """Speculator cast + model cast together, on a real BF16 linear."""
    fc = _Fc(NUM_TAPS * HIDDEN, HIDDEN)
    speculator = object.__new__(DFlashSpeculator)
    speculator.dtype = torch.bfloat16
    speculator.model = SimpleNamespace(
        combine_hidden_states=lambda x: DFlashQwen3ForCausalLM.combine_hidden_states(
            _draft(fc), x
        )
    )
    taps = _taps(5, torch.float32)

    out = speculator.combine_aux_hidden_states(taps)

    assert out.dtype == torch.bfloat16
    assert out.shape == (5, HIDDEN)


# The two released Muse Spark DFlash2 drafters, as shipped (config.json and the
# tensor layout of model.safetensors; only fields the loader reads are kept).
_COMMON_CONFIG = dict(
    vocab_size=202048,
    hidden_size=8192,
    num_hidden_layers=5,
    hidden_act="silu",
    max_position_embeddings=1048576,
    attention_bias=False,
    is_causal=False,
    sliding_window=2048,
    layer_types=["sliding_attention"] * 5,
    # As resolved by vLLM's config loader for both repos (the nvidia config.json
    # only has the legacy top-level rope_theta); the draft must run on theta 5e5,
    # not set_default_rope_theta's 1e6 fallback.
    rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
)
_NVIDIA = dict(
    _COMMON_CONFIG,
    num_attention_heads=128,
    num_key_value_heads=16,
    head_dim=64,
    intermediate_size=24576,
    rms_norm_eps=1e-6,
    dflash_config={
        "mask_token_id": 201817,
        "target_layer_ids": [3, 19, 31, 47, 59],
        "use_swa": True,
        "swa_window_size": 2048,
        "causal": False,
        "projector_type": "dflash2",
        "conv_kernel_size": 2,
        "conv_group_size": 16,
        "selector_rank": 256,
        "selector_top_k": 16,
        "block_size": 8,
        "output_multiplier": 0.17677669529663687,
        "final_logit_softcapping": 20.0,
    },
)
_RADIXARK = dict(
    _COMMON_CONFIG,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=28672,
    rms_norm_eps=1e-5,
    draft_vocab_size=202048,
    dflash_config={
        "attention_mode": "gqa",
        "block_size": 8,
        "conv_group_size": 16,
        "conv_kernel_size": 2,
        "final_logit_softcapping": 20,
        "mask_token_id": 200002,
        "output_multiplier": 0.1767766952966369,
        "selector_rank": 256,
        "selector_top_k": 16,
        "target_layer_ids": [1, 16, 30, 44, 59],
    },
)
_TP = 8
_TARGET_LAYERS = 62
_K = 7


def _checkpoint(cfg: dict) -> dict[str, tuple[int, ...]]:
    h, hd, inter = cfg["hidden_size"], cfg["head_dim"], cfg["intermediate_size"]
    q, kv = cfg["num_attention_heads"] * hd, cfg["num_key_value_heads"] * hd
    draft = cfg["dflash_config"]
    taps, groups = draft["conv_kernel_size"], h // draft["conv_group_size"]
    rank = draft["selector_rank"]
    shapes = {
        "fc.weight": (h, len(draft["target_layer_ids"]) * h),
        "hidden_norm.weight": (h,),
        "norm.weight": (h,),
        "candidate_selector.hidden_projection.weight": (rank, h),
        "candidate_selector.predecessor_codebook": (cfg["vocab_size"], rank),
        "candidate_selector.successor_codebook": (cfg["vocab_size"], rank),
    }
    for i in range(cfg["num_hidden_layers"]):
        prefix = f"layers.{i}."
        for conv in ("attention_conv", "mlp_conv"):
            shapes[f"{prefix}{conv}.base_kernel"] = (2, taps, h)
            shapes[f"{prefix}{conv}.kernel_projection.weight"] = (2 * taps * groups, h)
        shapes[f"{prefix}input_layernorm.weight"] = (h,)
        shapes[f"{prefix}post_attention_layernorm.weight"] = (h,)
        shapes[f"{prefix}self_attn.q_proj.weight"] = (q, h)
        shapes[f"{prefix}self_attn.k_proj.weight"] = (kv, h)
        shapes[f"{prefix}self_attn.v_proj.weight"] = (kv, h)
        shapes[f"{prefix}self_attn.o_proj.weight"] = (h, q)
        shapes[f"{prefix}self_attn.q_norm.weight"] = (hd,)
        shapes[f"{prefix}self_attn.k_norm.weight"] = (hd,)
        shapes[f"{prefix}mlp.gate_proj.weight"] = (inter, h)
        shapes[f"{prefix}mlp.up_proj.weight"] = (inter, h)
        shapes[f"{prefix}mlp.down_proj.weight"] = (h, inter)
    return shapes


def _build_draft(cfg: dict, monkeypatch):
    """Build DFlash2Qwen3ForCausalLM for a Muse Spark target at TP=8 on meta."""
    from vllm.config import set_current_vllm_config
    from vllm.distributed import parallel_state
    from vllm.model_executor.models import qwen3_dflash
    from vllm.model_executor.models.qwen3_dflash2 import DFlash2Qwen3ForCausalLM
    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda *args, **kwargs: (
            "vllm.v1.attention.backends.cpu_attn.CPUAttentionBackend"
        ),
    )
    monkeypatch.setattr(
        parallel_state, "_TP", SimpleNamespace(rank_in_group=0, world_size=_TP)
    )
    # No mask_embedding.pt next to either checkpoint (offline, nothing to fetch).
    monkeypatch.setattr(qwen3_dflash, "get_hf_file_bytes", lambda *a, **kw: None)

    model_config = SimpleNamespace(
        dtype=torch.bfloat16,
        is_mm_prefix_lm=False,
        rswa_window=None,
        get_total_num_hidden_layers=lambda: _TARGET_LAYERS,
        get_vocab_size=lambda: cfg["vocab_size"],
        head_dtype=None,
    )
    compilation_config = SimpleNamespace(
        compile_custom_ops=False,
        custom_ops="all",
        enabled_custom_ops=set(),
        static_forward_context={},
        mode=0,  # CompilationMode.NONE
    )
    speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(
            hf_config=SimpleNamespace(**cfg),
            quantization=None,
            model="draft",
            revision=None,
        ),
        num_speculative_tokens=_K,
        enable_adaptive_verification=False,
    )
    vllm_config = SimpleNamespace(
        speculative_config=speculative_config,
        model_config=model_config,
        compilation_config=compilation_config,
        load_config=SimpleNamespace(quantization=None, quantization_param_path=None),
        cache_config=SimpleNamespace(
            block_size=16,
            user_specified_block_size=False,
            kv_cache_dtype_skip_layers=[],
            cache_dtype="auto",
            sliding_window=None,
            enable_prefix_caching=False,
        ),
        kv_transfer_config=None,
        attention_config=SimpleNamespace(
            use_non_causal=True, backend=None, backend_per_kind={}
        ),
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, decode_context_parallel_size=1
        ),
        kernel_config=SimpleNamespace(linear_backend="auto"),
    )
    with set_current_vllm_config(vllm_config), torch.device("meta"):
        return DFlash2Qwen3ForCausalLM(vllm_config=vllm_config), vllm_config


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "cfg, mask_token_id, aux_layers, heads_per_rank, kv_heads_per_rank, head_dim",
    [
        (_NVIDIA, 201817, (4, 20, 32, 48, 60), 16, 2, 64),
        (_RADIXARK, 200002, (2, 17, 31, 45, 60), 8, 1, 128),
    ],
    ids=["nvidia", "radixark"],
)
def test_muse_spark_dflash2_checkpoint_builds_and_loads_at_tp8(
    monkeypatch,
    cfg,
    mask_token_id,
    aux_layers,
    heads_per_rank,
    kv_heads_per_rank,
    head_dim,
):
    from vllm.model_executor.models import qwen3_dflash
    from vllm.v1.worker.gpu.spec_decode.eagle.eagle3_utils import (
        get_eagle3_aux_layers_from_config,
    )
    from vllm.v1.worker.gpu.spec_decode.utils import get_parallel_drafting_token_id

    draft, vllm_config = _build_draft(cfg, monkeypatch)
    model = draft.model
    hf_config = vllm_config.speculative_config.draft_model_config.hf_config

    # Target taps: DFlash ids are "after layer i", the Muse capture index is i + 1.
    assert get_eagle3_aux_layers_from_config(vllm_config.speculative_config) == (
        aux_layers
    )
    assert model.fc.input_size == len(aux_layers) * cfg["hidden_size"]
    # combine_hidden_states casts the taps to this dtype before the fc GEMM.
    assert model.fc.params_dtype == torch.bfloat16
    assert get_parallel_drafting_token_id(hf_config) == mask_token_id
    assert model.mask_token_id == mask_token_id

    # Every layer: sliding window 2048, bidirectional over the block.
    assert draft.get_draft_attn_causal() == [False] * 5
    assert len(model.layers) == 5
    for layer in model.layers:
        attn = layer.self_attn
        assert attn.sliding_window == 2048
        assert attn.num_heads == heads_per_rank
        assert attn.num_kv_heads == kv_heads_per_rank
        assert attn.head_dim == head_dim
        assert attn.rotary_emb.base == 500000
        assert attn.q_norm.variance_epsilon == cfg["rms_norm_eps"]
        for conv in (layer.attention_conv, layer.mlp_conv):
            assert conv.block_size == 1 + _K
            assert conv.taps == 2 and conv.group_size == 16
            assert conv.num_groups == cfg["hidden_size"] // 16
    assert model.candidate_selector.top_k == 16
    assert draft.candidate_logits_processor.scale == pytest.approx(0.1767766952966369)
    assert draft.candidate_logits_processor.soft_cap == 20.0
    assert draft.draft_id_to_target_id is None  # draft vocab == target vocab

    # The checkpoint's tensors all find a destination with the right shape, and
    # every draft parameter except the target-shared ones gets loaded.
    loaded: set[str] = set()
    real_load_weights = qwen3_dflash.AutoWeightsLoader.load_weights

    def recording_load_weights(self, *args, **kwargs):
        result = real_load_weights(self, *args, **kwargs)
        loaded.update(result)
        return result

    monkeypatch.setattr(
        qwen3_dflash.AutoWeightsLoader, "load_weights", recording_load_weights
    )
    weights = [
        (name, torch.empty(shape, dtype=torch.bfloat16, device="meta"))
        for name, shape in _checkpoint(cfg).items()
    ]
    draft.load_weights(weights)

    shared_with_the_target = ("embed_tokens", "lm_head", "mask_embedding")
    expected = {
        name
        for name, _ in draft.named_parameters()
        if not any(shared in name for shared in shared_with_the_target)
    }
    assert expected <= loaded
    assert "model.candidate_selector.successor_codebook" in loaded
    assert "model.layers.4.mlp_conv.base_kernel" in loaded
