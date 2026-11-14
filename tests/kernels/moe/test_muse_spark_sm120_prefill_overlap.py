# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbatched prefill of MuseSpark on SM120: when the launch config turns it on,
what keeps it off, and the multimodal check of the DBO support list."""

from types import SimpleNamespace

import pytest

import vllm.platforms
from vllm.config.multimodal import MultiModalConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MuseSparkConfig

FLAG = "VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP"
MIN_TOKENS = "VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_MIN_TOKENS"
UBATCHES = "VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_UBATCHES"
SP_MIN_TOKENS = "VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS"


def _config(**parallel) -> SimpleNamespace:
    fields = dict(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
        use_ubatching=False,
        ubatch_size=0,
        dbo_prefill_token_threshold=512,
        dbo_decode_token_threshold=32,
    )
    fields.update(parallel)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**fields),
        speculative_config=None,
        lora_config=None,
    )


@pytest.fixture
def sm120_platform(monkeypatch):
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True, is_device_capability_family=lambda f: f == 120
        ),
        raising=False,
    )


def test_prefill_overlap_is_registered_for_both_muse_architectures() -> None:
    assert MODELS_CONFIG_MAP["MuseSpark1xForCausalLM"] is MuseSparkConfig
    assert MODELS_CONFIG_MAP["MuseSparkForConditionalGeneration"] is MuseSparkConfig


def test_prefill_overlap_is_on_by_default(monkeypatch, sm120_platform) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    config = _config()
    MuseSparkConfig.verify_and_update_config(config)
    pc = config.parallel_config
    assert pc.ubatch_size == 2
    # The default is the floor: both halves stay above the thresholds of the model.
    assert pc.dbo_prefill_token_threshold == 2 * (1024 + 1)
    # A decode batch is never split: its threshold cannot be reached.
    assert pc.dbo_decode_token_threshold == 2**31 - 1


def test_prefill_overlap_can_be_turned_off(monkeypatch, sm120_platform) -> None:
    monkeypatch.setenv(FLAG, "0")
    config = _config()
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.ubatch_size == 0
    assert config.parallel_config.dbo_prefill_token_threshold == 512


def test_prefill_overlap_threshold_cannot_undercut_the_model_thresholds(
    monkeypatch, sm120_platform
) -> None:
    """Both halves must take the same model path, so they stay above its thresholds."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(MIN_TOKENS, "64")
    config = _config()
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.dbo_prefill_token_threshold == 2 * (1024 + 1)

    monkeypatch.setenv(SP_MIN_TOKENS, "3000")
    config = _config()
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.dbo_prefill_token_threshold == 2 * (3000 + 1)

    monkeypatch.setenv(MIN_TOKENS, "100000")
    config = _config()
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.dbo_prefill_token_threshold == 100000


def test_prefill_overlap_microbatch_count(monkeypatch, sm120_platform) -> None:
    """Every microbatch must stay above the token thresholds of the model."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv(UBATCHES, "3")
    monkeypatch.setenv(MIN_TOKENS, "64")
    config = _config()
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.ubatch_size == 3
    assert config.parallel_config.dbo_prefill_token_threshold == 3 * (1024 + 1)

    # One microbatch is no microbatching: the count is at least two.
    monkeypatch.setenv(UBATCHES, "1")
    config = _config()
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.ubatch_size == 2


@pytest.mark.parametrize(
    "parallel",
    [
        dict(tensor_parallel_size=1),
        dict(enable_expert_parallel=False),
        dict(pipeline_parallel_size=2),
        dict(data_parallel_size=2),
        dict(use_ubatching=True, ubatch_size=2),
    ],
    ids=["tp1", "no-ep", "pp2", "dp2", "already-microbatched"],
)
def test_prefill_overlap_needs_the_replicated_ep_layout(
    monkeypatch, sm120_platform, parallel
) -> None:
    monkeypatch.setenv(FLAG, "1")
    config = _config(**parallel)
    before = dict(vars(config.parallel_config))
    assert MuseSparkConfig._prefill_overlap_blocker(config) is not None
    MuseSparkConfig.verify_and_update_config(config)
    assert vars(config.parallel_config) == before


@pytest.mark.parametrize("field", ["speculative_config", "lora_config"])
def test_prefill_overlap_is_off_with_speculative_decoding_and_lora(
    monkeypatch, sm120_platform, field
) -> None:
    monkeypatch.setenv(FLAG, "1")
    config = _config()
    setattr(config, field, object())
    assert MuseSparkConfig._prefill_overlap_blocker(config) is not None


def test_prefill_overlap_unsupported_layout_is_quiet_unless_asked(
    monkeypatch, sm120_platform, caplog
) -> None:
    config = _config(tensor_parallel_size=1)
    monkeypatch.delenv(FLAG, raising=False)
    with caplog.at_level("WARNING"):
        MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.ubatch_size == 0
    assert "PREFILL_OVERLAP" not in caplog.text


def test_prefill_overlap_is_for_sm120_only(monkeypatch) -> None:
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setattr(
        vllm.platforms,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: True, is_device_capability_family=lambda f: f == 100
        ),
        raising=False,
    )
    config = _config()
    assert MuseSparkConfig._prefill_overlap_blocker(config) == "the device is not SM120"
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.ubatch_size == 0


def test_prefill_overlap_respects_the_v1_runner(monkeypatch, sm120_platform) -> None:
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
    assert MuseSparkConfig._prefill_overlap_blocker(_config()) is not None


@pytest.mark.parametrize(
    ("limits", "language_model_only", "disabled"),
    [
        ({"image": 0, "audio": 0, "video": 0}, False, True),
        ({"image": 0, "video": 1}, False, False),
        ({}, False, False),
        ({}, True, True),
    ],
)
def test_dbo_multimodal_check_only_counts_enabled_modalities(
    limits, language_model_only, disabled
) -> None:
    mm_config = MultiModalConfig(
        language_model_only=language_model_only, limit_per_prompt=limits
    )
    model_config = SimpleNamespace(multimodal_config=mm_config)
    assert VllmConfig._mm_inputs_disabled(model_config) is disabled
    assert VllmConfig._mm_inputs_disabled(SimpleNamespace(multimodal_config=None))
