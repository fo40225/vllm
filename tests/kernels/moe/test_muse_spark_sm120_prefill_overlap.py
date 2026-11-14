# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbatched prefill of MuseSpark on SM120: when the launch config turns it on,
what keeps it off, the DBO support list it relies on, and how the V2 runner treats
the microbatches (threshold, warm-up, memory profile)."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

import vllm.platforms
from vllm.config.multimodal import MultiModalConfig
from vllm.config.vllm import VllmConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MuseSparkConfig

FLAG = "VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP"
MIN_TOKENS = "VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_MIN_TOKENS"
UBATCHES = "VLLM_MUSE_SPARK_SM120_PREFILL_OVERLAP_UBATCHES"
SP_MIN_TOKENS = "VLLM_MUSE_SPARK_SM120_SEQUENCE_PARALLEL_MIN_TOKENS"


def _config(dbo_unsupported=(), **parallel) -> SimpleNamespace:
    fields = dict(
        tensor_parallel_size=8,
        pipeline_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=True,
        use_ubatching=False,
        ubatch_size=0,
        ubatch_overlap_tp_collectives=False,
        dbo_prefill_token_threshold=512,
        dbo_decode_token_threshold=32,
    )
    fields.update(parallel)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(**fields),
        speculative_config=None,
        lora_config=None,
        _get_dbo_unsupported_features=lambda tp_collective_overlap=None: list(
            dbo_unsupported
        ),
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
    assert pc.ubatch_overlap_tp_collectives
    # The default is the floor: both halves stay above the thresholds of the model.
    assert pc.dbo_prefill_token_threshold == 2 * (1024 + 1)
    # A decode batch is never split: its threshold cannot be reached.
    assert pc.dbo_decode_token_threshold == 2**31 - 1


def test_prefill_overlap_can_be_turned_off(monkeypatch, sm120_platform) -> None:
    monkeypatch.setenv(FLAG, "0")
    config = _config()
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.ubatch_size == 0
    assert not config.parallel_config.ubatch_overlap_tp_collectives
    assert config.parallel_config.dbo_prefill_token_threshold == 512


def test_prefill_overlap_threshold_cannot_undercut_the_launch_floor(
    monkeypatch, sm120_platform
) -> None:
    """The halves stay above the sequence-parallel minimum known at launch."""
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


def test_prefill_overlap_is_off_in_batch_invariant_mode(
    monkeypatch, sm120_platform
) -> None:
    """A split changes where a request's rounding happens: not batch invariant."""
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("VLLM_BATCH_INVARIANT", "1")
    config = _config()
    assert (
        MuseSparkConfig._prefill_overlap_blocker(config)
        == "batch-invariant mode is on"
    )
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.ubatch_size == 0
    assert not config.parallel_config.ubatch_overlap_tp_collectives


@pytest.mark.parametrize(
    "unsupported",
    [
        ["dual batch overlap with multimodal models"],
        ["dual batch overlap with PIECEWISE CUDA graphs"],
        ["dual batch overlap with context parallelism"],
    ],
)
def test_prefill_overlap_never_changes_the_runner_the_launch_gets(
    monkeypatch, sm120_platform, unsupported
) -> None:
    """Whatever the V2 runner cannot microbatch keeps the flag off: turning
    microbatching on there would send the launch to the V1 runner."""
    monkeypatch.setenv(FLAG, "1")
    config = _config(dbo_unsupported=unsupported)
    blocker = MuseSparkConfig._prefill_overlap_blocker(config)
    assert blocker is not None and unsupported[0] in blocker
    MuseSparkConfig.verify_and_update_config(config)
    assert config.parallel_config.ubatch_size == 0
    assert not config.parallel_config.ubatch_overlap_tp_collectives


# ----- the DBO support list --------------------------------------------------------


@pytest.mark.parametrize(
    ("limits", "language_model_only", "disabled"),
    [
        ({"image": 0, "audio": 0, "video": 0}, False, True),
        # An unmentioned modality is not capped (999 per prompt).
        ({"image": 0}, False, False),
        ({"image": 0, "video": 0}, False, False),
        ({"image": 0, "video": 1}, False, False),
        ({"image": 1, "audio": 0, "video": 0}, False, False),
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


def test_dbo_multimodal_check_looks_at_the_modalities_the_model_supports(
    monkeypatch,
) -> None:
    """The modalities to check are the model's, not the keys the user gave."""
    from vllm.multimodal import MULTIMODAL_REGISTRY

    mm_config = MultiModalConfig(limit_per_prompt={"image": 0})
    model_config = SimpleNamespace(multimodal_config=mm_config)
    supported = SimpleNamespace(supported_mm_limits={"image": None})
    monkeypatch.setattr(
        MULTIMODAL_REGISTRY, "get_processing_info", lambda config: supported
    )
    assert VllmConfig._mm_inputs_disabled(model_config) is True
    supported.supported_mm_limits = {"image": None, "video": None}
    assert VllmConfig._mm_inputs_disabled(model_config) is False


def _verify_stub(**kwargs):
    fields = dict(
        ubatch_overlap_tp_collectives=True, ubatch_size=2, use_ubatching=True
    )
    pc = SimpleNamespace(**fields)
    return SimpleNamespace(
        parallel_config=pc,
        _get_dbo_unsupported_features=lambda: kwargs.get("dbo", []),
        use_v2_model_runner=kwargs.get("v2", True),
    )


def test_late_config_changes_drop_the_overlap_instead_of_the_runner() -> None:
    """Whatever the model hook could not see (a CUDA graph mode or a runner that
    settled later) turns the overlap off in the finished config."""
    for kwargs in (
        dict(dbo=["dual batch overlap with PIECEWISE CUDA graphs"]),
        dict(v2=False),
    ):
        stub = _verify_stub(**kwargs)
        VllmConfig._verify_tp_collective_overlap(stub)
        assert not stub.parallel_config.ubatch_overlap_tp_collectives
        assert stub.parallel_config.ubatch_size == 0

    stub = _verify_stub()
    VllmConfig._verify_tp_collective_overlap(stub)
    assert stub.parallel_config.ubatch_overlap_tp_collectives
    assert stub.parallel_config.ubatch_size == 2


# ----- the V2 runner ---------------------------------------------------------------


def _parallel(**fields):
    base = dict(
        use_ubatching=True,
        num_ubatches=2,
        dbo_prefill_token_threshold=2050,
        dbo_decode_token_threshold=2**31 - 1,
        ubatch_overlap_tp_collectives=True,
        data_parallel_size=1,
    )
    base.update(fields)
    return SimpleNamespace(**base)


def test_threshold_follows_the_sequence_parallel_minimum_of_the_model() -> None:
    """With a larger CUDA graph range the sequence-parallel minimum rises, and a
    batch is split only where both halves still shard the stream."""
    from vllm.models.muse_spark.nvidia.model import (
        MuseSpark1xForCausalLM,
        sm120_sequence_parallel_min_tokens,
    )
    from vllm.v1.worker.gpu.ubatch_utils import fit_prefill_threshold_to_model

    def model_for(max_graph_tokens):
        compilation_config = SimpleNamespace(
            max_cudagraph_capture_size=max_graph_tokens
        )
        config = SimpleNamespace(compilation_config=compilation_config)
        sp_min = sm120_sequence_parallel_min_tokens(config)
        stub = SimpleNamespace(model=SimpleNamespace(sp_min_tokens=sp_min))
        return sp_min, MuseSpark1xForCausalLM.min_tokens_per_ubatch.fget(stub)

    sp_min, per_ubatch = model_for(2048)
    assert sp_min == per_ubatch == 2049
    parallel = _parallel()
    fit_prefill_threshold_to_model(
        parallel, SimpleNamespace(min_tokens_per_ubatch=per_ubatch)
    )
    # (2048, 2049) tokens would have put one half on each side of the minimum.
    assert parallel.dbo_prefill_token_threshold == 2 * 2049

    # A small graph range leaves the launch floor alone.
    _, per_ubatch = model_for(256)
    parallel = _parallel()
    fit_prefill_threshold_to_model(
        parallel, SimpleNamespace(min_tokens_per_ubatch=per_ubatch)
    )
    assert parallel.dbo_prefill_token_threshold == 2050


def test_threshold_is_left_alone_for_a_model_without_a_minimum() -> None:
    from vllm.models.muse_spark.nvidia.model import MuseSpark1xForCausalLM
    from vllm.v1.worker.gpu.ubatch_utils import fit_prefill_threshold_to_model

    for model in (
        object(),
        SimpleNamespace(min_tokens_per_ubatch=0),
        SimpleNamespace(min_tokens_per_ubatch="x"),
        SimpleNamespace(
            min_tokens_per_ubatch=MuseSpark1xForCausalLM.min_tokens_per_ubatch.fget(
                SimpleNamespace(model=SimpleNamespace(sp_min_tokens=None))
            )
        ),
    ):
        parallel = _parallel()
        fit_prefill_threshold_to_model(parallel, model)
        assert parallel.dbo_prefill_token_threshold == 2050


def test_data_parallel_free_microbatching_needs_the_model_switch() -> None:
    """--enable-dbo at DP=1 stays what it was (no runner) for every other model."""
    from vllm.v1.worker.gpu import ubatch_utils

    vllm_config = SimpleNamespace(
        parallel_config=_parallel(ubatch_overlap_tp_collectives=False)
    )
    with patch.object(ubatch_utils, "UBatchRunner") as runner_cls:
        assert (
            ubatch_utils.maybe_build_ubatch_runner(
                vllm_config, None, None, [], None, 8
            )
            is None
        )
        runner_cls.assert_not_called()

        vllm_config.parallel_config.ubatch_overlap_tp_collectives = True
        assert ubatch_utils.maybe_build_ubatch_runner(
            vllm_config, None, None, [], None, 8
        ) is runner_cls.return_value

        vllm_config.parallel_config.use_ubatching = False
        assert (
            ubatch_utils.maybe_build_ubatch_runner(
                vllm_config, None, None, [], None, 8
            )
            is None
        )


@pytest.mark.parametrize(
    ("switch", "num_tokens", "split"),
    [
        (True, 4096, True),
        (True, 2050, True),
        (True, 2049, False),
        (False, 4096, False),
    ],
)
def test_dispatch_splits_only_big_batches_and_only_on_the_model_switch(
    switch, num_tokens, split
) -> None:
    from vllm.v1.worker.gpu.dp_utils import dispatch_cg_and_sync_dp

    batch_desc, sync = dispatch_cg_and_sync_dp(
        None,
        num_reqs=4,
        num_tokens=num_tokens,
        uniform_token_count=None,
        dp_size=1,
        dp_rank=0,
        need_eager=True,
        parallel_config=_parallel(ubatch_overlap_tp_collectives=switch),
        allow_ubatching=True,
        uniform_decode=False,
    )
    assert sync is None
    assert batch_desc.num_ubatches == (2 if split else 1)


@pytest.mark.skipif(
    "MUSE_R2E_REAL_CONFIG" not in os.environ,
    reason="needs the Muse checkpoint config in the Hugging Face cache and 8 GPUs "
    "visible (set MUSE_R2E_REAL_CONFIG=1 on the server)",
)
@pytest.mark.parametrize(
    ("scenario", "kwargs", "env", "overlap"),
    [
        ("zero-limits", dict(limit_mm_per_prompt="ZERO"), {}, True),
        ("no-limits", {}, {}, False),
        ("image-only", dict(limit_mm_per_prompt={"image": 0}), {}, False),
        (
            "piecewise",
            dict(
                limit_mm_per_prompt="ZERO",
                compilation_config={"cudagraph_mode": "PIECEWISE"},
            ),
            {},
            False,
        ),
        (
            "batch-invariant",
            dict(limit_mm_per_prompt="ZERO"),
            {"VLLM_BATCH_INVARIANT": "1"},
            False,
        ),
        (
            "explicit-v2-no-limits",
            {},
            {"VLLM_USE_V2_MODEL_RUNNER": "1"},
            False,
        ),
    ],
)
def test_real_config_the_overlap_flag_never_changes_the_runner(
    monkeypatch, scenario, kwargs, env, overlap
) -> None:
    """On the real Muse config (TP8, EP8): the flag is on exactly where the V2
    runner can microbatch, and the launch gets the same runner with it off."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    def build(flag):
        monkeypatch.setenv(FLAG, flag)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        kw = dict(
            model="unsloth/Muse-Spark-Open-816B-A42B-NVFP4",
            load_format="dummy",
            tensor_parallel_size=8,
            max_model_len=131072,
            enable_expert_parallel=True,
            max_num_seqs=128,
            kv_cache_dtype="fp8",
            quantization="modelopt_fp4",
        )
        kw.update(kwargs)
        if kw.get("limit_mm_per_prompt") == "ZERO":
            kw["limit_mm_per_prompt"] = {"image": 0, "audio": 0, "video": 0}
        return EngineArgs(**kw).create_engine_config(UsageContext.LLM_CLASS)

    on = build("1")
    off = build("0")
    assert on.parallel_config.ubatch_overlap_tp_collectives is overlap
    assert on.parallel_config.use_ubatching is overlap
    assert on.use_v2_model_runner == off.use_v2_model_runner
