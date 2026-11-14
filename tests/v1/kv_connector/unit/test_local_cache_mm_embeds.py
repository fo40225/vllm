# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A local KV cache connector (CPU offload) must not turn on the consumer-side
multimodal embedding machinery of a text-only deployment.

`kv_role=kv_both` makes `KVTransferConfig.is_kv_consumer` true. For a
disaggregated-prefill consumer that switches on `enable_mm_embeds` (the frontend
accepts embedding inputs without the tensors); for a local cache it only costs
memory: the V2 model runner then builds the encoder runner, whose
`[max_num_batched_tokens, hidden]` embedding buffer is a persistent GPU
allocation (128 MiB at 8192 x 8192 bf16), and the model runs from input embeddings.
"""

from types import SimpleNamespace

import pytest

from vllm.config import KVTransferConfig, VllmConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1

pytestmark = pytest.mark.cpu_test


def _connector_cls(name: str):
    return KVConnectorFactory.get_connector_class(
        KVTransferConfig(kv_connector=name, kv_role="kv_both")
    )


def test_base_and_transfer_connectors_receive_remote_kv():
    assert KVConnectorBase_V1.receives_remote_kv({}) is True
    assert _connector_cls("ExampleConnector").receives_remote_kv({}) is True


@pytest.mark.parametrize("name", ["SimpleCPUOffloadConnector", "OffloadingConnector"])
def test_local_cache_connectors_do_not_receive_remote_kv(name):
    assert _connector_cls(name).receives_remote_kv({}) is False


def _multi(*names: str) -> dict:
    return {
        "connectors": [{"kv_connector": n, "kv_role": "kv_both"} for n in names],
    }


def test_multi_connector_receives_remote_kv_if_any_child_does():
    multi = _connector_cls("MultiConnector")
    assert multi.receives_remote_kv(_multi("SimpleCPUOffloadConnector")) is False
    assert (
        multi.receives_remote_kv(
            _multi("SimpleCPUOffloadConnector", "OffloadingConnector")
        )
        is False
    )
    mixed = _multi("SimpleCPUOffloadConnector", "ExampleConnector")
    assert multi.receives_remote_kv(mixed) is True
    # No children configured: the conservative answer (unchanged behaviour).
    assert multi.receives_remote_kv({}) is True


def _ktc(name: str, role: str) -> KVTransferConfig:
    return KVTransferConfig(kv_connector=name, kv_role=role)


@pytest.mark.parametrize(
    "kv_config,expected",
    [
        (_ktc("SimpleCPUOffloadConnector", "kv_both"), False),
        (_ktc("OffloadingConnector", "kv_both"), False),
        (_ktc("ExampleConnector", "kv_both"), True),
        (_ktc("ExampleConnector", "kv_consumer"), True),
        # An unregistered connector keeps the old behaviour (consumer reads).
        (_ktc("NoSuchConnector", "kv_both"), True),
    ],
)
def test_vllm_config_helper(kv_config, expected):
    assert VllmConfig._kv_connector_receives_remote_kv(kv_config) is expected


def _fake_config(kv_config: KVTransferConfig | None, language_model_only: bool = False):
    mm_config = SimpleNamespace(
        allow_missing_mm_embeddings=False,
        enable_mm_embeds=False,
        language_model_only=language_model_only,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(multimodal_config=mm_config),
        ec_transfer_config=None,
        kv_transfer_config=kv_config,
        _kv_connector_receives_remote_kv=VllmConfig._kv_connector_receives_remote_kv,
    ), mm_config


@pytest.mark.parametrize(
    "kv_config,allow_missing",
    [
        (None, False),
        (_ktc("SimpleCPUOffloadConnector", "kv_both"), False),
        (_ktc("OffloadingConnector", "kv_both"), False),
        (_ktc("ExampleConnector", "kv_both"), True),
        (_ktc("ExampleConnector", "kv_consumer"), True),
        (_ktc("ExampleConnector", "kv_producer"), False),
    ],
)
def test_resolve_mm_embedding_inputs(kv_config, allow_missing):
    fake, mm_config = _fake_config(kv_config)
    VllmConfig._resolve_mm_embedding_inputs(fake)
    assert mm_config.allow_missing_mm_embeddings is allow_missing
    # enable_mm_embeds follows: it is only forced on together with the missing tensors.
    assert mm_config.enable_mm_embeds is allow_missing


def test_user_enable_mm_embeds_survives_for_a_local_cache():
    fake, mm_config = _fake_config(_ktc("SimpleCPUOffloadConnector", "kv_both"))
    mm_config.enable_mm_embeds = True
    VllmConfig._resolve_mm_embedding_inputs(fake)
    assert mm_config.enable_mm_embeds is True
    assert mm_config.allow_missing_mm_embeddings is False


@pytest.mark.parametrize(
    "name,enable_mm_embeds,allow_missing",
    [
        # v0.33.0: a language-model-only transfer consumer stays text-only ...
        ("ExampleConnector", False, False),
        # ... unless the user opts in with --enable-mm-embeds.
        ("ExampleConnector", True, True),
        # A local cache never receives remote KV, opt-in or not.
        ("SimpleCPUOffloadConnector", False, False),
        ("SimpleCPUOffloadConnector", True, False),
    ],
)
def test_language_model_only_consumer(name, enable_mm_embeds, allow_missing):
    fake, mm_config = _fake_config(_ktc(name, "kv_both"), language_model_only=True)
    mm_config.enable_mm_embeds = enable_mm_embeds
    VllmConfig._resolve_mm_embedding_inputs(fake)
    assert mm_config.allow_missing_mm_embeddings is allow_missing
    assert mm_config.enable_mm_embeds is (enable_mm_embeds or allow_missing)
