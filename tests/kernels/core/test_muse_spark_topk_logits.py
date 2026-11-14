# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark sampling from the top-K logits of every TP rank: the model side.

``compute_topk_logits`` keeps the K largest logits of each rank's lm_head shard,
gathers them and soft-caps them with the op of ``compute_logits``. The sampler draws
the token from the candidates; the test of that draw is
tests/v1/worker/test_gpu_topk_logits.py. Here: the soft cap gives a candidate the bits
the full path gives its logit, the launch gate, and (8 GPUs) the whole model hook on
real collectives against ``compute_logits``, with CUDA graphs and across ranks.
"""

import functools
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.multiprocessing as mp

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for top-k logits tests", allow_module_level=True)

from vllm.models.muse_spark.common.layers import MulTanhSoftCap
from vllm.models.muse_spark.nvidia import model as muse_model
from vllm.models.muse_spark.nvidia.model import MuseSpark1xForCausalLM
from vllm.v1.worker.gpu.sample.topk_logits import local_topk_candidates

FLAG = "VLLM_MUSE_SPARK_SM120_TOPK_LOGITS"
FLAG_K = "VLLM_MUSE_SPARK_SM120_TOPK_LOGITS_K"
VOCAB = 202048
MULTIPLIER = 0.17677669529663687
SOFT_CAP = 20.0


def fake_model(cap_module: bool = True, **overrides):
    """The members of MuseSpark1xForCausalLM that the logits methods read."""
    model = SimpleNamespace(
        mul_tanh_soft_cap=MulTanhSoftCap(MULTIPLIER, SOFT_CAP) if cap_module else None,
        metap_output_multiplier=MULTIPLIER,
        output_soft_cap_temp=SOFT_CAP,
        **overrides,
    )
    model._scale_and_soft_cap = functools.partial(
        MuseSpark1xForCausalLM._scale_and_soft_cap, model
    )
    return model


def all_bf16_values() -> torch.Tensor:
    bits = torch.arange(-32768, 32768, dtype=torch.int16, device="cuda")
    return bits.view(torch.bfloat16)


def bits_of(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.int32)


@pytest.mark.parametrize("cap_module", [True, False])
def test_the_helper_is_the_soft_cap_of_compute_logits(cap_module: bool):
    model = fake_model(cap_module)
    x = (torch.randn(16, 4096, device="cuda") * 30).to(torch.bfloat16)
    expected = SOFT_CAP * torch.tanh(x.float() * MULTIPLIER / SOFT_CAP)
    got = model._scale_and_soft_cap(x)
    assert got.dtype == torch.float32 and got.shape == x.shape
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("cap_module", [True, False])
def test_the_soft_cap_is_elementwise_for_every_bf16_value(cap_module: bool):
    """Whatever the shape, the alignment or the neighbours of an element, it gets the
    bits it gets in the [rows, vocab] logits: so a candidate's capped value is the
    full path's, for every value a logit can take."""
    model = fake_model(cap_module)
    values = all_bf16_values()
    reference = bits_of(model._scale_and_soft_cap(values.view(1, -1)))
    # As rows of the candidate words, with row lengths that are and are not vectors.
    for rows, width in ((512, 128), (256, 256), (1024, 64), (128, 512)):
        shaped = bits_of(model._scale_and_soft_cap(values.view(rows, width)))
        assert torch.equal(shaped.view(-1), reference.view(-1)), (rows, width)
    # The same values behind a start that is not vector aligned.
    padded = torch.cat([values[:3], values])[3:]
    assert padded.data_ptr() % 16 != 0
    shifted = bits_of(model._scale_and_soft_cap(padded.view(1, -1)))
    assert torch.equal(shifted.view(-1), reference.view(-1))


def shard_candidates(logits: torch.Tensor, tp: int, k: int) -> torch.Tensor:
    shard = logits.shape[1] // tp
    return torch.cat(
        [
            local_topk_candidates(logits[:, r * shard : (r + 1) * shard], k)
            for r in range(tp)
        ]
    ).view(torch.bfloat16)


@pytest.mark.parametrize("cap_module", [True, False])
def test_a_candidate_has_the_capped_value_of_its_logit(cap_module: bool):
    model = fake_model(cap_module)
    tp, k, rows = 8, 64, 48
    logits = (torch.randn(rows, VOCAB, device="cuda") * 3).to(torch.bfloat16)
    words = shard_candidates(logits, tp, k)
    capped = model._scale_and_soft_cap(words)
    full = model._scale_and_soft_cap(logits)
    shard = VOCAB // tp
    idx = words.view(torch.int16)[:, k:].to(torch.int64) & 0xFFFF
    for r in range(tp):
        block = slice(r * rows, (r + 1) * rows)
        token = idx[block] + r * shard
        expected = torch.gather(full, 1, token)
        assert torch.equal(bits_of(capped[block, :k]), bits_of(expected))


# --------------------------------------------------------------------------------
# The launch gate.
# --------------------------------------------------------------------------------


def head(shard=25256, added=0, dtype=torch.bfloat16):
    return SimpleNamespace(
        num_embeddings_per_partition=shard,
        num_added_embeddings=added,
        weight=SimpleNamespace(dtype=dtype),
    )


def resolve(monkeypatch, enabled=True, tp=8, k=None, last_pp=True, **kwargs):
    monkeypatch.setenv(FLAG, "1" if enabled else "0")
    if k is not None:
        monkeypatch.setenv(FLAG_K, str(k))
    monkeypatch.setattr(muse_model, "get_tensor_model_parallel_world_size", lambda: tp)
    monkeypatch.setattr(
        muse_model, "get_pp_group", lambda: SimpleNamespace(is_last_rank=last_pp)
    )
    model = SimpleNamespace(
        lm_head=kwargs.pop("lm_head", head()),
        logits_processor=SimpleNamespace(
            head_dtype=kwargs.pop("head_dtype", None),
            scale=kwargs.pop("scale", 1.0),
            soft_cap=kwargs.pop("soft_cap", None),
        ),
    )
    assert not kwargs
    return MuseSpark1xForCausalLM._resolve_topk_logits_k(model)


def test_the_flag_is_off_by_default(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.delenv(FLAG_K, raising=False)
    from vllm import envs

    assert envs.VLLM_MUSE_SPARK_SM120_TOPK_LOGITS is False
    assert envs.VLLM_MUSE_SPARK_SM120_TOPK_LOGITS_K == 64
    assert envs.VLLM_MUSE_SPARK_SM120_TOPK_LOGITS_MIN_TOKENS == 2


def test_the_gate_opens_for_the_deployed_layout(monkeypatch):
    assert resolve(monkeypatch) == 64
    assert resolve(monkeypatch, k=32) == 32
    assert resolve(monkeypatch, k=1) == 1


@pytest.mark.parametrize(
    "case",
    {
        "flag-off": dict(enabled=False),
        "not-last-pp-rank": dict(last_pp=False),
        "k-zero": dict(k=0),
        "k-too-large": dict(k=129),
        "tp1": dict(tp=1),
        "tp2": dict(tp=2, lm_head=head(101024)),
        "too-many-candidates": dict(tp=32, k=128, lm_head=head(6314)),
        "shard-narrower-than-k": dict(lm_head=head(40)),
        "added-embeddings": dict(lm_head=head(added=1)),
        "fp32-weight": dict(lm_head=head(dtype=torch.float32)),
        "fp32-head": dict(head_dtype=torch.float32),
        "scaled-processor": dict(scale=0.5),
        "capped-processor": dict(soft_cap=30.0),
    }.items(),
    ids=lambda item: item[0],
)
def test_the_gate_closes(monkeypatch, case):
    assert resolve(monkeypatch, **case[1]) == 0


# --------------------------------------------------------------------------------
# The model hook on 8 ranks.
# --------------------------------------------------------------------------------

HIDDEN = 512
ROWS = (1, 8, 64, 128)


def _hook_worker(rank: int, world: int, port: int, out_path: str) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
        tensor_model_parallel_all_gather,
    )
    from vllm.distributed.parallel_state import graph_capture
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature, gumbel_sample
    from vllm.v1.worker.gpu.sample.topk_logits import topk_merge_sample

    torch.cuda.set_device(rank)
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
    with set_current_vllm_config(config), torch.device("cuda"):
        lm_head = ParallelLMHead(
            VOCAB, HIDDEN, bias=False, params_dtype=torch.bfloat16, quant_config=None
        )
        logits_processor = LogitsProcessor(VOCAB)
    g = torch.Generator(device="cuda").manual_seed(7)
    weight = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda") * (3 / HIDDEN**0.5)
    start = lm_head.shard_indices.padded_org_vocab_start_index
    lm_head.weight.data.copy_(
        weight[start : start + lm_head.num_embeddings_per_partition]
    )
    model = fake_model(
        lm_head=lm_head, logits_processor=logits_processor, topk_logits_k=64
    )

    results = {}
    for rows in ROWS:
        hidden = torch.randn(rows, HIDDEN, generator=g, device="cuda")
        hidden = hidden.to(torch.bfloat16)
        idx = torch.arange(rows, device="cuda")
        temperature = torch.tensor(
            [0.0, 1.0] * (rows // 2) + [0.0] * (rows % 2),
            dtype=torch.float32,
            device="cuda",
        )
        top_k = torch.where(temperature == 0.0, VOCAB, 64).to(torch.int32)
        seeds = torch.randint(-(1 << 60), 1 << 60, (rows,), generator=g, device="cuda")
        pos = torch.randint(1, 5000, (rows,), generator=g, device="cuda")

        def hook():
            topk = MuseSpark1xForCausalLM.compute_topk_logits(model, hidden)
            return topk_merge_sample(topk, idx, temperature, top_k, seeds, pos)

        with torch.inference_mode():
            full = MuseSpark1xForCausalLM.compute_logits(model, hidden)
            logits = full.clone()
            apply_temperature(logits, idx, temperature)
            logits = apply_top_k_top_p(logits, top_k, None)
            expected = gumbel_sample(
                logits,
                idx,
                temperature,
                seeds,
                pos,
                apply_temperature=False,
                is_drafting=False,
            )
            first = hook()
            second = hook()
            # Every rank drew the same tokens.
            everyone = tensor_model_parallel_all_gather(first, 0).view(world, rows)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with graph_capture(device=torch.device("cuda", rank)) as capture:
                with torch.cuda.graph(graph, stream=capture.stream):
                    captured = hook()
            graph.replay()
            torch.cuda.synchronize()
        results[str(rows)] = dict(
            equal_full_path=bool(torch.equal(first, expected)),
            greedy_equal=bool(
                torch.equal(first[temperature == 0], expected[temperature == 0])
            ),
            deterministic=bool(torch.equal(first, second)),
            same_on_all_ranks=bool((everyone == everyone[0]).all()),
            graph_equal=bool(torch.equal(captured, first)),
            in_vocab=bool(((first >= 0) & (first < VOCAB)).all()),
        )
        torch.distributed.barrier()
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(results, f)
    torch.distributed.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires 8 GPUs")
def test_the_model_hook_matches_compute_logits_on_8_ranks(tmp_path):
    """The real lm_head shards and collectives: the token drawn from the gathered
    candidates is the token of ``compute_logits`` + the full sampler route, bit for bit
    (greedy rows and t1k64 rows), identical on all ranks, run-to-run, and from a CUDA
    graph."""
    out_path = str(tmp_path / "results.json")
    port = 29700 + os.getpid() % 1000
    mp.spawn(_hook_worker, args=(8, port, out_path), nprocs=8)
    with open(out_path) as f:
        results = json.load(f)
    assert set(results) == {str(rows) for rows in ROWS}
    for rows, result in results.items():
        assert all(result.values()), (rows, result)
