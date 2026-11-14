# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark verification of speculative drafts from the top-K logits of every TP rank:
the model side, on 8 ranks with the real collectives.

``compute_topk_logits`` with ``k`` keeps the K (1 for a greedy step) largest logits of
each rank's lm_head shard, gathers them and soft-caps them; ``topk_verify`` verifies the
drafts of every request from them. The reference is the full path of the verify step:
``compute_logits`` (the real gather), ``apply_temperature``, ``apply_top_k_top_p`` and
``rejection_sample`` (block method with draft logits). The tokens must be those of the
reference (greedy bit for bit, sampled rows with the same draws), identical on every
rank, run to run, and from a CUDA graph (no host synchronization).
"""

import functools
import json
import os

import pytest
import torch
import torch.multiprocessing as mp

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for top-k verify tests", allow_module_level=True)

from test_muse_spark_topk_logits import VOCAB, fake_model

from vllm.models.muse_spark.nvidia.model import MuseSpark1xForCausalLM

HIDDEN = 512
STEPS = 7
# (requests, scenario): a scenario fixes the temperatures and top_k of the requests.
CASES = (
    (2, "greedy"),
    (16, "greedy"),
    (16, "t1k64"),
    (16, "mixed"),
    (64, "mixed"),
)


def _hook(model, hidden, k, rows, sampling, draft_logits, sampled_rows):
    """The model's part of a verify step: the candidates, then the verification."""
    from vllm.v1.worker.gpu.spec_decode.topk_rejection import topk_verify

    draft_sampled, pos, cu, idx_mapping, expanded_idx, local_pos = rows
    temperature, top_k, seeds = sampling
    topk = MuseSpark1xForCausalLM.compute_topk_logits(model, hidden, k=k)
    return topk_verify(
        topk,
        draft_sampled,
        pos,
        cu,
        idx_mapping,
        expanded_idx,
        local_pos,
        temperature,
        top_k,
        seeds,
        STEPS,
        draft_logits=draft_logits,
        has_sampled=sampled_rows,
    )


def _verify_worker(rank: int, world: int, port: int, out_path: str) -> None:
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
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )

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
    weight = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda") * (6 / HIDDEN**0.5)
    start = lm_head.shard_indices.padded_org_vocab_start_index
    lm_head.weight.data.copy_(
        weight[start : start + lm_head.num_embeddings_per_partition]
    )
    model = fake_model(
        lm_head=lm_head, logits_processor=logits_processor, topk_logits_k=64
    )

    results = {}
    for num_reqs, scenario in CASES:
        rows_per = STEPS + 1
        rows = num_reqs * rows_per
        hidden = torch.randn(rows, HIDDEN, generator=g, device="cuda")
        hidden = hidden.to(torch.bfloat16)
        if scenario == "greedy":
            temp_list = [0.0] * num_reqs
        elif scenario == "t1k64":
            temp_list = [1.0] * num_reqs
        else:
            temp_list = [0.0, 1.0, 0.7, 1.3][:4] * (num_reqs // 4)
        temperature = torch.tensor(temp_list, dtype=torch.float32, device="cuda")
        top_k = torch.where(temperature == 0.0, VOCAB, 64).to(torch.int32)
        sampled_rows = bool((temperature != 0.0).any())
        k = 64 if sampled_rows else 1
        seeds = torch.randint(
            -(1 << 60), 1 << 60, (num_reqs,), generator=g, device="cuda"
        )
        idx_mapping = torch.arange(num_reqs, dtype=torch.int32, device="cuda")
        expanded_idx = idx_mapping.repeat_interleave(rows_per)
        row_idx = torch.arange(rows, device="cuda")
        local_pos = (row_idx % rows_per).to(torch.int32)
        req_of_row = row_idx // rows_per
        cu = torch.arange(num_reqs + 1, dtype=torch.int32, device="cuda") * rows_per
        base_pos = torch.randint(1, 5000, (num_reqs,), generator=g, device="cuda")
        pos = base_pos[req_of_row] + local_pos

        with torch.inference_mode():
            full = MuseSpark1xForCausalLM.compute_logits(model, hidden)
            # Drafts: the argmax with probability 3/4 per step, else one of the 8 best
            # tokens; the draft distribution lives on those 8 (as the DFlash2 cache).
            top = full.topk(8, dim=-1).indices
            vals = full.gather(1, top) + 0.3 * torch.randn(
                top.shape, generator=g, device="cuda"
            )
            pick = torch.multinomial(vals.softmax(-1), 1, generator=g).squeeze(1)
            pick = torch.where(
                torch.rand(rows, generator=g, device="cuda") < 0.75,
                torch.zeros_like(pick),
                pick,
            )
            draft_token = top.gather(1, pick[:, None]).squeeze(1)
            draft_sampled = torch.zeros(rows, dtype=torch.int64, device="cuda")
            not_last = local_pos < STEPS
            draft_sampled[row_idx[not_last] + 1] = draft_token[not_last]
            draft_logits = torch.full(
                (num_reqs, STEPS, VOCAB), float("-inf"), device="cuda"
            )
            for c in range(top.shape[1]):
                draft_logits[
                    req_of_row[not_last], local_pos[not_last].long(), top[not_last, c]
                ] = vals[not_last, c]

            logits = full.clone()
            apply_temperature(logits, expanded_idx, temperature)
            logits = apply_top_k_top_p(logits, top_k[expanded_idx.long()], None)
            ref_sampled, ref_num = rejection_sample(
                logits,
                draft_logits,
                draft_sampled,
                cu,
                pos,
                idx_mapping,
                expanded_idx,
                local_pos,
                temperature,
                seeds,
                STEPS,
                None,
                use_fp64=False,
                use_block_verification=True,
            )

            hook = functools.partial(
                _hook,
                model,
                hidden,
                k,
                (draft_sampled, pos, cu, idx_mapping, expanded_idx, local_pos),
                (temperature, top_k, seeds),
                draft_logits,
                sampled_rows,
            )

            first_sampled, first_num = hook()
            second_sampled, second_num = hook()
            valid = torch.arange(STEPS + 1, device="cuda")[None, :] < first_num[:, None]
            packed = torch.where(
                valid, first_sampled, torch.full_like(first_sampled, -1)
            )
            everyone = tensor_model_parallel_all_gather(packed.view(-1), 0).view(
                world, -1
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with (
                graph_capture(device=torch.device("cuda", rank)) as capture,
                torch.cuda.graph(graph, stream=capture.stream),
            ):
                captured_sampled, captured_num = hook()
            graph.replay()
            torch.cuda.synchronize()

        ref_valid = torch.arange(STEPS + 1, device="cuda")[None, :] < ref_num[:, None]
        ref_packed = torch.where(
            ref_valid, ref_sampled, torch.full_like(ref_sampled, -1)
        )
        same_requests = (ref_packed == packed).all(dim=1)
        greedy = temperature == 0.0
        results[f"{scenario}_{num_reqs}"] = dict(
            greedy_equal=bool(same_requests[greedy].all()) if greedy.any() else True,
            # Sampled rows: the draws of the full path, up to fp32 rounding of the sums.
            sampled_share=float(same_requests[~greedy].float().mean())
            if (~greedy).any()
            else 1.0,
            lengths_equal_share=float((ref_num == first_num).float().mean()),
            deterministic=bool(
                torch.equal(first_sampled[valid], second_sampled[valid])
                and torch.equal(first_num, second_num)
            ),
            same_on_all_ranks=bool((everyone == everyone[0]).all()),
            graph_equal=bool(
                torch.equal(captured_num, first_num)
                and torch.equal(captured_sampled[valid], first_sampled[valid])
            ),
            in_vocab=bool(((packed[valid] >= 0) & (packed[valid] < VOCAB)).all()),
            max_accepted=int(first_num.max()) - 1,
        )
        torch.distributed.barrier()
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(results, f)
    torch.distributed.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires 8 GPUs")
def test_the_verify_hook_matches_the_full_path_on_8_ranks(tmp_path):
    """Real lm_head shards and collectives: the tokens verified from the gathered
    candidates are those of ``compute_logits`` + the full rejection route (greedy
    requests bit for bit, sampled requests with the same draws), identical on all
    ranks, run-to-run, and from a CUDA graph."""
    out_path = str(tmp_path / "results.json")
    port = 29600 + os.getpid() % 1000
    mp.spawn(_verify_worker, args=(8, port, out_path), nprocs=8)
    with open(out_path) as f:
        results = json.load(f)
    assert set(results) == {f"{s}_{n}" for n, s in CASES}
    for name, result in results.items():
        flags = {key: value for key, value in result.items() if isinstance(value, bool)}
        assert all(flags.values()), (name, result)
        # The draws of the two paths agree except at fp32 rounding of the sums.
        assert result["sampled_share"] >= 0.97, (name, result)
        assert result["lengths_equal_share"] >= 0.97, (name, result)
