# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MuseSpark verification of a step whose requests take different routes (levels 3 and 4
of VLLM_MUSE_SPARK_SM120_VERIFY_TOPK_LOGITS): the model side, on 8 ranks with the real
collectives.

``compute_split_verify_logits`` runs ONE lm_head projection over all the rows, gathers
the top-K candidates of the rows of the candidate requests and the whole logits of the
rows of the others. The candidate requests are verified by ``topk_verify`` (min_p and
top_p included), the others by ``rejection_sample`` on the gathered logits, on views of
the batch; ``merge_outputs`` puts them back in the batch order. The reference is the
full path of the whole batch: ``compute_logits``, ``apply_temperature``,
``apply_min_p``, ``apply_top_k_top_p`` and ``rejection_sample``. The tokens must be
those of the reference (greedy bit for bit, the others with the same draws), identical
on every rank and run to run.
"""

import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for split verify tests", allow_module_level=True)

from test_muse_spark_topk_logits import VOCAB, fake_model

from vllm.models.muse_spark.nvidia.model import MuseSpark1xForCausalLM

HIDDEN = 512
STEPS = 7
# (requests): the requests cycle through the kinds below.
CASES = (16, 64)
# kind: (temperature, top_k, min_p, top_p, route)
KINDS = (
    (0.0, VOCAB, 0.0, 1.0, "candidates"),  # greedy
    (1.0, 64, 0.0, 1.0, "candidates"),  # plain
    (1.0, 64, 0.1, 1.0, "candidates"),  # min_p
    (0.7, 64, 0.0, 0.9, "candidates"),  # top_p
    (1.0, 200, 0.0, 1.0, "full"),  # top_k above K
    (1.0, 64, 0.0, 0.95, "full"),  # top_p kept on the logits (level 3)
)


def _run_case(world: int, model, g: torch.Generator, num_reqs: int) -> dict:
    from vllm.distributed import tensor_model_parallel_all_gather
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
    from vllm.v1.worker.gpu.sample.gumbel import apply_temperature
    from vllm.v1.worker.gpu.sample.min_p import apply_min_p
    from vllm.v1.worker.gpu.sample.output import SamplerOutput
    from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
        rejection_sample,
    )
    from vllm.v1.worker.gpu.spec_decode.topk_rejection import topk_verify
    from vllm.v1.worker.gpu.spec_decode.verify_split import (
        SplitIndices,
        merge_outputs,
        sampler_view,
    )

    rows_per = STEPS + 1
    rows = num_reqs * rows_per
    hidden = torch.randn(rows, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)
    kinds = [r % len(KINDS) for r in range(num_reqs)]
    temperature = torch.tensor(
        [KINDS[k][0] for k in kinds], dtype=torch.float32, device="cuda"
    )
    top_k = torch.tensor([KINDS[k][1] for k in kinds], dtype=torch.int32, device="cuda")
    min_p = torch.tensor(
        [KINDS[k][2] for k in kinds], dtype=torch.float32, device="cuda"
    )
    top_p = torch.tensor(
        [KINDS[k][3] for k in kinds], dtype=torch.float32, device="cuda"
    )
    seeds = torch.randint(-(1 << 60), 1 << 60, (num_reqs,), generator=g, device="cuda")
    idx_mapping = torch.arange(num_reqs, dtype=torch.int64, device="cuda")
    expanded_idx = idx_mapping.repeat_interleave(rows_per)
    row_idx = torch.arange(rows, device="cuda")
    local_pos = (row_idx % rows_per).to(torch.int32)
    req_of_row = row_idx // rows_per
    cu = torch.arange(num_reqs + 1, dtype=torch.int32, device="cuda") * rows_per
    base_pos = torch.randint(1, 5000, (num_reqs,), generator=g, device="cuda")
    pos = base_pos[req_of_row] + local_pos
    cu_np = (np.arange(num_reqs + 1) * rows_per).astype(np.int32)

    def process(logits, expanded):
        apply_temperature(logits, expanded, temperature)
        apply_min_p(logits, expanded, min_p)
        return apply_top_k_top_p(logits, top_k[expanded.long()], top_p[expanded.long()])

    with torch.inference_mode():
        full = MuseSpark1xForCausalLM.compute_logits(model, hidden)
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

        ref_sampled, ref_num = rejection_sample(
            process(full.clone(), expanded_idx),
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

        batch = SimpleNamespace(
            req_ids=[f"r{r}" for r in range(num_reqs)],
            num_reqs=num_reqs,
            num_reqs_after_padding=num_reqs,
            idx_mapping=idx_mapping,
            idx_mapping_np=idx_mapping.cpu().numpy(),
            expanded_idx_mapping=expanded_idx,
            expanded_local_pos=local_pos,
            num_scheduled_tokens=np.full(num_reqs, rows_per, dtype=np.int32),
            num_tokens=rows,
            num_tokens_after_padding=rows,
            num_draft_tokens=num_reqs * STEPS,
            num_draft_tokens_per_req=np.full(num_reqs, STEPS, dtype=np.int32),
            seq_lens=torch.full((num_reqs,), 1000, dtype=torch.int32, device="cuda"),
            seq_lens_cpu_upper_bound=torch.full((num_reqs,), 1000, dtype=torch.int32),
            input_ids=draft_sampled,
            positions=pos,
            logits_indices=row_idx,
            cu_num_logits=cu,
            cu_num_logits_np=cu_np,
        )
        cand_reqs = np.asarray(
            [r for r in range(num_reqs) if KINDS[kinds[r]][4] == "candidates"]
        )
        full_reqs = np.asarray(
            [r for r in range(num_reqs) if KINDS[kinds[r]][4] == "full"]
        )

        def split_step():
            cand = SplitIndices(batch, cand_reqs, torch.device("cuda"))
            other = SplitIndices(batch, full_reqs, torch.device("cuda"))
            topk, logits_i = MuseSpark1xForCausalLM.compute_split_verify_logits(
                model, hidden, 64, cand.rows_t, other.rows_t
            )
            vc, vo = sampler_view(batch, cand), sampler_view(batch, other)
            cand_sampled, cand_num = topk_verify(
                topk,
                vc.input_ids,
                vc.positions,
                vc.cu_num_logits,
                vc.idx_mapping,
                vc.expanded_idx_mapping,
                vc.expanded_local_pos,
                temperature,
                top_k,
                seeds,
                STEPS,
                draft_logits=draft_logits,
                has_sampled=True,
                min_p=min_p,
                top_p=top_p,
            )
            other_sampled, other_num = rejection_sample(
                process(logits_i, vo.expanded_idx_mapping),
                draft_logits,
                vo.input_ids,
                vo.cu_num_logits,
                vo.positions,
                vo.idx_mapping,
                vo.expanded_idx_mapping,
                vo.expanded_local_pos,
                temperature,
                seeds,
                STEPS,
                None,
                use_fp64=False,
                use_block_verification=True,
            )

            def out(sampled, num):
                return SamplerOutput(sampled, None, None, num, torch.zeros_like(num))

            merged = merge_outputs(
                num_reqs,
                [
                    (cand, out(cand_sampled, cand_num)),
                    (other, out(other_sampled, other_num)),
                ],
            )
            return merged.sampled_token_ids, merged.num_sampled

        first_sampled, first_num = split_step()
        second_sampled, second_num = split_step()
        valid = torch.arange(STEPS + 1, device="cuda")[None, :] < first_num[:, None]
        packed = torch.where(valid, first_sampled, torch.full_like(first_sampled, -1))
        everyone = tensor_model_parallel_all_gather(packed.view(-1), 0).view(world, -1)

    ref_valid = torch.arange(STEPS + 1, device="cuda")[None, :] < ref_num[:, None]
    ref_packed = torch.where(ref_valid, ref_sampled, torch.full_like(ref_sampled, -1))
    same = (ref_packed == packed).all(dim=1)
    kind_of = torch.tensor(kinds, device="cuda")
    by_kind = {
        str(k): float(same[kind_of == k].float().mean()) for k in range(len(KINDS))
    }
    return dict(
        greedy_equal=bool(same[kind_of == 0].all()),
        full_route_equal=bool(same[kind_of >= 4].all()),
        by_kind=by_kind,
        overall_share=float(same.float().mean()),
        lengths_equal_share=float((ref_num == first_num).float().mean()),
        deterministic=bool(
            torch.equal(first_sampled[valid], second_sampled[valid])
            and torch.equal(first_num, second_num)
        ),
        same_on_all_ranks=bool((everyone == everyone[0]).all()),
        in_vocab=bool(((packed[valid] >= 0) & (packed[valid] < VOCAB)).all()),
    )


def _verify_worker(rank: int, world: int, port: int, out_path: str) -> None:
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.model_executor.layers.logits_processor import LogitsProcessor
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

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
    # Large logits (std 12 raw, capped to about +-2.5 and a few of 6): a head with ties.
    weight = torch.randn(VOCAB, HIDDEN, generator=g, device="cuda") * (12 / HIDDEN**0.5)
    start = lm_head.shard_indices.padded_org_vocab_start_index
    lm_head.weight.data.copy_(
        weight[start : start + lm_head.num_embeddings_per_partition]
    )
    model = fake_model(
        lm_head=lm_head, logits_processor=logits_processor, topk_logits_k=64
    )

    results = {}
    for num_reqs in CASES:
        results[str(num_reqs)] = _run_case(world, model, g, num_reqs)
        torch.distributed.barrier()
    if rank == 0:
        with open(out_path, "w") as f:
            json.dump(results, f)
    torch.distributed.barrier()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="Requires 8 GPUs")
def test_the_split_step_matches_the_full_path_on_8_ranks(tmp_path):
    """Real lm_head shards and collectives: the requests verified from the gathered
    candidates and those verified from the gathered logits, merged, are the tokens of
    ``compute_logits`` + the full route for the whole batch (greedy requests and the
    requests of the full route bit for bit, the others with the same draws), identical
    on all ranks and run to run."""
    out_path = str(tmp_path / "results.json")
    port = 29700 + os.getpid() % 1000
    mp.spawn(_verify_worker, args=(8, port, out_path), nprocs=8)
    with open(out_path) as f:
        results = json.load(f)
    assert set(results) == {str(n) for n in CASES}
    for name, result in results.items():
        flags = {key: value for key, value in result.items() if isinstance(value, bool)}
        assert all(flags.values()), (name, result)
        assert result["overall_share"] >= 0.97, (name, result)
        assert result["lengths_equal_share"] >= 0.97, (name, result)
