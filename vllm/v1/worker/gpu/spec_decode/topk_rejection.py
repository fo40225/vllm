# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Verifying speculative drafts from the top-K logits of every vocab-parallel rank.

The full verification path gathers the whole ``[rows, vocab]`` logits of the target
over the tensor-parallel group (50 KB per row and rank at a 25k shard), copies them to
a fp32 tensor, masks them (temperature, top-k) and runs the block-verification
statistics of ``rejection_sampler_utils`` over the vocabulary. What it decides is
small: for a greedy row an argmax, for a row with temperature > 0 and top_k <= K the
acceptance of each draft token and one resampled token, and all of them depend on the
``top_k`` largest logits of the row only: the top-k mask sends every other logit to
-inf, so the target distribution of a row lives on its top_k tokens, and every sum over
the vocabulary of the verification (the target log-sum-exp, the residual mass
``sum_x max(p * M_b(x) - M_s(x), 0)``, the residual of the resampled token) is a sum
over those tokens (``M_b(x) = 0`` elsewhere, and the residual is clamped at zero).

``topk_verify`` therefore takes the gathered candidates of ``compute_topk_logits`` (the
K largest logits of each rank, see ``sample/topk_logits.py``) and runs the whole
verification of a request in one program:

* a greedy row (temperature 0) accepts a draft token iff it is the argmax of the row
  (the lowest token id among equal logits, as the full path), and the bonus or
  correcting token is that argmax;
* any other row follows ``_rejection_kernel`` / ``_resample_kernel`` of the block
  method with draft logits step by step, on the candidates: the kept set is the top_k
  of the candidates (the lowest token ids among the ones tied at the boundary, as
  ``apply_top_k_top_p``), the target log-sum-exp is taken over it, a draft token that
  is not in it has target probability 0, the residual mass sums over it, and the
  resampled token is the Gumbel-max draw over it with the noise the full path keys by
  (seed, position, token id). The acceptance uniform is keyed by (seed with the draft
  step mixed in, position), as the block method of ``_rejection_kernel``.
  So the draws are those of the full path for the same logits, up to the order in which
  fp32 terms are summed (the full path adds per 8192-column block, this adds in
  candidate order): a draw differs only where the acceptance uniform or the Gumbel
  scores of the full path sit within fp32 rounding of a threshold.

The draft half of the statistics (the log-sum-exp of the draft distribution at each
step) still reads the draft logits row, which lives on every rank (no gather).

Everything is deterministic and free of host synchronisation.
"""

import torch

from vllm.triton_utils import tl, tldevice, triton
from vllm.v1.worker.gpu.sample.gumbel import (
    gumbel_noised_argmax,
    spec_step_seed,
    tl_rand32,
)
from vllm.v1.worker.gpu.sample.topk_logits import (
    _INT32_MAX,
    MAX_CANDIDATES,
    TopKLogits,
    _load_candidates,
    _select_top,
)
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_global_logsumexp,
    _compute_max_and_sumexp,
)

# Columns of one program of the draft statistics (as the full path).
DRAFT_STATS_BLOCK = 8192
# Warps of the program that verifies a request: the candidate vector of a row is
# TP * K elements (512 at TP 8, K 64).
VERIFY_NUM_WARPS = 4


@triton.jit
def _draft_stats_kernel(
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    # [num_logits, num_blocks]
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits]
    expanded_idx_mapping_ptr,
    # [num_logits]
    expanded_local_pos_ptr,
    # [max_num_reqs]
    temp_ptr,
    vocab_size,
    num_speculative_steps,
    BLOCK_SIZE: tl.constexpr,
):
    """Per-block max and summed exponentials of the draft logits of a row, divided by
    the temperature: the draft half of ``_compute_local_logits_stats_kernel``."""
    logit_idx = tl.program_id(0).to(tl.int64)
    draft_step_idx = tl.load(expanded_local_pos_ptr + logit_idx)
    if draft_step_idx >= num_speculative_steps:
        # Bonus token: no draft distribution.
        return
    req_state_idx = tl.load(expanded_idx_mapping_ptr + logit_idx).to(tl.int64)
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    if temp == 0.0:
        return
    block_idx = tl.program_id(1)
    block_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_offsets < vocab_size
    draft_logits = (
        tl.load(
            draft_logits_ptr
            + req_state_idx * draft_logits_stride_0
            + draft_step_idx * draft_logits_stride_1
            + block_offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        / temp
    )
    draft_max, draft_sumexp = _compute_max_and_sumexp(draft_logits)
    tl.store(
        draft_local_max_ptr + logit_idx * draft_local_max_stride + block_idx,
        draft_max,
    )
    tl.store(
        draft_local_sumexp_ptr + logit_idx * draft_local_sumexp_stride + block_idx,
        draft_sumexp,
    )


@triton.jit
def _target_row(
    words_ptr,
    capped_ptr,
    j,
    row,
    num_rows,
    shard_size,
    temp,
    top_k,
    TP: tl.constexpr,
    K: tl.constexpr,
    ROW_WORDS: tl.constexpr,
):
    """The target distribution of ``row`` on its candidates: the value of each
    candidate divided by the temperature (as ``apply_temperature``: not at all for 1.0),
    its token id, and whether the top-k mask keeps it (exactly top_k candidates, the
    lowest token ids among the ones tied with the top_k-th)."""
    value, token_id, in_range, key = _load_candidates(
        words_ptr, capped_ptr, j, row, num_rows, shard_size, TP, K, ROW_WORDS
    )
    value = tl.where(temp != 1.0, value / temp, value)
    keep, _ = _select_top(key, top_k)
    return value, token_id, keep & in_range


@triton.jit
def _masked_logsumexp(value, keep):
    """log(sum(exp(value))) over the kept candidates (-inf when there is none)."""
    masked = tl.where(keep, value, float("-inf"))
    peak = tl.max(masked, axis=0)
    safe_peak = tl.where(peak > float("-inf"), peak, 0.0)
    total = tl.sum(tl.where(keep, tl.exp(value - safe_peak), 0.0), axis=0)
    return tl.where(peak > float("-inf"), peak + tl.log(total), float("-inf"))


@triton.jit
def _row_argmax(value, token_id, in_range):
    """The lowest token id among the largest values of the candidates."""
    best = tl.max(value, axis=0)
    at_best = in_range & (value == best)
    return tl.min(tl.where(at_best, token_id, _INT32_MAX), axis=0).to(tl.int64)


@triton.jit(do_not_specialize=["num_rows"])
def _topk_verify_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [tp * num_rows, ROW_WORDS] int16 / fp32
    words_ptr,
    capped_ptr,
    # [num_logits]
    draft_sampled_ptr,
    # [num_logits]
    pos_ptr,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    top_k_ptr,
    seed_ptr,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits_ptr,
    draft_logits_stride_0,
    draft_logits_stride_1,
    # [num_logits, num_blocks]
    draft_local_max_ptr,
    draft_local_max_stride,
    draft_local_sumexp_ptr,
    draft_local_sumexp_stride,
    num_rows,
    shard_size,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
    TP: tl.constexpr,
    K: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    NUM_CANDIDATES: tl.constexpr,
    USE_FP64: tl.constexpr,
    HAS_SAMPLED: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start_idx = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    num_draft_tokens = end_idx - start_idx - 1
    temp = tl.load(temp_ptr + req_state_idx).to(tl.float32)
    j = tl.arange(0, NUM_CANDIDATES)
    sampled_row_ptr = sampled_ptr + req_idx * sampled_stride

    accepted_length = tl.zeros((), tl.int64)
    if temp == 0.0:
        # Greedy: accept a draft token iff it is the argmax of its row, and stop at
        # the first rejection, which stores the argmax. The row after the last
        # accepted draft token (the bonus row when all were accepted) gives the token.
        verifying = True
        for i in range(num_draft_tokens):
            row = start_idx + i
            draft_token = tl.load(draft_sampled_ptr + row + 1).to(tl.int64)
            if verifying:
                value, token_id, in_range, _ = _load_candidates(
                    words_ptr,
                    capped_ptr,
                    j,
                    row,
                    num_rows,
                    shard_size,
                    TP,
                    K,
                    ROW_WORDS,
                )
                target_argmax = _row_argmax(value, token_id, in_range)
                # A -1 placeholder is never accepted.
                accepted = (target_argmax == draft_token) & (draft_token >= 0)
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_row_ptr + i,
                    tl.where(accepted, draft_token, target_argmax),
                )
        if accepted_length == num_draft_tokens:
            row = start_idx + num_draft_tokens
            value, token_id, in_range, _ = _load_candidates(
                words_ptr, capped_ptr, j, row, num_rows, shard_size, TP, K, ROW_WORDS
            )
            tl.store(
                sampled_row_ptr + num_draft_tokens,
                _row_argmax(value, token_id, in_range),
            )
    elif HAS_SAMPLED:
        seed = tl.load(seed_ptr + req_state_idx)
        top_k = tl.minimum(tl.load(top_k_ptr + req_state_idx), TP * K)
        draft_row_ptr = draft_logits_ptr + req_state_idx * draft_logits_stride_0

        # Block verification (Sun et al., 2024) with draft logits, as
        # _rejection_kernel: the accepted length is the last step whose uniform is at
        # most its threshold h, and a -1 placeholder ends the verification.
        log_p = tl.zeros((), tl.float32)
        log_p_tau = tl.zeros((), tl.float32)
        verifying = True
        for i in range(num_draft_tokens):
            row = start_idx + i
            draft_token = tl.load(draft_sampled_ptr + row + 1).to(tl.int64)
            verifying &= draft_token >= 0
            if verifying:
                u = tl_rand32(
                    spec_step_seed(seed, i), tl.load(pos_ptr + row), includes_zero=False
                )
                value, token_id, keep = _target_row(
                    words_ptr,
                    capped_ptr,
                    j,
                    row,
                    num_rows,
                    shard_size,
                    temp,
                    top_k,
                    TP,
                    K,
                    ROW_WORDS,
                )
                target_lse = _masked_logsumexp(value, keep)
                # The target logit of the draft token (-inf if the top-k mask drops it).
                target_logit = tl.max(
                    tl.where(keep & (token_id == draft_token), value, float("-inf")),
                    axis=0,
                )
                draft_lse = _compute_global_logsumexp(
                    draft_local_max_ptr,
                    draft_local_max_stride,
                    draft_local_sumexp_ptr,
                    draft_local_sumexp_stride,
                    row,
                    vocab_num_blocks,
                    PADDED_VOCAB_NUM_BLOCKS,
                )
                draft_logit = (
                    tl.load(
                        draft_row_ptr
                        + i.to(tl.int64) * draft_logits_stride_1
                        + draft_token
                    ).to(tl.float32)
                    / temp
                )
                log_p = tl.minimum(
                    log_p + ((target_logit - target_lse) - (draft_logit - draft_lse)),
                    0.0,
                )
                prefix_joint_ratio = tl.exp(log_p)
                next_draft_token = tl.load(
                    draft_sampled_ptr + row + 2,
                    mask=i < num_draft_tokens - 1,
                    other=-1,
                ).to(tl.int64)
                if next_draft_token >= 0:
                    # Residual mass of the next row: sum over its kept candidates of
                    # max(p * M_b(x) - M_s(x), 0).
                    next_value, next_token_id, next_keep = _target_row(
                        words_ptr,
                        capped_ptr,
                        j,
                        row + 1,
                        num_rows,
                        shard_size,
                        temp,
                        top_k,
                        TP,
                        K,
                        ROW_WORDS,
                    )
                    next_target_lse = _masked_logsumexp(next_value, next_keep)
                    next_draft_lse = _compute_global_logsumexp(
                        draft_local_max_ptr,
                        draft_local_max_stride,
                        draft_local_sumexp_ptr,
                        draft_local_sumexp_stride,
                        row + 1,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                    )
                    next_draft_logit = (
                        tl.load(
                            draft_row_ptr
                            + (i + 1).to(tl.int64) * draft_logits_stride_1
                            + next_token_id,
                            mask=next_keep,
                            other=float("-inf"),
                        ).to(tl.float32)
                        / temp
                    )
                    m_b = tl.exp(next_value - next_target_lse)
                    m_s = tl.exp(next_draft_logit - next_draft_lse)
                    residual_mass = tl.sum(
                        tl.where(
                            next_keep,
                            tl.maximum(prefix_joint_ratio * m_b - m_s, 0.0),
                            0.0,
                        ),
                        axis=0,
                    )
                    denom = residual_mass + 1.0 - prefix_joint_ratio
                    h = tl.where(denom > 0.0, residual_mass / denom, 1.0)
                else:
                    h = prefix_joint_ratio
                accepted = u <= h
                accepted_length = tl.where(accepted, i + 1, accepted_length)
                log_p_tau = tl.where(accepted, log_p, log_p_tau)
                tl.store(sampled_row_ptr + i, draft_token)

        # The rejected (or bonus) token: a Gumbel-max draw from the residual.
        row = start_idx + accepted_length
        is_bonus = row == end_idx - 1
        rejected_draft_token = tl.load(
            draft_sampled_ptr + row + 1, mask=not is_bonus, other=0
        )
        value, token_id, keep = _target_row(
            words_ptr,
            capped_ptr,
            j,
            row,
            num_rows,
            shard_size,
            temp,
            top_k,
            TP,
            K,
            ROW_WORDS,
        )
        if is_bonus or rejected_draft_token < 0:
            # No rejection (or a placeholder): the target distribution.
            residual_logits = value
        else:
            # max(p_tau * M_b(x) - M_s(x), 0) / Z in log space: p_tau * M_b = exp(a),
            # M_s = exp(b), log(max(exp(a) - exp(b), 0)) = a + log1p(-exp(b - a)).
            target_lse = _masked_logsumexp(value, keep)
            draft_lse = _compute_global_logsumexp(
                draft_local_max_ptr,
                draft_local_max_stride,
                draft_local_sumexp_ptr,
                draft_local_sumexp_stride,
                row,
                vocab_num_blocks,
                PADDED_VOCAB_NUM_BLOCKS,
            )
            draft_logit = (
                tl.load(
                    draft_row_ptr + accepted_length * draft_logits_stride_1 + token_id,
                    mask=keep,
                    other=float("-inf"),
                ).to(tl.float32)
                / temp
            )
            target_log_probs = value - target_lse
            target_log_probs += log_p_tau
            draft_log_probs = draft_logit - draft_lse
            ratio = tl.exp(draft_log_probs - target_log_probs)
            residual_logits = tl.where(
                ratio < 1.0,
                target_log_probs + tldevice.log1p(-ratio),
                float("-inf"),
            ).to(tl.float32)
        score, winner = gumbel_noised_argmax(
            residual_logits,
            token_id,
            keep,
            seed,
            tl.load(pos_ptr + row),
            temp,
            IS_DRAFTING=False,
            USE_FP64=USE_FP64,
            APPLY_TEMPERATURE=False,
        )
        token = tl.sum(tl.where(j == winner, token_id, 0), axis=0).to(tl.int64)
        # All scores -inf (an empty residual): the full path's argmax lands on token 0.
        token = tl.where(score > float("-inf"), token, 0)
        tl.store(sampled_row_ptr + accepted_length, token)
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))


def topk_verify(
    topk: TopKLogits,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    seed: torch.Tensor,
    num_speculative_steps: int,
    draft_logits: torch.Tensor | None = None,
    has_sampled: bool = True,
    use_fp64: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Verify the draft tokens of a batch against the candidates of ``topk``.

    The result of ``rejection_sample`` (block method, no watermark) for the same
    logits, as ``(sampled, num_sampled)``: ``sampled[r, :num_sampled[r]]`` are the
    accepted draft tokens followed by the resampled (or bonus) token of request ``r``.

    Args:
        topk: the gathered candidates of ``num_logits`` rows. A batch with a row of
            temperature > 0 needs ``topk.k >= top_k`` of that request; a batch without
            one can take ``k = 1``.
        draft_sampled: the token id of each logit row (the draft token verified by the
            row before it); -1 marks a placeholder that is never accepted.
        pos: the position of each logit row (keys the draws).
        cu_num_logits: the first logit row of each request, and the total at the end.
        idx_mapping: the request state index of each request.
        expanded_idx_mapping: the request state index of each logit row.
        expanded_local_pos: the index of each logit row within its request.
        temperature: per request state; 0 is greedy.
        top_k: per request state; at most ``topk.k`` for a request with temperature > 0.
        seed: per request state; keys the draws.
        num_speculative_steps: draft tokens per request at most.
        draft_logits: the pre-temperature draft logits of every request and step;
            required when ``has_sampled`` (the block method with a draft distribution).
        has_sampled: whether any request of the batch has temperature > 0. Host state:
            all-greedy batches skip the draft statistics and the sampled branch.
        use_fp64: the precision of the Gumbel noise (as ``rejection_sample``).

    Returns:
        ``sampled`` ``[num_reqs, num_speculative_steps + 1]`` int64 and ``num_sampled``
        ``[num_reqs]`` int32.

    """
    num_logits, k, tp = topk.num_rows, topk.k, topk.tp_size
    num_reqs = cu_num_logits.shape[0] - 1
    num_candidates = triton.next_power_of_2(tp * k)
    assert tp * k <= MAX_CANDIDATES and not topk.threshold
    row_words = 2 * k
    assert topk.words.is_contiguous() and topk.capped.is_contiguous()
    assert topk.words.shape == (tp * num_logits, row_words) == tuple(topk.capped.shape)
    device = topk.words.device
    sampled = torch.empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64, device=device
    )
    num_sampled = torch.empty(num_reqs, dtype=torch.int32, device=device)
    if num_reqs == 0:
        return sampled, num_sampled

    vocab_num_blocks = 1
    padded_vocab_num_blocks = 1
    draft_local_max = draft_local_sumexp = sampled  # unused placeholders
    draft_stats_strides = (0, 0)
    draft_strides = (0, 0)
    if has_sampled:
        assert draft_logits is not None and draft_logits.ndim == 3
        assert draft_logits.stride(-1) == 1
        vocab_size = draft_logits.shape[-1]
        vocab_num_blocks = triton.cdiv(vocab_size, DRAFT_STATS_BLOCK)
        padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
        draft_local_max = torch.empty(
            num_logits, vocab_num_blocks, dtype=torch.float32, device=device
        )
        draft_local_sumexp = torch.empty_like(draft_local_max)
        draft_stats_strides = (draft_local_max.stride(0), draft_local_sumexp.stride(0))
        draft_strides = (draft_logits.stride(0), draft_logits.stride(1))
        _draft_stats_kernel[(num_logits, vocab_num_blocks)](
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            draft_logits,
            draft_logits.stride(0),
            draft_logits.stride(1),
            expanded_idx_mapping,
            expanded_local_pos,
            temperature,
            vocab_size,
            num_speculative_steps,
            BLOCK_SIZE=DRAFT_STATS_BLOCK,
        )
    _topk_verify_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        topk.words.view(torch.int16),
        topk.capped,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        temperature,
        top_k,
        seed,
        draft_logits,
        *draft_strides,
        draft_local_max,
        draft_stats_strides[0],
        draft_local_sumexp,
        draft_stats_strides[1],
        num_logits,
        topk.shard_size,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        TP=tp,
        K=k,
        ROW_WORDS=row_words,
        NUM_CANDIDATES=num_candidates,
        USE_FP64=use_fp64,
        HAS_SAMPLED=has_sampled,
        num_warps=min(VERIFY_NUM_WARPS, max(1, num_candidates // 128)),
    )
    return sampled, num_sampled
