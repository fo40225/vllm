# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standard and block verification against the proposal caches of the Muse drafters.

DFlash2 caches its proposal as a sparse fp32 [max_reqs, K, V] tensor: the realized
selector scores at the (top-16) candidate ids and -inf everywhere else. DSpark caches
dense head logits in the head dtype (bf16); its dspark_draft_topk path and a reduced
draft vocabulary hand the sampler -inf outside a top-k set, so they cache a sparse
tensor as well. The tests pin that both verification methods read those caches
correctly, against a float64 reference of the two acceptance rules:

* exact bookkeeping: the verifier's uniforms are reproduced with the kernel's own
  generator, so the accepted length of every request must equal the reference's, the
  accepted prefix must be the drafted prefix, and the last token must lie in the
  reference residual (or target) support;
* distribution preservation: on a Markov target over a tiny vocabulary, the output
  block extended by ancestral target samples must follow the target's joint law, with
  drafts produced by the production samplers (the DFlash2 candidate walk and DSpark's
  gumbel_sample cache) and the chain-conditioned proposals they realize;
* forced outcomes (q equal to p accepts everything, a draft the target excludes ends
  the block exactly there), and the RejectionSampler wiring of the method;
* the reference itself is checked against an exact enumeration of the output law.
"""

import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample, tl_rand32
from vllm.v1.worker.gpu.sample.states import NO_LOGPROBS
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import CandidateSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    rejection_sample,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for the verification kernels"
)

METHODS = ["standard", "block"]
# Proposal caches: dense fp32, dense bf16 (DSpark head dtype), sparse fp32 (DFlash2
# candidates), sparse bf16 (DSpark top-k) and none (one-hot, greedy drafting).
KINDS = ["dense_fp32", "dense_bf16", "sparse_fp32", "topk_bf16", "one_hot"]
ANCHOR = 0


# ---------------------------------------------------------------------------
# Float64 reference of the two acceptance rules (CPU)
# ---------------------------------------------------------------------------


def _ref_draft_probs(p, q, drafts, num_drafts):
    """Target and draft probability of every drafted token, and the validity mask.

    p: [B, K+1, V] target probabilities (row i conditions on the first i drafts).
    q: [B, K, V] distribution drafts[:, i] was drawn from. Drafts at or beyond
    num_drafts are ignored.
    """
    num_steps = q.shape[1]
    valid = torch.arange(num_steps)[None, :] < num_drafts[:, None]
    index = drafts.clamp(min=0).unsqueeze(-1)
    p_draft = p[:, :num_steps].gather(-1, index).squeeze(-1)
    q_draft = q.gather(-1, index).squeeze(-1)
    p_draft = torch.where(valid, p_draft, torch.ones_like(p_draft))
    q_draft = torch.where(valid, q_draft, torch.ones_like(q_draft))
    return p_draft, q_draft, valid


def _ref_block_weights(p, q, drafts, num_drafts):
    """Sun et al. (2024), Algorithm 2: running ratios p_i and weights h_i.

    Both are [B, K+1] and indexed by the number of verified drafts i. h[:, 0] is 1 and
    h[:, i] is 0 beyond the drafts the request has.
    """
    batch, num_steps = q.shape[:2]
    p_draft, q_draft, _ = _ref_draft_probs(p, q, drafts, num_drafts)
    running = torch.ones(batch, num_steps + 1, dtype=torch.float64)
    for i in range(1, num_steps + 1):
        ratio = running[:, i - 1] * p_draft[:, i - 1] / q_draft[:, i - 1]
        running[:, i] = torch.clamp(ratio, max=1.0)
    h = torch.zeros(batch, num_steps + 1, dtype=torch.float64)
    h[:, 0] = 1.0
    for i in range(1, num_steps + 1):
        if i == num_steps:
            h_i = running[:, i]
        else:
            excess = torch.clamp(running[:, i, None] * p[:, i] - q[:, i], min=0.0)
            mass = excess.sum(-1)
            denom = mass + 1.0 - running[:, i]
            h_i = torch.where(
                denom > 0, mass / denom.clamp(min=1e-300), torch.ones_like(mass)
            )
            h_i = torch.where(num_drafts == i, running[:, i], h_i)
        h[:, i] = torch.where(i <= num_drafts, h_i, torch.zeros_like(h_i))
    return running, h


def _ref_standard_alpha(p, q, drafts, num_drafts):
    """Leviathan et al. (2023) acceptance probability min(1, p/q), [B, K]."""
    p_draft, q_draft, valid = _ref_draft_probs(p, q, drafts, num_drafts)
    alpha = torch.clamp(p_draft / q_draft, max=1.0)
    return torch.where(valid, alpha, torch.zeros_like(alpha))


def _ref_tau_probs(p, q, drafts, num_drafts, block):
    """Exact law of the number of accepted drafts for a given block, [B, K+1]."""
    batch, num_steps = q.shape[:2]
    out = torch.zeros(batch, num_steps + 1, dtype=torch.float64)
    if block:
        _, h = _ref_block_weights(p, q, drafts, num_drafts)
        for tau in range(num_steps + 1):
            prob = h[:, tau].clone()
            for i in range(tau + 1, num_steps + 1):
                miss = 1.0 - h[:, i]
                prob = prob * torch.where(i <= num_drafts, miss, torch.ones_like(prob))
            out[:, tau] = torch.where(tau <= num_drafts, prob, torch.zeros_like(prob))
        return out
    alpha = _ref_standard_alpha(p, q, drafts, num_drafts)
    lead = torch.ones(batch, dtype=torch.float64)
    for tau in range(num_steps + 1):
        stop = lead if tau == num_steps else lead * (1.0 - alpha[:, tau])
        stop = torch.where(tau == num_drafts, lead, stop)
        out[:, tau] = torch.where(tau <= num_drafts, stop, torch.zeros_like(stop))
        if tau < num_steps:
            lead = lead * alpha[:, tau]
    return out


def _ref_tau(p, q, drafts, num_drafts, uniforms, block):
    """Accepted length for given uniforms, and the closest decision's margin.

    uniforms[:, i] decides draft i (0-based), as in the kernel. The margin is the
    smallest distance between a uniform and its threshold over the decisions the rule
    consults, so near-ties that float32 may resolve differently can be excluded.
    """
    batch, num_steps = q.shape[:2]
    valid = torch.arange(num_steps)[None, :] < num_drafts[:, None]
    if block:
        _, h = _ref_block_weights(p, q, drafts, num_drafts)
        passed = (uniforms <= h[:, 1:]) & valid
        index = torch.arange(1, num_steps + 1)[None, :].expand(batch, -1)
        tau = torch.where(passed, index, torch.zeros_like(index)).amax(-1)
        margin = torch.where(
            valid, (uniforms - h[:, 1:]).abs(), torch.full_like(uniforms, 1e9)
        )
        return tau, margin.amin(-1)
    p_draft, q_draft, _ = _ref_draft_probs(p, q, drafts, num_drafts)
    accepted = (uniforms * q_draft < p_draft) & valid
    tau = torch.cumprod(accepted.long(), dim=-1).sum(-1)
    slack = (p_draft.log() - q_draft.log() - uniforms.log()).abs()
    slack = torch.where(valid & slack.isfinite(), slack, torch.full_like(slack, 1e9))
    return tau, slack.amin(-1)


def _ref_residual(p, q, drafts, num_drafts, tau, block):
    """Unnormalized law of the token emitted after tau accepted drafts, [B, V]."""
    batch, num_steps = q.shape[:2]
    rows = torch.arange(batch)
    p_tau = p[rows, tau]
    if block:
        running, _ = _ref_block_weights(p, q, drafts, num_drafts)
        scale = running[rows, tau]
    else:
        scale = torch.ones(batch, dtype=torch.float64)
    q_tau = q[rows, tau.clamp(max=num_steps - 1)]
    residual = torch.clamp(scale[:, None] * p_tau - q_tau, min=0.0)
    return torch.where((tau == num_drafts)[:, None], p_tau, residual)


# ---------------------------------------------------------------------------
# Batches for the kernels
# ---------------------------------------------------------------------------


@triton.jit
def _verifier_uniform_kernel(uniform_ptr, seed_ptr, pos_ptr, expanded_idx_mapping_ptr):
    # The draw _rejection_kernel makes at each logit row.
    row = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row).to(tl.int64)
    seed = tl.load(seed_ptr + req_state_idx)
    pos = tl.load(pos_ptr + row)
    tl.store(uniform_ptr + row, tl_rand32(seed, pos, includes_zero=False))


def _verifier_uniforms(inputs):
    rows = inputs["pos"].shape[0]
    uniforms = torch.empty(rows, dtype=torch.float32, device=inputs["pos"].device)
    _verifier_uniform_kernel[(rows,)](
        uniforms, inputs["seed"], inputs["pos"], inputs["expanded_idx_mapping"]
    )
    return uniforms


def _topk_mask(logits, keep, gen, rate):
    """Mask all but the top-`keep` tokens of a random `rate` of the rows with -inf."""
    kth = logits.topk(keep, dim=-1).values[..., -1:]
    chosen = torch.rand(logits.shape[:-1] + (1,), generator=gen) < rate
    return torch.where(chosen & (logits < kth), float("-inf"), logits)


def _cache_proposal(kind, raw_draft, gen, candidates):
    """The proposal cache a drafter of `kind` leaves behind, or None for one-hot."""
    if kind == "dense_fp32":
        return raw_draft
    if kind == "dense_bf16":
        return raw_draft.bfloat16()
    if kind == "sparse_fp32":
        # DFlash2: scores at the walk's candidates, which need not be exactly the
        # draft's own top choices.
        noisy = raw_draft + torch.randn(raw_draft.shape, generator=gen)
        ids = noisy.topk(candidates, dim=-1).indices
        proposal = torch.full_like(raw_draft, float("-inf"))
        return proposal.scatter_(-1, ids, raw_draft.gather(-1, ids))
    if kind == "topk_bf16":
        # DSpark dspark_draft_topk: bf16 logits at the top-k, -inf elsewhere.
        ids = raw_draft.topk(candidates, dim=-1).indices
        proposal = torch.full_like(raw_draft, float("-inf")).bfloat16()
        return proposal.scatter_(-1, ids, raw_draft.gather(-1, ids).bfloat16())
    assert kind == "one_hot"
    return None


def _build_batch(
    *,
    kind,
    num_reqs,
    vocab,
    num_steps,
    temperatures,
    seed,
    candidates=16,
    placeholders=True,
    sharpness=2.0,
    draft_noise=0.3,
    mirror_target=False,
    device="cuda",
):
    """Random verification batch: kernel inputs plus the CPU views the reference needs.

    Requests have 0..K drafts, mixed temperatures (0 is greedy), a processed target
    with a random top-k mask, a proposal cache in the layout of `kind` whose
    request-state rows are permuted and surrounded by unrelated rows, and optionally -1
    placeholder drafts. With mirror_target the target rows equal the proposal rows
    (temperature 1, no mask), so p == q wherever a draft is verified.
    """
    gen = torch.Generator().manual_seed(seed)
    batch, steps = num_reqs, num_steps
    temps = torch.tensor(temperatures, dtype=torch.float32)[
        torch.randint(len(temperatures), (batch,), generator=gen)
    ]
    safe_temps = torch.where(temps > 0, temps, torch.ones_like(temps))

    raw_target = sharpness * torch.randn(batch, steps + 1, vocab, generator=gen)
    target = raw_target / safe_temps[:, None, None]
    if not mirror_target:
        target = _topk_mask(target, max(4, min(64, vocab // 3)), gen, 0.3)
    noise = torch.randn(batch, steps, vocab, generator=gen)
    raw_draft = 0.9 * raw_target[:, :steps] + draft_noise * sharpness * noise
    proposal = _cache_proposal(kind, raw_draft, gen, candidates)
    if mirror_target:
        assert proposal is not None and temperatures == [1.0]
        target[:, :steps] = proposal.float()

    sampling_logits = raw_draft if proposal is None else proposal.float()
    sampling_probs = torch.softmax(sampling_logits / safe_temps[:, None, None], dim=-1)
    drafts = torch.multinomial(
        sampling_probs.reshape(batch * steps, vocab), 1, generator=gen
    ).view(batch, steps)

    num_rows_drafts = torch.where(
        torch.rand(batch, generator=gen) < 0.5,
        torch.full((batch,), steps, dtype=torch.long),
        torch.randint(0, steps + 1, (batch,), generator=gen),
    )
    columns = torch.arange(steps)[None, :]
    if placeholders:
        holes = (torch.rand(batch, steps, generator=gen) < 0.15) & (
            torch.rand(batch, 1, generator=gen) < 0.25
        )
        drafts = torch.where(holes & (temps[:, None] > 0), -1, drafts)
    in_rows = columns < num_rows_drafts[:, None]
    # A placeholder ends the block: later drafts are never verified.
    first_hole = torch.where((drafts < 0) & in_rows, columns, steps).amin(-1)
    num_drafts = torch.minimum(num_rows_drafts, first_hole)

    max_reqs = batch + 7
    idx_mapping = torch.randperm(max_reqs, generator=gen)[:batch]
    cache = None
    if proposal is not None:
        cache = torch.randn(max_reqs, steps, vocab, generator=gen).to(proposal.dtype)
        cache[idx_mapping] = proposal
    temp_state = torch.full((max_reqs,), 0.5)
    temp_state[idx_mapping] = temps
    # Request seeds span the whole int64 range, negative values included.
    seed_state = torch.randint(-(2**62), 2**62, (max_reqs,), generator=gen)

    rows = torch.arange(steps + 1)[None, :] <= num_rows_drafts[:, None]
    anchor = torch.randint(0, vocab, (batch, 1), generator=gen)
    draft_sampled = torch.cat([anchor, drafts], dim=1)
    pos = torch.randint(100, 5000, (batch, 1), generator=gen) + torch.arange(steps + 1)
    local_pos = torch.arange(steps + 1, dtype=torch.int32)[None, :].expand(batch, -1)
    expanded_state = idx_mapping.to(torch.int32)[:, None].expand(-1, steps + 1)
    cu = torch.cat([torch.zeros(1, dtype=torch.long), (num_rows_drafts + 1).cumsum(0)])

    ns = SimpleNamespace(
        kernel_inputs=dict(
            target_logits=target[rows].contiguous().to(device),
            draft_logits=None if cache is None else cache.to(device),
            draft_sampled=draft_sampled[rows].to(device),
            cu_num_logits=cu.to(torch.int32).to(device),
            pos=pos[rows].to(device),
            idx_mapping=idx_mapping.to(torch.int32).to(device),
            expanded_idx_mapping=expanded_state[rows].to(device),
            expanded_local_pos=local_pos[rows].to(device),
            temperature=temp_state.to(device),
            seed=seed_state.to(device),
        ),
        num_steps=steps,
        vocab=vocab,
        kind=kind,
        cu_num_logits=cu,
        temperature=temps,
        num_rows_drafts=num_rows_drafts,
        num_drafts=num_drafts,
        drafts=torch.where(columns < num_drafts[:, None], drafts, -1),
        target_logits=target,
        proposal=proposal,
    )
    return ns


def _reference_probabilities(batch):
    """Float64 target and draft probabilities of the batch's requests."""
    temps = batch.temperature.double()
    safe_temp = torch.where(temps > 0, temps, torch.ones_like(temps))[:, None, None]
    p = torch.softmax(batch.target_logits.double(), dim=-1)
    if batch.proposal is None:
        q = torch.nn.functional.one_hot(batch.drafts.clamp(min=0), batch.vocab).double()
    else:
        q = torch.softmax(batch.proposal.double() / safe_temp, dim=-1)
    return p, q


def _unflatten(values, batch, fill=0.0):
    """[num_logits] kernel-order values -> [B, K+1]."""
    rows = (
        torch.arange(batch.num_steps + 1)[None, :] <= batch.num_rows_drafts[:, None]
    )
    out = torch.full(rows.shape, fill, dtype=values.dtype)
    out[rows] = values
    return out


def _check_against_reference(batch, method, label):
    """Compare rejection_sample with the reference, request by request."""
    block = method == "block"
    inputs = batch.kernel_inputs
    steps = batch.num_steps
    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=steps, use_block_verification=block
    )
    sampled, num_sampled = sampled.cpu(), num_sampled.cpu().long()
    uniforms = _unflatten(_verifier_uniforms(inputs).cpu().double(), batch, 0.5)
    uniforms = uniforms[:, :steps]
    p, q = _reference_probabilities(batch)
    drafts, num_drafts = batch.drafts, batch.num_drafts
    greedy = batch.temperature == 0

    tau, margin = _ref_tau(p, q, drafts, num_drafts, uniforms, block)
    comparable = ~greedy & (margin > 1e-4)
    assert comparable.float().mean() > 0.3, f"[{label}] too few comparable requests"
    wrong = comparable & (num_sampled - 1 != tau)
    assert not wrong.any(), (
        f"[{label}] accepted length differs from the reference in {int(wrong.sum())} "
        f"of {int(comparable.sum())} requests; first rows "
        f"{wrong.nonzero().flatten()[:5].tolist()}: kernel "
        f"{(num_sampled - 1)[wrong][:5].tolist()}, reference {tau[wrong][:5].tolist()}"
    )

    # The accepted prefix is the drafted prefix, and the last token is a legal draw
    # from the reference residual (or target, for a bonus token).
    rows = comparable.nonzero().flatten()
    kept = torch.arange(steps)[None, :] < tau[rows, None]
    assert torch.equal(sampled[rows, :steps][kept], drafts[rows][kept]), label
    last = sampled[rows, tau[rows]]
    assert ((last >= 0) & (last < batch.vocab)).all(), label
    residual = _ref_residual(p, q, drafts, num_drafts, tau, block)
    outside = residual[rows, last] <= 0
    assert not outside.any(), (
        f"[{label}] {int(outside.sum())} requests emitted a token outside the "
        "residual support"
    )

    # Greedy rows: accept while the draft is the target argmax, then emit the argmax.
    rows = greedy.nonzero().flatten()
    if len(rows):
        argmax = batch.target_logits[rows].argmax(-1)
        match = drafts[rows] == argmax[:, :steps]
        expected = torch.cumprod(match.long(), dim=-1).sum(-1)
        assert torch.equal(num_sampled[rows] - 1, expected), f"[{label}] greedy"
        last = sampled[rows, expected]
        assert torch.equal(last, argmax[torch.arange(len(rows)), expected]), label


# ---------------------------------------------------------------------------
# The reference is a valid speculative sampler (exact enumeration, CPU)
# ---------------------------------------------------------------------------


def _random_markov_tables(vocab, steps, gen, sharpness=1.5):
    def table(rows):
        logits = sharpness * torch.randn(
            rows, vocab, vocab, generator=gen, dtype=torch.float64
        )
        return torch.softmax(logits, -1)

    return table(steps + 1), table(steps)


def _place_values(vocab, steps):
    return [vocab ** (steps - j) for j in range(steps + 1)]


def _markov_joint_law(target):
    """Law of (y_0..y_K) when sampling the target chain; y_0 most significant."""
    steps, vocab = target.shape[0] - 1, target.shape[1]
    place = _place_values(vocab, steps)
    law = torch.zeros(vocab ** (steps + 1), dtype=torch.float64)
    for y in itertools.product(range(vocab), repeat=steps + 1):
        prob = 1.0
        for j in range(steps + 1):
            prob *= float(target[j, ANCHOR if j == 0 else y[j - 1], y[j]])
        law[sum(y[j] * place[j] for j in range(steps + 1))] = prob
    return law


def _exact_extended_law(method, target, draft):
    """Exact law of (output block + ancestral target continuation), by enumeration.

    Row i of the target and draft chains conditions on the token that precedes position
    i (the anchor for i = 0).
    """
    steps, vocab = draft.shape[0], draft.shape[1]
    block = method == "block"
    place = _place_values(vocab, steps)
    law = torch.zeros(vocab ** (steps + 1), dtype=torch.float64)
    for x in itertools.product(range(vocab), repeat=steps):
        inputs = [ANCHOR, *x]
        weight = math.prod(float(draft[i, inputs[i], x[i]]) for i in range(steps))
        p = torch.stack([target[i, inputs[i]] for i in range(steps + 1)])[None]
        q = torch.stack([draft[i, inputs[i]] for i in range(steps)])[None]
        drafts, num_drafts = torch.tensor([x]), torch.tensor([steps])
        tau_probs = _ref_tau_probs(p, q, drafts, num_drafts, block)[0]
        for tau in range(steps + 1):
            if tau_probs[tau] == 0:
                continue
            at_tau = torch.tensor([tau])
            residual = _ref_residual(p, q, drafts, num_drafts, at_tau, block)[0]
            residual = residual / residual.sum()
            for y_tau in range(vocab):
                prob = weight * float(tau_probs[tau]) * float(residual[y_tau])
                if prob == 0:
                    continue
                for tail in itertools.product(range(vocab), repeat=steps - tau):
                    y = [*x[:tau], y_tau, *tail]
                    extension = math.prod(
                        float(target[j, y[j - 1], y[j]])
                        for j in range(tau + 1, steps + 1)
                    )
                    law[sum(y[j] * place[j] for j in range(steps + 1))] += (
                        prob * extension
                    )
    return law


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("vocab,steps", [(3, 2), (3, 3), (4, 2)])
def test_reference_rules_preserve_the_target_law_exactly(method, vocab, steps):
    """The float64 reference is a valid speculative sampler (enumerated, no sampling).

    For every draft block, the reference's accepted-length law and residual combine into
    the exact law of the emitted block extended by ancestral target samples, which
    must be the product of the target conditionals.
    """
    gen = torch.Generator().manual_seed(1234 + vocab * 10 + steps)
    target, draft = _random_markov_tables(vocab, steps, gen)
    law = _exact_extended_law(method, target, draft)
    torch.testing.assert_close(law, _markov_joint_law(target), rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_reference_rules_preserve_the_law_with_a_masked_target(method):
    """A target that excludes tokens (top-k, top-p) keeps the law, never emits them."""
    gen = torch.Generator().manual_seed(77)
    vocab, steps = 4, 2
    target, draft = _random_markov_tables(vocab, steps, gen)
    keep = target.topk(2, dim=-1).values[..., -1:]
    target = torch.where(target >= keep, target, torch.zeros_like(target))
    target = target / target.sum(-1, keepdim=True)
    law = _exact_extended_law(method, target, draft)
    torch.testing.assert_close(law, _markov_joint_law(target), rtol=1e-10, atol=1e-12)


def test_reference_block_accepts_more_than_standard():
    """Block verification accepts more than token verification in expectation."""
    gen = torch.Generator().manual_seed(5)
    vocab, steps = 3, 3
    target, draft = _random_markov_tables(vocab, steps, gen, sharpness=0.5)
    total = {"standard": 0.0, "block": 0.0}
    for x in itertools.product(range(vocab), repeat=steps):
        inputs = [ANCHOR, *x]
        weight = math.prod(float(draft[i, inputs[i], x[i]]) for i in range(steps))
        p = torch.stack([target[i, inputs[i]] for i in range(steps + 1)])[None]
        q = torch.stack([draft[i, inputs[i]] for i in range(steps)])[None]
        drafts, num_drafts = torch.tensor([x]), torch.tensor([steps])
        for method in total:
            probs = _ref_tau_probs(p, q, drafts, num_drafts, method == "block")[0]
            mean = float((probs * torch.arange(steps + 1)).sum())
            total[method] += weight * mean
    assert total["block"] > total["standard"] + 1e-4


# ---------------------------------------------------------------------------
# Exact bookkeeping of the kernels
# ---------------------------------------------------------------------------


@requires_cuda
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", KINDS)
def test_accepted_length_matches_reference_small_vocab(method, kind):
    """Mixed temperatures, draft counts and placeholders; persistent-row indexing."""
    batch = _build_batch(
        kind=kind,
        num_reqs=40000,
        vocab=24,
        num_steps=7,
        temperatures=[1.0, 0.6, 0.0],
        seed=11,
        candidates=8,
    )
    _check_against_reference(batch, method, f"{method}/{kind}/V24")


@requires_cuda
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["dense_bf16", "sparse_fp32", "topk_bf16"])
def test_accepted_length_matches_reference_multi_block_vocab(method, kind):
    """V = 20000 spans three 8192-wide vocabulary blocks.

    Sixteen sparse candidates leave most blocks of the proposal entirely -inf, so the
    per-block max and sum-exp partials, and the residual-mass partials, must be
    neutral there.
    """
    batch = _build_batch(
        kind=kind,
        num_reqs=160,
        vocab=20000,
        num_steps=7,
        temperatures=[1.0, 0.6],
        seed=21,
        candidates=16,
        placeholders=False,
        sharpness=4.0,
    )
    _check_against_reference(batch, method, f"{method}/{kind}/V20000")


@requires_cuda
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", ["dense_bf16", "sparse_fp32"])
def test_accepted_length_matches_reference_production_vocab(method, kind):
    """Muse's vocabulary width: 202048 is 25 vocabulary blocks, 16 sparse candidates."""
    batch = _build_batch(
        kind=kind,
        num_reqs=48,
        vocab=202048,
        num_steps=7,
        temperatures=[1.0, 0.6],
        seed=41,
        candidates=16,
        placeholders=False,
        sharpness=4.0,
        draft_noise=0.1,
    )
    _check_against_reference(batch, method, f"{method}/{kind}/V202048")


# ---------------------------------------------------------------------------
# Distribution preservation with the production drafters' caches
# ---------------------------------------------------------------------------

DIST_VOCAB = 5
DIST_STEPS = 3
DIST_CANDIDATES = 3
DIST_TRIALS = 1_500_000
DIST_CHUNK = 500_000
# name -> (temperature, tokens the target keeps; None keeps all)
DIST_SETTINGS = {"t1.0-full": (1.0, None), "t0.6-top3": (0.6, 3)}
DIST_KINDS = ["dense_bf16", "topk_bf16", "sparse_fp32", "one_hot"]


def _dist_tables(temperature, keep, seed):
    """Markov target/draft logits: [row][previous token][next token]."""
    gen = torch.Generator().manual_seed(seed)
    steps, vocab = DIST_STEPS, DIST_VOCAB
    raw_target = 1.5 * torch.randn(steps + 1, vocab, vocab, generator=gen)
    target = raw_target / temperature
    if keep is not None:
        kth = target.topk(keep, dim=-1).values[..., -1:]
        target = torch.where(target >= kth, target, float("-inf"))
    raw_draft = 0.5 * raw_target[:steps] + 1.2 * torch.randn(
        steps, vocab, vocab, generator=gen
    )
    return target, torch.softmax(target.double(), dim=-1), raw_draft


def _draft_chain(kind, raw_draft, temperature, seeds, start_pos):
    """Drafts and proposal cache written by the production sampler of `kind`.

    DSpark kinds feed per-step logits (conditioned on the previous draft, the Markov
    head) to gumbel_sample with the draft-logits cache, as DSpark's _sample_logits
    does; DFlash2 runs the candidate walk, whose realized rows are conditioned on the
    previously chosen candidate. Returns drafts [N, K] and the cache [N, K, V], or
    None for one-hot (greedy) drafting.
    """
    device = seeds.device
    n, steps, vocab = seeds.shape[0], DIST_STEPS, DIST_VOCAB
    pos = start_pos[:, None] + torch.arange(steps, device=device)
    temps = torch.full((n,), temperature, dtype=torch.float32, device=device)
    # The speculators hold their sampling-row to request-state map in int64.
    state = torch.arange(n, dtype=torch.int64, device=device)
    drafts = torch.empty(n, steps, dtype=torch.long, device=device)
    table = raw_draft.to(device)
    prev = torch.full((n,), ANCHOR, dtype=torch.long, device=device)

    if kind == "sparse_fp32":
        top_k = DIST_CANDIDATES
        candidates = torch.rand(n, steps, vocab, device=device).topk(top_k, -1).indices
        unary = table.mean(1)[None].expand(n, -1, -1).gather(-1, candidates)
        scores = unary[:, :, None, :] + 0.7 * torch.randn(
            n, steps, top_k, top_k, device=device
        )
        cache = torch.full((n, steps, vocab), float("-inf"), device=device)
        CandidateSampler(n, steps, top_k, device).sample(
            candidates,
            scores.contiguous(),
            n,
            (pos + 1).reshape(-1),
            state.repeat_interleave(steps),
            temps,
            seeds,
            drafts,
            cache,
            False,
        )
        return drafts, cache

    if kind == "one_hot":
        for i in range(steps):
            drafts[:, i] = table[i][prev].argmax(-1)
            prev = drafts[:, i]
        return drafts, None

    dtype = torch.float32 if kind == "dense_fp32" else torch.bfloat16
    cache = torch.zeros(n, steps, vocab, dtype=dtype, device=device)
    columns = torch.arange(steps, dtype=torch.int32, device=device)
    for i in range(steps):
        logits = table[i][prev].to(dtype)
        if kind == "topk_bf16":
            keep = logits.float().topk(DIST_CANDIDATES, dim=-1).indices
            sparse = torch.full_like(logits, float("-inf"))
            logits = sparse.scatter_(-1, keep, logits.gather(-1, keep))
        drafts[:, i] = gumbel_sample(
            logits,
            state,
            temps,
            seeds,
            pos[:, i].contiguous(),
            apply_temperature=True,
            is_drafting=True,
            logits_cache=cache,
            logits_cache_col=columns[i],
        )
        prev = drafts[:, i]
    return drafts, cache


def _joint_counts(method, kind, setting, seed, cache_scale=1.0):
    """Histogram of (output block + ancestral target continuation) over many trials."""
    temperature, keep = DIST_SETTINGS[setting]
    steps, vocab = DIST_STEPS, DIST_VOCAB
    device = torch.device("cuda")
    target, probs, raw_draft = _dist_tables(temperature, keep, seed)
    target, probs = target.to(device), probs.float().to(device)
    place = torch.tensor(
        [vocab ** (steps - j) for j in range(steps + 1)], device=device
    )
    counts = torch.zeros(vocab ** (steps + 1), dtype=torch.float64, device=device)
    for chunk in range(DIST_TRIALS // DIST_CHUNK):
        torch.manual_seed(seed * 100 + chunk)
        n = DIST_CHUNK
        seeds = torch.randint(-(2**62), 2**62, (n,), dtype=torch.int64, device=device)
        start_pos = torch.randint(100, 5000, (n,), dtype=torch.int64, device=device)
        drafts, cache = _draft_chain(kind, raw_draft, temperature, seeds, start_pos)
        if cache is not None and cache_scale != 1.0:
            cache = cache * cache_scale
        tokens_in = torch.cat([torch.full_like(drafts[:, :1], ANCHOR), drafts], dim=1)
        rows = target[torch.arange(steps + 1, device=device)[None, :], tokens_in]
        pos = start_pos[:, None] + torch.arange(steps + 1, device=device)
        request = torch.arange(n, dtype=torch.int32, device=device)
        local = torch.arange(steps + 1, dtype=torch.int32, device=device)
        cu_num_logits = torch.arange(n + 1, dtype=torch.int32, device=device)
        sampled, num_sampled = rejection_sample(
            target_logits=rows.reshape(-1, vocab).contiguous(),
            draft_logits=cache,
            draft_sampled=tokens_in.reshape(-1).contiguous(),
            cu_num_logits=cu_num_logits * (steps + 1),
            pos=pos.reshape(-1).contiguous(),
            idx_mapping=request,
            expanded_idx_mapping=request.repeat_interleave(steps + 1),
            expanded_local_pos=local.repeat(n),
            temperature=torch.full((n,), temperature, device=device),
            seed=seeds,
            num_speculative_steps=steps,
            use_block_verification=method == "block",
        )
        emitted = torch.arange(steps + 1, device=device)[None, :] < num_sampled[:, None]
        assert ((sampled[emitted] >= 0) & (sampled[emitted] < vocab)).all()
        out = sampled.clone()
        for j in range(1, steps + 1):
            extension = torch.multinomial(probs[j][out[:, j - 1]], 1).squeeze(-1)
            out[:, j] = torch.where(num_sampled <= j, extension, out[:, j])
        counts += torch.bincount((out * place).sum(-1), minlength=counts.numel())
    return counts.cpu(), probs.double().cpu(), steps


def _joint_chi2(counts, probs, steps):
    """Chi-squared statistic, degrees of freedom and a ~6 sigma threshold."""
    law = _markov_joint_law(probs)
    total = counts.sum().item()
    assert total == DIST_TRIALS
    # A token the target excludes must never be emitted.
    assert counts[law == 0].sum().item() == 0, "emitted a token with target mass 0"
    expected = law * total
    big = expected >= 5.0
    observed_all = torch.cat([counts[big], counts[~big].sum()[None]])
    expected_all = torch.cat([expected[big], expected[~big].sum()[None]])
    if expected_all[-1] < 5.0:
        observed_all, expected_all = observed_all[:-1], expected_all[:-1]
    chi2 = ((observed_all - expected_all) ** 2 / expected_all).sum().item()
    df = observed_all.numel() - 1
    return chi2, df, df + 6.0 * math.sqrt(2.0 * df)


@requires_cuda
@pytest.mark.parametrize("setting", list(DIST_SETTINGS))
@pytest.mark.parametrize("kind", DIST_KINDS)
@pytest.mark.parametrize("method", METHODS)
def test_output_law_follows_the_target_with_production_proposals(method, kind, setting):
    """The first tokens of a block, extended by the target, are target samples.

    Drafts come from the production samplers (DSpark's gumbel_sample cache, DFlash2's
    candidate walk) on a chain whose target and proposal both depend on the previous
    token, so every position's residual and the block rule's h_i are exercised. The
    joint law of the whole extended block is compared with the target's, with the
    proposal cache in the dense bf16, top-k bf16, sparse fp32 and one-hot layouts, with
    and without a temperature and a top-k truncated target.
    """
    counts, probs, steps = _joint_counts(method, kind, setting, seed=3)
    chi2, df, threshold = _joint_chi2(counts, probs, steps)
    assert chi2 < threshold, (
        f"[{method}/{kind}/{setting}] joint law differs from the target: "
        f"chi2={chi2:.1f}, df={df}, threshold={threshold:.1f}"
    )


@requires_cuda
@pytest.mark.parametrize("method", METHODS)
def test_output_law_check_detects_a_cache_that_is_not_the_proposal(method):
    """Negative control: the same check fails when the cache is not what was sampled.

    Scaling the cached DFlash2 scores after the walk makes the verifier see a sharper
    proposal than the one the drafts came from, which must show in the joint law.
    """
    counts, probs, steps = _joint_counts(
        method, "sparse_fp32", "t0.6-top3", seed=3, cache_scale=1.5
    )
    chi2, df, threshold = _joint_chi2(counts, probs, steps)
    assert chi2 > threshold, (
        f"[{method}] the joint-law check did not notice a wrong cache: "
        f"chi2={chi2:.1f}, df={df}, threshold={threshold:.1f}"
    )


# ---------------------------------------------------------------------------
# The caches themselves
# ---------------------------------------------------------------------------


def _chi2_against(counts, probs):
    """Chi-squared statistic, degrees of freedom and a ~6 sigma threshold."""
    expected = probs * counts.sum()
    assert (counts[probs == 0] == 0).all()
    keep = expected >= 5.0
    chi2 = ((counts[keep] - expected[keep]) ** 2 / expected[keep]).sum().item()
    df = int(keep.sum()) - 1
    return chi2, df, df + 6.0 * math.sqrt(2.0 * df)


@requires_cuda
@pytest.mark.parametrize("temperature", [1.0, 0.6])
@pytest.mark.parametrize("kind", ["sparse_fp32", "dense_bf16", "topk_bf16"])
def test_drafts_follow_the_cached_proposal(kind, temperature):
    """What the verifier reads back from the cache is what the drafts were drawn from.

    The cache holds pre-temperature scores and the verifier divides by the request
    temperature on load; the DFlash2 walk and DSpark's Gumbel sampler divide before
    adding noise. A fixed row is drawn many times and compared with the softmax of the
    cached row, at a temperature other than 1 as well.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    n, vocab, top_k = 1_000_000, 9, 4
    seeds = torch.randint(-(2**62), 2**62, (n,), dtype=torch.int64, device=device)
    temps = torch.full((n,), temperature, device=device)
    state = torch.arange(n, dtype=torch.int64, device=device)
    pos = torch.randint(100, 5000, (n,), dtype=torch.int64, device=device)
    row = 1.5 * torch.randn(vocab, device=device)
    if kind == "sparse_fp32":
        ids = torch.tensor([1, 3, 4, 7], device=device)
        candidates = ids.expand(n, 1, top_k).contiguous()
        scores = row[ids].expand(n, 1, top_k, top_k).contiguous()
        cache = torch.full((n, 1, vocab), float("-inf"), device=device)
        drafts = torch.empty(n, 1, dtype=torch.long, device=device)
        CandidateSampler(n, 1, top_k, device).sample(
            candidates, scores, n, pos + 1, state, temps, seeds, drafts, cache, False
        )
    else:
        logits = row.bfloat16()
        if kind == "topk_bf16":
            keep = logits >= logits.topk(top_k).values[-1]
            logits = torch.where(keep, logits, -math.inf)
        cache = torch.zeros(n, 1, vocab, dtype=torch.bfloat16, device=device)
        drafts = gumbel_sample(
            logits.expand(n, vocab).contiguous(),
            state,
            temps,
            seeds,
            pos,
            apply_temperature=True,
            is_drafting=True,
            logits_cache=cache,
            logits_cache_col=torch.zeros((), dtype=torch.int32, device=device),
        ).view(n, 1)
    cached = cache[0, 0].double()
    assert torch.equal(cache[:, 0], cache[0, 0].expand(n, vocab))
    probs = torch.softmax(cached / temperature, dim=-1).cpu()
    counts = torch.bincount(drafts.flatten(), minlength=vocab).double().cpu()
    chi2, df, threshold = _chi2_against(counts, probs)
    assert chi2 < threshold, f"chi2={chi2:.1f}, df={df}, threshold={threshold:.1f}"
    if kind != "dense_bf16":
        assert (probs == 0).any(), "the cache is not sparse"


@requires_cuda
def test_candidate_cache_holds_exactly_the_realized_candidates():
    """Reusing a request-state row clears the previous candidates, overlaps included.

    The cache kernel writes -inf at the previous step's candidates and then the new
    scores at this step's, so an id that stays a candidate at another slot must end up
    with its new score, an id that drops out must be cleared, and the result must equal
    the cache of a sampler that never ran before.
    """
    device = torch.device("cuda")
    gen = torch.Generator().manual_seed(0)
    batch, steps, top_k, vocab = 6, 3, 5, 12
    state = torch.tensor([3, 0, 5, 1, 4, 2], dtype=torch.int64, device=device)
    temps = torch.tensor([0.0, 1.0, 0.7, 1.0, 0.0, 0.6], device=device)
    seeds = torch.arange(11, 11 + batch, dtype=torch.int64, device=device)
    sample_pos = torch.arange(batch * steps, dtype=torch.int64, device=device) + 40
    mapping = state.repeat_interleave(steps)

    def walk(sampler, cache, candidates, scores):
        tokens = torch.empty(batch, steps, dtype=torch.long, device=device)
        sampler.sample(
            candidates,
            scores,
            batch,
            sample_pos,
            mapping,
            temps,
            seeds,
            tokens,
            cache,
            False,
        )
        return tokens

    ids = torch.rand(batch, steps, vocab, generator=gen).topk(top_k).indices
    # The same ids at other slots, with one id that was not a candidate before.
    outside = torch.rand(batch, steps, vocab, generator=gen).scatter(-1, ids, -1.0)
    moved = ids.roll(2, dims=-1)
    moved[..., 0] = outside.argmax(-1)
    ids, moved = ids.to(device), moved.to(device)
    scores = [
        torch.randn(batch, steps, top_k, top_k, generator=gen).to(device)
        for _ in range(2)
    ]

    cache = torch.full((batch, steps, vocab), float("-inf"), device=device)
    sampler = CandidateSampler(batch, steps, top_k, device)
    walk(sampler, cache, ids, scores[0])
    tokens = walk(sampler, cache, moved, scores[1])

    fresh = torch.full((batch, steps, vocab), float("-inf"), device=device)
    fresh_sampler = CandidateSampler(batch, steps, top_k, device)
    fresh_tokens = walk(fresh_sampler, fresh, moved, scores[1])
    assert torch.equal(tokens, fresh_tokens)
    assert torch.equal(cache, fresh)

    # The cache is the realized rows, conditioned on the previously chosen candidate.
    expected = torch.full_like(cache, float("-inf")).scatter_(
        -1, moved, sampler.scores[:batch]
    )
    assert torch.equal(cache[state], expected)
    assert (cache.isfinite().sum(-1) == top_k).all()
    previous = torch.zeros(batch, dtype=torch.long, device=device)
    for step in range(steps):
        realized = scores[1][torch.arange(batch), step, previous]
        assert torch.equal(sampler.scores[:batch, step], realized)
        previous = (moved[:, step] == tokens[:, step, None]).long().argmax(-1)


# ---------------------------------------------------------------------------
# Forced outcomes
# ---------------------------------------------------------------------------

MIRROR_KINDS = ["dense_bf16", "topk_bf16", "sparse_fp32"]


def _mirror_batch(kind, num_reqs=3000):
    return _build_batch(
        kind=kind,
        num_reqs=num_reqs,
        vocab=24,
        num_steps=7,
        temperatures=[1.0],
        seed=31,
        candidates=8,
        placeholders=False,
        mirror_target=True,
    )


@requires_cuda
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", MIRROR_KINDS)
def test_target_equal_to_proposal_accepts_every_draft(method, kind):
    """With p == q at every verified row, no draft may be rejected.

    Standard verification accepts with probability min(1, p/q) = 1, and in block
    verification the running ratio stays 1 and the residual mass is 0, so the 0/0
    weight must resolve to 1. -inf is shared by p and q outside a sparse support.
    """
    batch = _mirror_batch(kind)
    inputs = batch.kernel_inputs
    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=7, use_block_verification=method == "block"
    )
    sampled, num_sampled = sampled.cpu(), num_sampled.cpu().long()
    assert torch.equal(num_sampled, batch.num_rows_drafts + 1)
    keep = torch.arange(7)[None, :] < batch.num_drafts[:, None]
    assert torch.equal(sampled[:, :7][keep], batch.drafts[keep])
    # The extra token is drawn from the target row, which has the proposal's support.
    partial = (batch.num_drafts < 7).nonzero().flatten()
    extra = sampled[partial, batch.num_drafts[partial]]
    assert batch.proposal[partial, batch.num_drafts[partial], extra].isfinite().all()


@requires_cuda
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind", MIRROR_KINDS)
def test_draft_excluded_by_the_target_ends_the_block_there(method, kind):
    """A draft with target probability 0 (top-k/top-p) is rejected, and the block
    stops exactly at it: earlier drafts (p == q) are all accepted, later ones, which
    could be accepted on their own, are not, and the replacement comes from the
    target's support without the excluded token.
    """
    batch = _mirror_batch(kind)
    inputs = batch.kernel_inputs
    steps = 7
    full = (batch.num_drafts == steps).nonzero().flatten()
    gen = torch.Generator().manual_seed(1)
    stop = torch.randint(0, steps, (len(full),), generator=gen)
    rows = (batch.cu_num_logits[full] + stop).to(inputs["target_logits"].device)
    excluded = batch.drafts[full, stop].to(rows.device)
    inputs["target_logits"][rows, excluded] = float("-inf")

    sampled, num_sampled = rejection_sample(
        **inputs, num_speculative_steps=steps, use_block_verification=method == "block"
    )
    sampled, num_sampled = sampled.cpu(), num_sampled.cpu().long()
    assert torch.equal(num_sampled[full], stop + 1)
    before = torch.arange(steps)[None, :] < stop[:, None]
    assert torch.equal(sampled[full, :steps][before], batch.drafts[full][before])
    replacement = sampled[full, stop]
    assert (replacement != batch.drafts[full, stop]).all()
    assert inputs["target_logits"][rows, replacement.to(rows.device)].isfinite().all()


# ---------------------------------------------------------------------------
# RejectionSampler wiring
# ---------------------------------------------------------------------------


def _wired_sampler(method, kernel_inputs, steps):
    sampler = SimpleNamespace(
        logprobs_mode="raw_logprobs",
        return_sampling_mask=False,
        sampling_states=SimpleNamespace(
            temperature=SimpleNamespace(gpu=kernel_inputs["temperature"]),
            seeds=SimpleNamespace(gpu=kernel_inputs["seed"]),
        ),
        use_fp64_gumbel=False,
        apply_sampling_params=lambda logits, *_: logits,
    )
    spec_config = SimpleNamespace(
        num_speculative_tokens=steps,
        enable_adaptive_verification=False,
        rejection_sample_method=method,
    )
    return RejectionSampler(sampler, spec_config, kernel_inputs["pos"].device)


def _wired_input_batch(kernel_inputs):
    cu = kernel_inputs["cu_num_logits"].cpu().numpy()
    idx = kernel_inputs["idx_mapping"].cpu().numpy()
    return SimpleNamespace(
        num_reqs=len(idx),
        cu_num_logits_np=cu,
        cu_num_logits=kernel_inputs["cu_num_logits"],
        idx_mapping_np=idx,
        idx_mapping=kernel_inputs["idx_mapping"],
        expanded_idx_mapping=kernel_inputs["expanded_idx_mapping"],
        expanded_local_pos=kernel_inputs["expanded_local_pos"],
        seq_lens_cpu_upper_bound=torch.full((len(idx),), 64, dtype=torch.int32),
    )


def _emitted(sampled, num_sampled):
    keep = torch.arange(sampled.shape[1], device=sampled.device)[None, :]
    return sampled[keep < num_sampled[:, None]]


@requires_cuda
@pytest.mark.parametrize("kind", ["sparse_fp32", "dense_bf16"])
def test_rejection_sampler_applies_the_configured_method(kind):
    """rejection_sample_method selects the rule, also through the request chunks.

    The block and standard samplers are driven on one batch with a sparse (DFlash2) or
    dense (DSpark) cache: each must equal the kernel run with the matching flag, one
    chunk or many (the cache is indexed by request state, not by chunk), and the two
    rules must differ while block accepts at least as much as standard.
    """
    steps = 7
    batch = _build_batch(
        kind=kind,
        num_reqs=2000,
        vocab=24,
        num_steps=steps,
        temperatures=[1.0, 0.6, 0.0],
        seed=5,
        candidates=8,
        placeholders=False,
    )
    inputs = batch.kernel_inputs
    input_batch = _wired_input_batch(inputs)
    results = {}
    for method in METHODS:
        rejection_sampler = _wired_sampler(method, inputs, steps)
        assert rejection_sampler.use_block_verification == (method == "block")
        expected = rejection_sample(
            **inputs,
            num_speculative_steps=steps,
            use_block_verification=method == "block",
        )
        for limit in (10_000, 37):
            sampled, num_sampled, logprobs, masks = rejection_sampler._verify_in_chunks(
                inputs["target_logits"],
                input_batch,
                inputs["draft_logits"],
                inputs["draft_sampled"],
                inputs["pos"],
                limit,
                NO_LOGPROBS,
            )
            assert logprobs is None and masks is None
            assert torch.equal(num_sampled, expected[1]), (method, limit)
            assert torch.equal(
                _emitted(sampled, num_sampled), _emitted(*expected)
            ), (method, limit)
        results[method] = expected[1]
    assert not torch.equal(results["standard"], results["block"])
    mean = {m: (n - 1).float().mean().item() for m, n in results.items()}
    assert mean["block"] >= mean["standard"] - 0.01, mean
