# Muse Spark static expert placement: table builder

Makes `vllm/models/muse_spark/nvidia/data/expert_placement_<snapshot>.json` (see
`vllm/models/muse_spark/nvidia/ops/expert_placement.py` for what the table is used for).
The four scripts are the ones the table's `provenance.built_by_sha256` names; a unit test
checks the hashes, so edit them only together with a rebuilt table.

Pipeline (numpy + numba, CPU only; the routing trace is not in the tree):

1. Record a routing trace on the model (ids of every layer per step, prefill and decode,
   EP8, eager, prefix caching off, `PREFILL_OVERLAP=0`), natural-text domains: gsm, chat,
   code, docs, zh, bio (+ eval-only sets). The recorder is the S1-5 `routing_trace` tool,
   not shipped.
2. `r2d_agg.py <trace_dir> <agg_dir>`: per-domain arrays (4096-row prefill windows with
   routes per expert, decode rows, real pure-decode batches).
3. `r2d_cv.py <agg_dir> <cv_dir> --variants comb --final`: leave-one-domain-out
   cross-validation and the final table (`ranks_comb_final.npy`, `cv_report.json`). The
   objective per layer is `T_prefill / 721.7 + T_c1 / 8.623 + T_c64 / 41.977` (every
   regime weighted by 1 / its production step time, every domain equal); the optimizer
   is an LPT start, swap annealing and a greedy polish (`r2d_lib.py`).
4. `r2d_table.py build <cv_dir>/ranks_comb_final.npy <cv_dir>/cv_report.json <agg_dir>
   <trace_dir> <out.json> --variant comb`, then `r2d_table.py check <out.json>`.

## How the table is applied (and two stale texts)

The table is applied as the layer's per-rank `expert_map` (global expert id -> local slot,
or -1), installed before the weights load; **nothing remaps the routing ids** (no id
remap kernel, no use of `new_id` at run time: `new_id` is only the numbering of the slots,
`rank * 32 + ascending slot`, and the loader derives the slot itself).

Two texts that the table builder wrote still describe the first design of the item (an id
remap after the routing, with the kernels seeing `new_id` through `local_expert_offset =
ep_rank x 32`). They are kept as they are, because `provenance.built_by_sha256` pins the
scripts and the table file together, and a rebuild is needed to change them:

- the table's `semantics` field ("apply the remap there only if their path allows it");
- the docstring of `r2d_table.py`.

Do not add an id remap on top of the installed map: it would remap twice. The behavior is
in `vllm/models/muse_spark/nvidia/ops/expert_placement.py` and the gates are the
unit tests next to it.

## End-to-end checks (not in the tree)

The unit tests cover the table, the install, every expert path's use of the map and the
constructor wiring (`tests/kernels/core/test_muse_spark_expert_placement*.py`,
`tests/kernels/moe/test_muse_spark_sm120_expert_placement.py`, the placement cases of
`test_muse_spark_sm120_sequence_parallel.py`). What they cannot do is build the 62-layer model
with the real weights; that is checked on the 8-GPU box with the flag on:

- every EP rank logs `static expert placement (VLLM_MUSE_SPARK_SM120_EXPERT_PLACEMENT=1) in 62
  of 62 MoE layers`, and no `keeps its ... expert placement` warning appears (a layer that
  stayed linear would show there);
- two cold processes with the flag on give bit-identical logprobs and tokens (the placement
  is deterministic; against the flag off the outputs differ at the rounding level only);
- the GSM8K smoke stays inside the usual 95.5-98.0 % band.
