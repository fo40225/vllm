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
