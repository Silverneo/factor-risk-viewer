# Plan — On-the-fly systematic risk over a weekly history

Frozen execution plan as it was actually run.

1. **Synthetic snapshot extension.** New script
   `backend/build_weekly_cov.py` that emits `weekly_cov_N{n}_W{w}.npz`
   files at parameterised N. Synthetic factor model:
   `Σ_t = L_t L_tᵀ + diag(σ²_t)` with K=30 latent macros and small
   per-week drift.

2. **Backend endpoint.** `POST /api/risk/quadratic` and
   `GET /api/risk/quadratic/info` added to `backend/api.py`. The largest
   matching artefact in `backend/` is memory-mapped at FastAPI startup
   (override with `FRV_WEEKLY_COV` env var). Request body accepts either
   a dense `exposures: list[float]` aligned to the artefact's factor order
   _or_ a sparse `exposures_by_factor: dict[str, float]`. Compute is a
   single `np.einsum('i,wij,j->w', x, Σ, x)` over the requested week
   range.

3. **Bench harness.** `backend/bench/quadratic_bench.py` runs both
   "full" (direct einsum) and "approx-k=K" (low-rank via base loadings)
   at N ∈ {500, 1000, 2000, 4000}, W=104, 10 iters + 2 warm-ups, writes
   the per-iter rows to `results/quadratic_bench_<ts>.csv` and prints a
   median summary.

4. **Frontend UI.** `frontend/src/charts/RiskOverTime.tsx` — new sub-tab
   in the Charts view. Loads `info` once for factor order; user picks
   one of four presets (`unit`, `normal`, `sparse`, `zero`) plus
   per-factor text overrides; debounced (200 ms) POST to
   `/api/risk/quadratic`; AG Charts line series of σ_t with min/max/mean
   summary tiles.

5. **Verification.**
   - `tsc -b` clean.
   - `frontend/tests/risk-over-time.spec.ts` (Playwright) — 4 tests, one
     per preset, screenshot per preset under `results/rot-*.png`.
   - Manual sanity check: roundtrip < 50 ms at N=500 in the browser.

## Things deliberately not done in this pass

- Per-week loadings in the artefact (would let "approx" mode track Σ_t
  drift exactly). Saved for v2 — current bench is an upper bound on
  approx-mode speed, not a fidelity claim.
- A streaming response that emits weeks as they're computed. Not needed
  at the latency we measured.
- Auth / rate limiting on the new endpoint. Backend assumed local-only
  for the experiment.
- Disk-mmap performance comparison vs full RAM load. The artefact loader
  uses `mmap_mode='r'`; the bench uses RAM-resident loads on purpose to
  isolate compute from I/O.
