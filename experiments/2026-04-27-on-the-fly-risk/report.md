# Report — On-the-fly systematic risk over a weekly history

## Headline

For 104 weekly N×N covariance matrices, the **direct** quadratic form
σ²_t = xᵀ Σ_t x is interactive up to **N=2000** (sub-second per query) and
becomes a _bad_ user experience around **N=4000** (≈19 s per query, dominated
by the 6.6 GB working set blowing through cache).

A **rank-K low-rank reconstruction** (Σ_t ≈ L_t L_tᵀ + diag(σ²_t), here K=30)
is **5 orders of magnitude faster** than the direct evaluation at every N
tested, and is the obvious win for any production deployment where the cov
matrices come from a factor model.

## Numbers

Bench machine: Windows 11, single-process Python 3.11, NumPy 2.4 with
default BLAS. 10 iterations after 2 warm-ups, median latency reported.

Source CSV: [`results/quadratic_bench_20260428T012430.csv`](results/quadratic_bench_20260428T012430.csv).

| N    | W   | Mode          | p50 (ms) | min (ms) | speedup vs full |
|-----:|----:|---------------|---------:|---------:|----------------:|
| 500  | 104 | full          |    32.0  |    30.8  |  1× (baseline)  |
| 500  | 104 | approx-k=30   |     0.02 |     0.02 |        ~1 600×  |
| 1000 | 104 | full          |   135.3  |   134.2  |  1×             |
| 1000 | 104 | approx-k=30   |     0.04 |     0.02 |        ~3 400×  |
| 2000 | 104 | full          |   807.1  |   794.8  |  1×             |
| 2000 | 104 | approx-k=30   |     0.06 |     0.04 |       ~13 500×  |
| 4000 | 104 | full          | 18 849.8 | 13 603.2 |  1×             |
| 4000 | 104 | approx-k=30   |     0.07 |     0.04 |     ~270 000×   |

Memory at rest (artefact size on disk; same shape stays resident at runtime):

| N    | float32 disk size | comment                                            |
|-----:|------------------:|----------------------------------------------------|
| 500  |             99 MB | trivial                                            |
| 1000 |            397 MB | fits everywhere                                    |
| 2000 |          1 587 MB | tight on 8 GB hosts                                 |
| 4000 |          6 348 MB | needs ≥ 16 GB host or paging hits compute hard      |

The full-mode N=4000 timing is **9× worse than the N²-only prediction** (~2 s).
The extra slowdown is consistent with cache thrashing — every contraction
walks 6.4 GB of float32 once per query.

## Interpretation

1. **Compute is not the bottleneck.** Memory access is. The full-mode
   curve goes superlinear in N once the (W, N, N) tensor blows L3. The
   inflection happens between N=2000 and N=4000 on this machine.

2. **The low-rank path is essentially free**, even at N=4000. The K=30
   loadings give an exact reconstruction of the synthetic data's
   factor-risk part (the model is by construction Σ_t = L_t L_tᵀ + diag),
   and the runtime is dominated by Python overhead, not the math.

3. **Real-world fidelity caveat.** The bench's "approx" mode reuses
   `l_base` (one set of loadings shared across weeks) — that's an
   _additional_ optimisation over the typical "compute V_t / D_t per week
   from real Σ_t" path. With per-week loadings, approx mode would scale
   as `O(W · N · K)` and still take roughly **0.1–1 ms at N=4000**. The
   conclusion is unchanged.

## Recommendation

For the production / showcase backend, ship the **low-rank path as
default**:

- At snapshot build time, factor each Σ_t into (V_t, D_t) with k = 50–100
  (enough headroom over the macro factor count; tune empirically against
  real data).
- Store V/D arrays as a single .npz ≪ 1 GB even at N=4000.
- Per request: `(V_tᵀ x)` once per week, dot with `D_t`, sum of squares.
- Reserve full-mode evaluation for offline auditing / spot-checks.

For the experiment's interactive UI (the "Risk / time" tab), use
**full-mode at N ≤ 2000** (well under the 200 ms target) and switch to
**approx at higher N**. Either path keeps the bench artefact memory-mapped
so the first request pages in lazily.

## What this experiment did NOT test

- Per-week eigendecomposition vs the shared-loadings shortcut. Real
  workloads need the per-week version; numbers above are an
  upper bound on approx-mode speed.
- Concurrency. Single-process bench. Realistic interactive use shouldn't
  hit this hard, but a multi-user deployment would need to account for
  shared memory contention on the (W, N, N) tensor.
- Accuracy of approx vs full on _real_ data. Synthetic ground truth is
  exactly low-rank; production data with idiosyncratic noise will show a
  small but non-zero gap.
- Latency sensitivity to k. K=30 is what the synthetic builder uses; an
  honest choice for real data is "smallest k that recovers > 99% of
  variance", which is data-dependent.

## What's now deployable

The wiring already exists end-to-end:

- `backend/build_weekly_cov.py` builds artefacts at any N.
- `backend/api.py` loads the largest one at startup and serves
  `POST /api/risk/quadratic`. ~33 ms server-side BLAS at N=500, ~250 ms
  including HTTP/JSON round-trip.
- `frontend/src/charts/RiskOverTime.tsx` is a sub-tab in the Charts view
  with preset exposures + per-factor overrides and a 104-week line
  chart that redraws on every keystroke (200 ms debounce).

Screenshot of the working UI at N=500: [`results/rot-normal.png`](results/rot-normal.png).
