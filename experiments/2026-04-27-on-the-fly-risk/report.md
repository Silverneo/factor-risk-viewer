# Report — On-the-fly systematic risk over a weekly history

## Headline

For 104 weekly N×N covariance matrices, the **direct** quadratic form
σ²_t = xᵀ Σ_t x is interactive up to **N=2000** (~800 ms per query) and
becomes unusable at **N=4000** (~18 s, memory-bound on a 6.6 GB working
set).

A **per-week eigendecomposition** to top-k components is the right
production path: at the synthetic model's intrinsic rank (K=30, since
Σ_t = L_t L_tᵀ + diag, L_t is N×30) it delivers **0.05 % max relative
error** at every N tested, while running **13× faster at N=500 and 783×
faster at N=4000**. Setting `k=30` is the sweet spot — going below the
true rank (k=10) blows error up to ~30 %; going above it (k=50, k=100)
adds compute without improving accuracy.

## Phase 1 vs Phase 2

Phase 1 only had a "shared base loadings" approximation, which was a
useful upper bound on speed but unrealistic for production. Phase 2
adds a precomputed per-week (V_t, D_t) eigendecomposition stored in
the artefact, plus a `mode=approx` query path. This report supersedes
Phase 1's numbers.

## Numbers

Bench machine: Windows 11, single-process Python 3.11, NumPy 2.4 with
default BLAS. 5 iterations after 1 warm-up, median latency reported.

Source CSV: [`results/quadratic_bench_20260428T085112.csv`](results/quadratic_bench_20260428T085112.csv).

### Latency

| N    | W   | Mode                  | p50 (ms)   | min (ms)   | speedup vs full |
|-----:|----:|-----------------------|-----------:|-----------:|----------------:|
| 500  | 104 | full                  |     30.6   |     30.4   |  1× (baseline)  |
| 500  | 104 | approx-eig k=10       |      1.61  |      1.45  |          ~19×   |
| 500  | 104 | approx-eig k=30       |      2.41  |      2.32  |          ~13×   |
| 500  | 104 | approx-eig k=50       |      3.00  |      2.97  |          ~10×   |
| 500  | 104 | approx-eig k=100      |      4.97  |      4.94  |           ~6×   |
| 1000 | 104 | full                  |    129.9   |    129.4   |  1×             |
| 1000 | 104 | approx-eig k=10       |      2.72  |      2.69  |          ~48×   |
| 1000 | 104 | approx-eig k=30       |      4.15  |      4.12  |          ~31×   |
| 1000 | 104 | approx-eig k=100      |      9.80  |      9.66  |          ~13×   |
| 2000 | 104 | full                  |    820.0   |    791.0   |  1×             |
| 2000 | 104 | approx-eig k=10       |      8.37  |      8.19  |          ~98×   |
| 2000 | 104 | approx-eig k=30       |     11.28  |     11.04  |          ~73×   |
| 2000 | 104 | approx-eig k=100      |     23.07  |     22.21  |          ~36×   |
| 4000 | 104 | full                  | 18 002.8   | 16 174.3   |  1×             |
| 4000 | 104 | approx-eig k=10       |     16.18  |     15.53  |       ~1 113×   |
| 4000 | 104 | approx-eig k=30       |     23.02  |     22.22  |       **~783×** |
| 4000 | 104 | approx-eig k=50       |     29.28  |     28.87  |         ~614×   |
| 4000 | 104 | approx-eig k=100      |     46.10  |     45.06  |         ~390×   |

### Accuracy (max relative error vs full mode)

| N    | k=10     | k=30     | k=50     | k=100    | shared-k=30 (no per-week) |
|-----:|---------:|---------:|---------:|---------:|--------------------------:|
| 500  | 40.96 %  | 0.024 %  | 0.023 %  | 0.020 %  |  1.88 %                   |
| 1000 | 28.67 %  | 0.047 %  | 0.045 %  | 0.041 %  |  3.09 %                   |
| 2000 | 37.85 %  | 0.050 %  | 0.050 %  | 0.047 %  |  3.31 %                   |
| 4000 | 29.22 %  | 0.054 %  | 0.053 %  | 0.052 %  |  2.76 %                   |

Two cliffs are visible:

1. **Below the true rank (k=10 < K_latent=30)**: error explodes to
   ~30–40 %. The truncation drops eigenvectors that carry real signal.
2. **At or above the true rank (k ≥ 30)**: error collapses to numerical
   noise (~0.05 %). Adding more components past 30 buys nothing
   because the additional eigenvalues are essentially zero in this
   factor model.

The `shared-k=30` row confirms what Phase 1 hinted at: re-using a single
set of base loadings across all weeks is fast but loses ~3 % accuracy
because the loadings drift over time. Don't ship that path.

### Build cost (one-time, at snapshot time)

Per-week `np.linalg.eigh` of Σ_t (full O(N³), top-100 retained):

| N    | eigh time (s) | total artefact size |
|-----:|--------------:|--------------------:|
| 500  |          9.0  |         119 MB      |
| 1000 |         28.3  |         437 MB      |
| 2000 |         89.2  |       1 667 MB      |
| 4000 |        508.9  |       6 507 MB      |

The eig arrays themselves are tiny — at N=4000, 104 weeks × 4000 × 100
× 4 bytes = 159 MB. The bloat in the artefact is from the saved full
Σ_t we still keep around for the bench's `full` mode.

## Recommendation

For a production deployment of this app:

1. **Default to `mode=approx` with `k = K_macro_factors`** (e.g. k=30 for
   our synthetic model; tune empirically for real data — pick the
   smallest k that recovers ≥ 99 % of variance per week).

2. **Drop `cov_full` from the deployed artefact** — keep only the eig
   arrays. That shrinks the artefact from ~6.6 GB to ~160 MB at
   N=4000, fits in any backend host's RAM, and eliminates the
   memory-bound penalty.

3. **Keep `mode=full` available as an off-by-default audit path** —
   useful for spot-checking the approximation on demand, not for the
   hot interactive loop.

4. **Don't ship the shared-loadings shortcut.** It looks fast but loses
   3 % accuracy; the per-week eig path is just as cheap (per query)
   and 50–100× more accurate.

## What this experiment did NOT test

- **Real data.** The synthetic model is exactly low-rank (K=30) by
  construction; real factor cov matrices have a less clean spectrum,
  and the right k will be data-dependent.
- **k tuning method.** "Recover 99 % of variance" is the obvious
  starting heuristic but won't always be best. Future iteration:
  search k that minimises max rel error over a held-out exposure set.
- **Float32 vs float64 numerics at large N.** We promote to float64 for
  the eigh itself but store float32 — adequate here, may want re-test
  at extreme N or precision-sensitive use cases.
- **Concurrency.** Single-process bench. A multi-user deployment would
  need to validate that mmap'd reads don't serialise on the OS page
  cache; spot evidence suggests they don't, but it isn't measured.

## What's deployable now

- `backend/build_weekly_cov.py --eig-k 100` builds artefacts with both
  the full Σ_t (for the bench) and the per-week (V_t, D_t) (for
  approx mode). Pass `--eig-k 0` to skip the eigh step entirely if
  you want a smaller artefact.
- `backend/api.py` — `POST /api/risk/quadratic` accepts `mode: 'full'`
  or `mode: 'approx'` plus an optional `k` (capped at the saved k).
  503 if approx is requested but the artefact has no eig arrays.
- `backend/bench/quadratic_bench.py` — sweeps full / shared / per-week-k
  across multiple N values; accuracy and latency in one run.
- `frontend/src/charts/RiskOverTime.tsx` — Risk / time tab supports
  `Full` / `Approx` / `Both` modes plus a k selector. In `Both` the
  chart overlays both lines and a "max rel err" stat tile shows the
  numerical agreement at a glance.

Working comparison at N=500, k=100 (Both mode):
[`results/rot-normal.png`](results/rot-normal.png) — full and approx lines
visually overlap, max rel err 0.025 %.
