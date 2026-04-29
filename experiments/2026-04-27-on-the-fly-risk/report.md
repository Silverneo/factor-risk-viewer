# Report — On-the-fly systematic risk over a weekly history

## Headline

For 104 weekly N×N covariance matrices, the **direct** quadratic form
σ²_t = xᵀ Σ_t x is interactive at every N tested up to N=4000
(**160 ms per query at N=4000** — the bandwidth wall, not a code wall).
Interactivity at large N comes from a single fix: replace
`np.einsum('i,wij,j->w', x, S, x, optimize=True)` with
`(S @ x) · x` (batched matmul + row-wise dot). einsum's contraction-path
optimiser picks a memory-thrashing order for this signature; the BLAS
path runs **~110× faster** at N=4000 with no functional change.

A **per-week eigendecomposition** to top-k components is still useful
for shrinking the deployed artefact (160 MB vs 6.6 GB at N=4000) and
gives modest extra speed (~8× over the now-fixed full mode at N=4000,
not the ~700× that the einsum path implied). At the synthetic model's
intrinsic rank (K=30) it delivers **0.05 % max relative error** at every
N tested. Setting `k=30` is the sweet spot — going below the true rank
(k=10) blows error up to ~30 %; going above it adds compute without
improving accuracy.

## Phase 1 vs Phase 2 vs Phase 3

- **Phase 1**: full-mode endpoint + frontend + bench against a
  "shared base loadings" approximation (a useful upper bound on
  speed but unrealistic for production).
- **Phase 2**: precomputed per-week (V_t, D_t) eigendecomposition,
  `mode=approx` query path with realistic accuracy numbers.
- **Phase 3**: full-mode einsum bug fix. ~110× speedup at N=4000.
  Approx mode's relative advantage shrinks but the eig precompute is
  still worthwhile because the eig-only artefact is ~40× smaller on
  disk.

This report reflects post–Phase-3 numbers throughout.

## Numbers

Bench machine: Windows 11, single-process Python 3.11, NumPy 2.4 with
default BLAS. 5 iterations after 1 warm-up, median latency reported.

Source CSV: [`results/quadratic_bench_20260429T091717.csv`](results/quadratic_bench_20260429T091717.csv)
(post Phase-3 fix; pre-Phase-3 numbers in
[`quadratic_bench_20260428T085112.csv`](results/quadratic_bench_20260428T085112.csv)
for the historical record).

### Latency (post Phase-3 fix)

| N    | W   | Mode                  | p50 (ms)   | min (ms)   | speedup vs full |
|-----:|----:|-----------------------|-----------:|-----------:|----------------:|
| 500  | 104 | full                  |      3.80  |      3.73  |  1× (baseline)  |
| 500  | 104 | approx-eig k=10       |      1.71  |      1.70  |          ~2.2×  |
| 500  | 104 | approx-eig k=30       |      2.39  |      2.33  |          ~1.6×  |
| 500  | 104 | approx-eig k=100      |      5.34  |      5.22  |        ~0.7×    |
| 1000 | 104 | full                  |     12.07  |     12.02  |  1×             |
| 1000 | 104 | approx-eig k=10       |      2.58  |      2.46  |          ~4.7×  |
| 1000 | 104 | approx-eig k=30       |      4.29  |      4.22  |          ~2.8×  |
| 1000 | 104 | approx-eig k=100      |      9.94  |      9.85  |          ~1.2×  |
| 2000 | 104 | full                  |     33.13  |     31.91  |  1×             |
| 2000 | 104 | approx-eig k=10       |      4.98  |      4.83  |          ~6.6×  |
| 2000 | 104 | approx-eig k=30       |      8.18  |      8.06  |          ~4.0×  |
| 2000 | 104 | approx-eig k=100      |     19.89  |     19.64  |          ~1.7×  |
| 4000 | 104 | full                  |    160.00  |    156.96  |  1×             |
| 4000 | 104 | approx-eig k=10       |     12.54  |     12.41  |         ~12.8×  |
| 4000 | 104 | approx-eig k=30       |     20.47  |     19.43  |         **~7.8×** |
| 4000 | 104 | approx-eig k=50       |     31.37  |     29.81  |          ~5.1×  |
| 4000 | 104 | approx-eig k=100      |     49.43  |     42.24  |          ~3.2×  |

#### What Phase-3 changed

The full-mode column above is roughly two orders of magnitude better
than what the same code reported in Phase 2:

| N    | full p50 pre-fix | full p50 post-fix | speedup | bandwidth (post) |
|-----:|------------------:|-------------------:|--------:|-----------------:|
| 500  |        30.6 ms    |          3.8 ms    |  ~8×    |   28 GB/s        |
| 1000 |       129.9 ms    |         12.1 ms    | ~11×    |   34 GB/s        |
| 2000 |       820.0 ms    |         33.1 ms    | ~25×    |   47 GB/s        |
| 4000 |    18 003   ms    |        160.0 ms    | ~113×   |   37 GB/s        |

The post-fix bandwidth is at or near peak DDR4 sequential — full mode is
now genuinely memory-bound, no algorithmic headroom left on CPU.

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

1. **`mode=full` is interactive at every N tested** post Phase-3. If
   your storage can hold `cov_full` (local disk or on-demand S3
   stream), that's the simplest path — exact answers, ~160 ms at
   N=4000, no precompute step.

2. **`mode=approx` is still the win when storage is the constraint.**
   The eig-only artefact is ~160 MB at N=4000 vs ~6.6 GB for cov_full —
   ~40× smaller. That matters for serverless / Lambda / Vercel-style
   deploys with small ephemeral disk, or when you want to colocate
   many tenants' artefacts in RAM. Default `k = K_macro_factors`
   (e.g. k=30 for our synthetic model — pick the smallest k that
   recovers ≥ 99 % of variance on real data).

3. **Don't ship the shared-loadings shortcut.** It looks fast but loses
   3 % accuracy; the per-week eig path is just as cheap (per query)
   and 50–100× more accurate.

4. **Profile your einsum calls.** If anything in the codebase still
   uses `np.einsum(... optimize=True)` on a 3-tensor contraction with
   shape (W, N, N) at large N, replace it with explicit batched
   matmul. The 110× we recovered here was a one-line code change with
   identical numerics.

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
