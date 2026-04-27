# Spec — On-the-fly systematic risk over a weekly history

## What we're testing

For a user-supplied factor exposure vector `x ∈ ℝ^N`, evaluate

> σ²_t = xᵀ Σ_t x  for t = 1 … W

across W ≈ 104 weekly snapshots and return the time series interactively.

We want to know:

1. **At what N does compute stop being trivial on a single CPU?** (we believe
   it's well above N=4000 for compute, but memory bites first)
2. **Does a low-rank approximation `Σ ≈ V D Vᵀ` with k ≪ N preserve the risk
   number to acceptable accuracy?** And what is "acceptable"?
3. **What's the right backend shape** — load all Σ_t into RAM at startup,
   stream from disk, or precompute V/D and ship those instead?

## In scope

- Synthetic Σ_t generation in `build_snapshot.py`, parameterised by N and W.
- Backend endpoint `POST /api/risk/quadratic` taking an exposure vector and
  returning `{weeks, systematic_var, systematic_vol}` for all W.
- Two compute modes: `full` (Σ_t directly) and `approx` (V_t D_t V_tᵀ with
  configurable k).
- Wall-clock + memory bench at N ∈ {500, 1000, 2000, 4000} for both modes.
- Frontend Risk-over-time page with editable exposures + line chart.

## Out of scope (for v1)

- GPU compute (CuPy/JAX). Revisit only if N=4000 + concurrency demands it.
- Client-side WASM compute. Possible v2 if low-rank artefacts are small.
- Long-horizon historical (>2 years). 104 weeks is the target.
- User-defined factor sets that change Σ on the fly (we use the snapshot's
  factor universe; user picks exposures over those).
- Multi-factor-model support — single factor model per snapshot.

## Success criteria

1. **Latency.** End-to-end p50 < 200 ms at N=2000, < 1 s at N=4000, on a
   single FastAPI worker on a laptop-class CPU.
2. **Memory.** Resident set size stays under 4 GB at N=2000 and under
   8 GB at N=4000 in `full` mode. In `approx` mode at k=100, under 1 GB
   for any N up to 4000.
3. **Approximation accuracy.** `approx` mode with k=100 has relative error
   < 1% on σ_t vs `full` mode for representative exposures, on the synthetic
   factor model in the snapshot.

## Non-goals

- Not chasing absolute peak FLOPs. BLAS via NumPy is sufficient.
- Not building an ergonomic exposure editor. A pasteable text vector or a
  basic table is fine for the experiment.

## Decisions ruled out before we started

- **Per-request matrix load from disk.** Reading 6 GB per query is too slow.
  Memory-resident is the floor.
- **Pre-aggregating x · Σ across all factor combinations.** Combinatorial
  blow-up for any non-trivial N.
- **Sparse storage.** Factor cov matrices are dense (a factor model implies
  every leaf factor has nontrivial loadings on the macro factors).

## Acceptance check

The experiment is complete when:
- The bench has run at every (N, mode, k) cell in the matrix.
- The report shows latency + memory + accuracy curves.
- A recommendation is written: which mode for which N range.
- The Risk-over-time page works against the chosen default.
