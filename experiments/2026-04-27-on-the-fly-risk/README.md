# 2026-04-27 — On-the-fly systematic risk over a weekly history

**Question.** Given W ≈ 104 weekly factor covariance matrices Σ_t (size N × N,
N from 500 to 4000) and a user-supplied factor exposure vector x, can we
compute the per-week systematic variance σ²_t = xᵀ Σ_t x interactively
(<200 ms end-to-end) as the user edits x?

**Headline finding (post Phase-3 einsum fix).** Direct evaluation is
**interactive at every N tested** — 4 ms at N=500 to 160 ms at N=4000,
all bandwidth-bound. The earlier "18 s at N=4000" was numpy's
contraction-path optimiser picking a pathological order for our
einsum, not the math itself; replacing the call with `(S @ x) · x`
gave a ~110× speedup. The per-week eigendecomposition (Phase 2) is
still a useful path because the eig-only artefact is ~40× smaller on
disk (160 MB vs 6.6 GB at N=4000). See [report.md](report.md).

**Status.** Done — Phases 1-3 (math + UI + bug fix) plus Phase 4
(Zarr-on-S3 simulation, real MinIO calibration) and Phase 5 (LRU
chunk cache wired into the production API) all shipped on `main`.

## Files

- [spec.md](spec.md) — what we set out to test, design choices, ruled-out paths.
- [plan.md](plan.md) — implementation steps, frozen as executed.
- [report.md](report.md) — numerical findings + recommendation (filled in after the bench).
- `results/` — raw bench output (CSV, screenshots).

## Code locations

- Snapshot extension — `backend/build_snapshot.py` (`gen_weekly_covariance`).
- Backend endpoint — `backend/api.py` (`POST /api/risk/quadratic`).
- Bench harness — `backend/bench/quadratic_bench.py`.
- Frontend page — `frontend/src/charts/RiskOverTime.tsx`.

## Re-run

```bash
# 1. Rebuild the snapshot (synthetic, parameterised by N + W).
cd backend
FRV_COV_N=2000 FRV_COV_WEEKS=104 uv run python build_snapshot.py

# 2. Start the backend.
uv run uvicorn api:app --reload --port 8000

# 3. Run the bench.
uv run python -m bench.quadratic_bench --n 500 1000 2000 4000 --modes full approx
```
