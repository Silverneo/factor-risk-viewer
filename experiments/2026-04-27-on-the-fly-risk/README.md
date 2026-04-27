# 2026-04-27 — On-the-fly systematic risk over a weekly history

**Question.** Given W ≈ 104 weekly factor covariance matrices Σ_t (size N × N,
N from 500 to 4000) and a user-supplied factor exposure vector x, can we
compute the per-week systematic variance σ²_t = xᵀ Σ_t x interactively
(<200 ms end-to-end) as the user edits x?

**Headline finding.** Direct evaluation is interactive up to **N=2000**
(<1 s) and bad UX at **N=4000** (~19 s, memory-bound). A rank-30 low-rank
reconstruction is **5 orders of magnitude faster** at every N tested and
is the right default for production. See [report.md](report.md).

**Status.** Done — backend endpoint, frontend UI, bench, and writeup are all on `main`.

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
