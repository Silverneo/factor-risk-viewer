"""Generate a synthetic weekly covariance history for the on-the-fly risk
experiment.

Output is a single `.npz` per (N, W) combination, written to backend/. The
backend memory-maps it at startup and exposes /api/risk/quadratic. The file
is gitignored — it can be regenerated cheaply (a few seconds even at
N=4000) so we don't pollute the repo with binary data.

Run:
    cd backend
    uv run python build_weekly_cov.py --n 2000 --weeks 104
    uv run python build_weekly_cov.py --n 500 1000 2000 4000 --weeks 104

The structure intentionally mirrors a real factor model:
  Σ_t  =  L_t L_tᵀ  +  diag(σ²_t)
where L_t ∈ ℝ^{N×K} drifts slowly week-over-week (so risk numbers move but
matrices remain related). K is small (latent factors) so the matrices are
low-rank-plus-diagonal — which means the approx mode in the bench should
recover near-perfect accuracy with k ≈ K.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

DEFAULT_N_LATENT = 30
DEFAULT_SEED = 7919
HERE = Path(__file__).parent


def weekly_cov_path(n: int, weeks: int) -> Path:
    return HERE / f"weekly_cov_N{n}_W{weeks}.npz"


def gen_weekly_cov(
    n: int,
    weeks: int,
    n_latent: int = DEFAULT_N_LATENT,
    seed: int = DEFAULT_SEED,
    drift_std: float = 0.04,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (cov_full, factor_ids, week_dates, eig_V, eig_D, latent_K).

    cov_full   : (W, N, N) float32 — dense weekly covariance matrices
    factor_ids : (N,)      object  — synthetic factor ids "F0001" … "FNNNN"
    week_dates : (W,)      str     — ISO date strings, weekly back from latest
    """
    rng = np.random.default_rng(seed)
    # Base loadings (N × K). Latent macro factors are independent N(0, 1).
    l_base = rng.normal(0.0, 1.0 / np.sqrt(n_latent), (n, n_latent)).astype(np.float32)
    # Small specific (idiosyncratic) variance per leaf factor.
    spec_base = rng.uniform(0.0001, 0.0010, n).astype(np.float32)

    cov_full = np.empty((weeks, n, n), dtype=np.float32)
    for t in range(weeks):
        if t == 0:
            l_t = l_base
            spec_t = spec_base
        else:
            drift_rng = np.random.default_rng(seed + 5000 * t)
            l_t = l_base + drift_rng.normal(0.0, drift_std / np.sqrt(n_latent), l_base.shape).astype(np.float32)
            spec_t = np.maximum(spec_base + drift_rng.normal(0.0, 0.0001, n).astype(np.float32), 1e-6)
        cov = (l_t @ l_t.T).astype(np.float32)
        cov[np.diag_indices(n)] += spec_t
        cov_full[t] = cov

    factor_ids = np.array([f"F{i:05d}" for i in range(n)], dtype=object)

    # Week dates: latest = 2026-04-26, going back weekly.
    end = np.datetime64("2026-04-26")
    week_offsets = np.arange(weeks - 1, -1, -1, dtype=int)
    week_dates_np = end - np.timedelta64(7, "D") * week_offsets
    week_dates = np.array([str(d) for d in week_dates_np], dtype=object)

    return cov_full, factor_ids, week_dates, l_base, spec_base


def write_artefact(out_path: Path, n: int, weeks: int, n_latent: int, seed: int) -> None:
    if out_path.exists():
        out_path.unlink()
    print(f"generating N={n} W={weeks} (latent={n_latent})…", flush=True)
    t0 = time.perf_counter()
    cov_full, factor_ids, week_dates, l_base, spec_base = gen_weekly_cov(
        n=n, weeks=weeks, n_latent=n_latent, seed=seed,
    )
    gen_secs = time.perf_counter() - t0

    # Eigendecomposition of cov_full[0] for the bench's "approx" mode.
    # Storing the top-k eigenpairs across weeks is a v2 optimisation; for now
    # we save the latent loadings (`l_base`) which gives an *exact* low-rank
    # part of Σ_t since Σ_t = L_t L_tᵀ + diag. The bench can use these to
    # validate accuracy.

    # Float32 dense saves ~2x vs float64. We accept the rounding for risk-vis
    # purposes (basis-point precision is more than enough).
    np.savez(
        out_path,
        cov_full=cov_full,
        factor_ids=factor_ids,
        week_dates=week_dates,
        l_base=l_base,
        spec_base=spec_base,
        meta=np.array([n, weeks, n_latent, seed], dtype=np.int64),
    )
    write_secs = time.perf_counter() - t0 - gen_secs
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"  wrote {out_path.name}  shape=(W={weeks}, N={n})  "
        f"size={size_mb:.1f} MB  gen={gen_secs:.1f}s  io={write_secs:.1f}s",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build synthetic weekly covariance artefacts")
    ap.add_argument("--n", type=int, nargs="+", default=[500],
                    help="Factor universe size(s). Default: 500. Pass multiple to build several.")
    ap.add_argument("--weeks", type=int, default=104, help="Number of weekly snapshots (default: 104).")
    ap.add_argument("--latent", type=int, default=DEFAULT_N_LATENT, help="Latent factor count (default: 30).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    for n in args.n:
        path = weekly_cov_path(n, args.weeks)
        write_artefact(path, n=n, weeks=args.weeks, n_latent=args.latent, seed=args.seed)


if __name__ == "__main__":
    main()
