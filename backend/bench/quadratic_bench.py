"""Wall-clock + memory bench for the on-the-fly systematic risk experiment.

Measures three things across N ∈ {500, 1000, 2000, 4000} and W=104:

1. **full** — direct `xᵀ Σ_t x` via numpy.einsum (BLAS GEMV). The naive
   in-memory approach.
2. **approx** — low-rank reconstruction `Σ ≈ V D Vᵀ` from the saved
   loadings + specific variance. Computes (Vᵀx)ᵀ D (Vᵀx) per week — O(NK).
3. **memory** — RSS before/after artefact load, so we can tell where the
   ceiling is for each N.

Output goes to results/<timestamp>.csv next to this script's experiment
folder. Run:

    cd backend
    uv run python -m bench.quadratic_bench
    uv run python -m bench.quadratic_bench --n 500 1000 --warmup 1 --iters 5

Pre-requisite: the relevant `weekly_cov_N{n}_W{w}.npz` artefacts must exist
(build via `uv run python build_weekly_cov.py --n N1 N2 ... --weeks W`).
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]  # backend/
EXPERIMENT_DIR = HERE.parents[0] / "experiments" / "2026-04-27-on-the-fly-risk"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def _rss_mb() -> float:
    """Resident set size in MB. psutil-free for portability — uses
    GetProcessMemoryInfo via ctypes on Windows, /proc/self/status on Linux,
    and ru_maxrss on macOS."""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ctypes.windll.psapi.GetProcessMemoryInfo(
                ctypes.windll.kernel32.GetCurrentProcess(),
                ctypes.byref(counters),
                counters.cb,
            )
            return counters.WorkingSetSize / (1024 * 1024)
        elif sys.platform == "darwin":
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        else:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        return 0.0
    except Exception:
        return 0.0


def artefact_path(n: int, weeks: int) -> Path:
    return HERE / f"weekly_cov_N{n}_W{weeks}.npz"


@dataclass
class BenchRow:
    n: int
    w: int
    mode: str          # "full" | "approx-shared-K" | "approx-eig-K"
    iter_idx: int
    elapsed_ms: float
    vol_first: float
    vol_last: float
    vol_norm: float    # ‖vol‖₂ — useful sanity check across modes
    max_rel_err: float # vs full, computed once per (N, mode); -1 for full
    rss_mb: float


def _gen_exposure(n: int, seed: int = 17) -> np.ndarray:
    """A non-trivial exposure vector — not all-ones and not a delta. Loadings
    on macro factors look like a real portfolio's residual exposures: small,
    mostly bounded, with a few outliers."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.3, n).astype(np.float32)
    # Sprinkle a handful of larger exposures to exercise tails.
    for k in rng.choice(n, size=min(10, n // 10), replace=False):
        x[k] = rng.normal(0.0, 1.5)
    return x


def _max_rel_err(approx_vol: np.ndarray, full_vol: np.ndarray) -> float:
    """max |a-b|/|b| across weeks, ignoring zero-vol entries."""
    mask = np.abs(full_vol) > 1e-10
    if not mask.any():
        return 0.0
    return float(np.max(np.abs(approx_vol[mask] - full_vol[mask]) / np.abs(full_vol[mask])))


def _full_quadratic(S: np.ndarray, x: np.ndarray) -> np.ndarray:
    """xᵀ Σ_t x via batched matmul + row-wise dot. This matches the
    backend's WeeklyCovStore.quadratic implementation; see api.py for the
    note on why this beats `np.einsum('i,wij,j->w', ...)` by ~100× at
    large N."""
    y_all = S @ x                       # (W, N)
    return (y_all * x).sum(axis=1)      # (W,)


def bench_full(cov_full: np.ndarray, x: np.ndarray, iters: int, warmup: int) -> tuple[list[BenchRow], np.ndarray]:
    """Benchmark x^T Σ_t x via batched matmul across all weeks. Also
    returns the full-mode vol vector so the approx benches can compute
    relative error against it."""
    w = cov_full.shape[0]
    n = cov_full.shape[1]
    rows: list[BenchRow] = []
    full_vol: np.ndarray | None = None
    for _ in range(warmup):
        _ = _full_quadratic(cov_full, x)
    for k in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        var = _full_quadratic(cov_full, x)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        var_clamped = np.clip(var, 0.0, None)
        vol = np.sqrt(var_clamped)
        if full_vol is None:
            full_vol = vol
        rows.append(BenchRow(
            n=n, w=w, mode="full", iter_idx=k,
            elapsed_ms=elapsed_ms,
            vol_first=float(vol[0]), vol_last=float(vol[-1]),
            vol_norm=float(np.linalg.norm(vol)),
            max_rel_err=-1.0,
            rss_mb=_rss_mb(),
        ))
    return rows, full_vol if full_vol is not None else np.zeros(w)


def bench_approx_shared(
    l_base: np.ndarray, spec_base: np.ndarray, w: int, x: np.ndarray, iters: int, warmup: int,
    full_vol: np.ndarray,
) -> list[BenchRow]:
    """Approximation using a single shared (l_base, spec_base) across all
    weeks — the strictest production shortcut. Doesn't track per-week drift
    of the loadings, so accuracy degrades when drift_std × √k is comparable
    to signal magnitudes."""
    n, k = l_base.shape
    rows: list[BenchRow] = []
    last_vol: np.ndarray | None = None
    for _ in range(warmup):
        y = l_base.T @ x
        diag_part = float(np.dot(x * x, spec_base))
        v = float(y @ y) + diag_part
        _ = np.full(w, v, dtype=np.float64)
    for itr in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        y = l_base.T @ x
        diag_part = float(np.dot(x * x, spec_base))
        var_scalar = float(y @ y) + diag_part
        var = np.full(w, var_scalar, dtype=np.float64)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        var_clamped = np.clip(var, 0.0, None)
        vol = np.sqrt(var_clamped)
        last_vol = vol
        rows.append(BenchRow(
            n=n, w=w, mode=f"approx-shared-k={k}", iter_idx=itr,
            elapsed_ms=elapsed_ms,
            vol_first=float(vol[0]), vol_last=float(vol[-1]),
            vol_norm=float(np.linalg.norm(vol)),
            max_rel_err=-1.0,  # filled in below
            rss_mb=_rss_mb(),
        ))
    if last_vol is not None:
        err = _max_rel_err(last_vol, full_vol)
        for r in rows:
            r.max_rel_err = err
    return rows


def bench_approx_eig(
    eig_V: np.ndarray, eig_D: np.ndarray, x: np.ndarray, iters: int, warmup: int,
    full_vol: np.ndarray, k_request: int,
) -> list[BenchRow]:
    """Approximation using per-week saved eigendecomposition with k_request
    leading components. This is the production-realistic path — what an
    actual snapshot-build pipeline would emit."""
    weeks, n, k_saved = eig_V.shape
    k_active = min(k_request, k_saved)
    V = eig_V[:, :, -k_active:]
    D = eig_D[:, -k_active:]
    rows: list[BenchRow] = []
    last_vol: np.ndarray | None = None
    for _ in range(warmup):
        y = np.einsum("wnk,n->wk", V, x, optimize=True)
        _ = np.einsum("wk,wk->w", D, y * y, optimize=True)
    for itr in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        y = np.einsum("wnk,n->wk", V, x, optimize=True)
        var = np.einsum("wk,wk->w", D, y * y, optimize=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        var_clamped = np.clip(var, 0.0, None)
        vol = np.sqrt(var_clamped)
        last_vol = vol
        rows.append(BenchRow(
            n=n, w=weeks, mode=f"approx-eig-k={k_active}", iter_idx=itr,
            elapsed_ms=elapsed_ms,
            vol_first=float(vol[0]), vol_last=float(vol[-1]),
            vol_norm=float(np.linalg.norm(vol)),
            max_rel_err=-1.0,
            rss_mb=_rss_mb(),
        ))
    if last_vol is not None:
        err = _max_rel_err(last_vol, full_vol)
        for r in rows:
            r.max_rel_err = err
    return rows


def run_for_size(n: int, weeks: int, iters: int, warmup: int, k_grid: list[int]) -> list[BenchRow]:
    path = artefact_path(n, weeks)
    if not path.exists():
        print(f"  [skip] missing artefact: {path.name}", flush=True)
        return []
    rss_before = _rss_mb()
    npz = np.load(path, allow_pickle=True)  # not mmap — bench wants hot RAM
    cov_full = npz["cov_full"]
    l_base = npz["l_base"]
    spec_base = npz["spec_base"]
    has_eig = "eig_V" in npz.files and "eig_D" in npz.files
    eig_V = npz["eig_V"] if has_eig else None
    eig_D = npz["eig_D"] if has_eig else None
    rss_after = _rss_mb()
    print(
        f"  loaded {path.name}  "
        f"shape=(W={cov_full.shape[0]}, N={cov_full.shape[1]})  "
        f"eig={'yes' if has_eig else 'no'}  "
        f"rss delta={rss_after - rss_before:+.1f} MB ({rss_after:.1f} MB total)",
        flush=True,
    )

    x = _gen_exposure(n)
    rows: list[BenchRow] = []
    full_rows, full_vol = bench_full(cov_full, x, iters=iters, warmup=warmup)
    rows.extend(full_rows)
    rows.extend(bench_approx_shared(l_base, spec_base, weeks, x, iters=iters, warmup=warmup, full_vol=full_vol))
    if has_eig and eig_V is not None and eig_D is not None:
        for k_req in k_grid:
            if k_req > eig_V.shape[2]:
                continue
            rows.extend(bench_approx_eig(eig_V, eig_D, x, iters=iters, warmup=warmup, full_vol=full_vol, k_request=k_req))

    del cov_full, l_base, spec_base, eig_V, eig_D, npz
    gc.collect()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--k", type=int, nargs="+", default=[10, 30, 50, 100],
                    help="k values to test for the per-week eig approx mode (default: 10 30 50 100).")
    ap.add_argument("--out", type=Path, default=None,
                    help="CSV path (default: experiments/.../results/<timestamp>.csv)")
    args = ap.parse_args()

    if args.out is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.out = RESULTS_DIR / f"quadratic_bench_{ts}.csv"

    print(f"weeks={args.weeks}  iters={args.iters}  warmup={args.warmup}", flush=True)
    print(f"out: {args.out}", flush=True)

    all_rows: list[BenchRow] = []
    for n in args.n:
        print(f"--- N={n} ---", flush=True)
        all_rows.extend(run_for_size(n, args.weeks, args.iters, args.warmup, args.k))

    if not all_rows:
        print("no results — were any artefacts present?", flush=True)
        return

    # Aggregate per (N, mode): median elapsed_ms, the vol_norm (should be
    # consistent across iters), and the rss after the run.
    print("\nsummary (median over iters):")
    print(f"  {'N':>5}  {'W':>4}  {'mode':<22}  {'p50_ms':>11}  {'min_ms':>11}  {'max_rel_err':>12}")
    by_key: dict[tuple[int, str], list[BenchRow]] = {}
    for r in all_rows:
        by_key.setdefault((r.n, r.mode), []).append(r)
    for (n, mode), rs in sorted(by_key.items()):
        ms = sorted(r.elapsed_ms for r in rs)
        p50 = ms[len(ms) // 2]
        err = rs[0].max_rel_err
        err_str = f"{err * 100:.4f}%" if err >= 0 else "n/a"
        print(
            f"  {n:>5}  {rs[0].w:>4}  {mode:<22}  {p50:>10.3f}ms  {ms[0]:>10.3f}ms  {err_str:>12}",
            flush=True,
        )

    with open(args.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        wr.writeheader()
        for r in all_rows:
            wr.writerow(asdict(r))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
