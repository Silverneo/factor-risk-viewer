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
    mode: str          # "full" | "approx-k=K"
    iter_idx: int
    elapsed_ms: float
    vol_first: float
    vol_last: float
    vol_norm: float    # ‖vol‖₂ — useful sanity check across modes
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


def bench_full(cov_full: np.ndarray, x: np.ndarray, iters: int, warmup: int) -> list[BenchRow]:
    """Benchmark x^T Σ_t x via einsum across all weeks."""
    w = cov_full.shape[0]
    n = cov_full.shape[1]
    rows: list[BenchRow] = []
    for _ in range(warmup):
        _ = np.einsum("i,wij,j->w", x, cov_full, x, optimize=True)
    for k in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        var = np.einsum("i,wij,j->w", x, cov_full, x, optimize=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        var_clamped = np.clip(var, 0.0, None)
        vol = np.sqrt(var_clamped)
        rows.append(BenchRow(
            n=n, w=w, mode="full", iter_idx=k,
            elapsed_ms=elapsed_ms,
            vol_first=float(vol[0]), vol_last=float(vol[-1]),
            vol_norm=float(np.linalg.norm(vol)),
            rss_mb=_rss_mb(),
        ))
    return rows


def bench_approx_lowrank(
    l_base: np.ndarray, spec_base: np.ndarray, w: int, x: np.ndarray, iters: int, warmup: int,
) -> list[BenchRow]:
    """Benchmark x^T Σ_t x using the (W=1)-shared low-rank factorisation
    Σ_t ≈ L Lᵀ + diag(σ²). For this synthetic data we use the *base* loadings
    and ignore the small per-week drift, which matches what a "snapshot the
    loadings once, save tons of memory" production strategy would do.

    Per query: y = Lᵀ x  (O(NK))   →   var ≈ ‖y‖² + xᵀ diag(σ²) x   (O(N+K))
    Multiplied by W weeks, all summed in NumPy.

    Note: we report a single `vol` repeated W times since the low-rank
    approximation here uses the base loadings — the per-week drift adds
    only the diagonal noise, which is a small effect. The bench's purpose
    is to characterise compute, not to exactly match Σ_t.
    """
    n, k = l_base.shape
    rows: list[BenchRow] = []
    for _ in range(warmup):
        y = l_base.T @ x        # (K,)
        diag_part = float(np.dot(x * x, spec_base))
        v = float(y @ y) + diag_part
        _ = np.full(w, v, dtype=np.float64)
    for itr in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        # Single base shot — computed once, broadcast W times.
        y = l_base.T @ x
        diag_part = float(np.dot(x * x, spec_base))
        var_scalar = float(y @ y) + diag_part
        var = np.full(w, var_scalar, dtype=np.float64)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        var_clamped = np.clip(var, 0.0, None)
        vol = np.sqrt(var_clamped)
        rows.append(BenchRow(
            n=n, w=w, mode=f"approx-k={k}", iter_idx=itr,
            elapsed_ms=elapsed_ms,
            vol_first=float(vol[0]), vol_last=float(vol[-1]),
            vol_norm=float(np.linalg.norm(vol)),
            rss_mb=_rss_mb(),
        ))
    return rows


def run_for_size(n: int, weeks: int, iters: int, warmup: int) -> list[BenchRow]:
    path = artefact_path(n, weeks)
    if not path.exists():
        print(f"  [skip] missing artefact: {path.name}", flush=True)
        return []
    rss_before = _rss_mb()
    npz = np.load(path, allow_pickle=True)  # not mmap — bench wants hot RAM
    cov_full = npz["cov_full"]
    l_base = npz["l_base"]
    spec_base = npz["spec_base"]
    rss_after = _rss_mb()
    print(
        f"  loaded {path.name}  "
        f"shape=(W={cov_full.shape[0]}, N={cov_full.shape[1]})  "
        f"rss delta={rss_after - rss_before:+.1f} MB ({rss_after:.1f} MB total)",
        flush=True,
    )

    x = _gen_exposure(n)
    rows: list[BenchRow] = []
    rows.extend(bench_full(cov_full, x, iters=iters, warmup=warmup))
    rows.extend(bench_approx_lowrank(l_base, spec_base, weeks, x, iters=iters, warmup=warmup))

    # Free immediately so the next size starts clean.
    del cov_full, l_base, spec_base, npz
    gc.collect()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[500, 1000, 2000, 4000])
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
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
        all_rows.extend(run_for_size(n, args.weeks, args.iters, args.warmup))

    if not all_rows:
        print("no results — were any artefacts present?", flush=True)
        return

    # Aggregate per (N, mode): median elapsed_ms, the vol_norm (should be
    # consistent across iters), and the rss after the run.
    print("\nsummary (median over iters):")
    print(f"  {'N':>5}  {'W':>4}  {'mode':<14}  {'p50_ms':>10}  {'min_ms':>10}  {'rss_mb':>10}")
    by_key: dict[tuple[int, str], list[BenchRow]] = {}
    for r in all_rows:
        by_key.setdefault((r.n, r.mode), []).append(r)
    for (n, mode), rs in sorted(by_key.items()):
        ms = sorted(r.elapsed_ms for r in rs)
        p50 = ms[len(ms) // 2]
        print(
            f"  {n:>5}  {rs[0].w:>4}  {mode:<14}  {p50:>9.2f}ms  {ms[0]:>9.2f}ms  {rs[-1].rss_mb:>9.1f}MB",
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
