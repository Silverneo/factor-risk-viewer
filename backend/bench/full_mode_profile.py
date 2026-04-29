"""Why is full mode so much slower than the FLOP budget says it should be?

Hypothesis (from the report): einsum's contraction path stops fitting cache
once Σ_t exceeds L3 (≈ N=2000), and the apparent N² scaling becomes worse
than N². To confirm, this script:

  1. Tries five implementations of σ²_t = xᵀ Σ_t x for all weeks t.
  2. Runs at single-thread and max-thread BLAS settings.
  3. Reports wall-clock + bytes/s effective bandwidth.

Run:
    cd backend
    uv run python -m bench.full_mode_profile --n 2000 4000

Pre-req: weekly_cov_N{n}_W104.npz in backend/, built via build_weekly_cov.py.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def artefact_path(n: int, weeks: int = 104) -> Path:
    return HERE / f"weekly_cov_N{n}_W{weeks}.npz"


def _set_blas_threads(n: int) -> None:
    """Best-effort BLAS thread-count override. threadpoolctl gives the
    cleanest control; fall back to env vars if unavailable."""
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
        threadpool_limits(limits=n)
    except ImportError:
        # NOTE: env vars only take effect if BLAS reads them at import
        # time, so this is a weak fallback. Run with OMP_NUM_THREADS=1
        # in your shell for guaranteed single-thread.
        os.environ["OMP_NUM_THREADS"] = str(n)
        os.environ["OPENBLAS_NUM_THREADS"] = str(n)
        os.environ["MKL_NUM_THREADS"] = str(n)


# ---------- Variants -----------------------------------------------------

def variant_einsum_optimize(S: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Current backend implementation."""
    return np.einsum("i,wij,j->w", x, S, x, optimize=True)


def variant_einsum_no_optimize(S: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Same call without optimise=True — exposes the contraction-path
    selection cost."""
    return np.einsum("i,wij,j->w", x, S, x)


def variant_loop_dot(S: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Explicit per-slice loop with two BLAS calls per week.

    Each iteration is one GEMV (Σ_t @ x) and one DOT (x · y). Forces
    numpy to use BLAS routines on contiguous slices — the access pattern
    most likely to hit cache cleanly.
    """
    w = S.shape[0]
    out = np.empty(w, dtype=np.float64)
    for t in range(w):
        y = S[t] @ x
        out[t] = x @ y
    return out


def variant_batched_matmul(S: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Single batched matvec (S @ x) then row-wise dot. numpy treats the
    leading axis as a batch; calls BLAS GEMM under the hood for the
    matvec batch, which can be faster than a Python loop of GEMVs."""
    y_all = S @ x  # (W, N)
    return np.einsum("wn,n->w", y_all, x, optimize=True)


def variant_loop_einsum_per_slice(S: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Loop with einsum per slice — same access pattern as loop_dot but
    via numpy's own contraction. Mostly here as a sanity check that the
    overhead difference between einsum and explicit np.dot is small."""
    w = S.shape[0]
    out = np.empty(w, dtype=np.float64)
    for t in range(w):
        out[t] = np.einsum("i,ij,j->", x, S[t], x, optimize=True)
    return out


VARIANTS = [
    ("einsum_optimize", variant_einsum_optimize),
    ("einsum_no_optimize", variant_einsum_no_optimize),
    ("loop_dot", variant_loop_dot),
    ("batched_matmul", variant_batched_matmul),
    ("loop_einsum_per_slice", variant_loop_einsum_per_slice),
]


# ---------- Driver -------------------------------------------------------

def gen_exposure(n: int, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.3, n).astype(np.float32)
    for k in rng.choice(n, size=min(10, n // 10), replace=False):
        x[k] = rng.normal(0.0, 1.5)
    return x


def bench_variant(name: str, fn, S: np.ndarray, x: np.ndarray, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        _ = fn(S, x)
    times: list[float] = []
    for _ in range(iters):
        gc.collect()
        t0 = time.perf_counter()
        result = fn(S, x)
        times.append(time.perf_counter() - t0)
    times.sort()
    n_bytes = S.nbytes  # full sweep touches every byte once
    bw = n_bytes / times[len(times) // 2] / (1024**3)  # GB/s
    return {
        "name": name,
        "median_ms": times[len(times) // 2] * 1000,
        "min_ms": times[0] * 1000,
        "bandwidth_gbps": bw,
        "first_value": float(result[0]),
        "last_value": float(result[-1]),
    }


def run_for_size(n: int, threads: int, iters: int, warmup: int) -> list[dict]:
    path = artefact_path(n)
    if not path.exists():
        print(f"  [skip] missing artefact: {path.name}", flush=True)
        return []
    print(f"\n=== N={n}, threads={threads} ===", flush=True)
    npz = np.load(path, allow_pickle=True)
    S = npz["cov_full"]
    x = gen_exposure(n)
    print(f"  loaded {path.name}  shape={S.shape}  dtype={S.dtype}  size={S.nbytes/(1024**3):.2f} GB", flush=True)
    _set_blas_threads(threads)
    rows: list[dict] = []
    for vname, vfn in VARIANTS:
        # At N=4000 a single full pass already takes 10–20 s. Cap warmup
        # and iters so we don't sit here for half an hour.
        local_warmup = warmup if n <= 2000 else 1
        local_iters = iters if n <= 2000 else 3
        try:
            r = bench_variant(vname, vfn, S, x, iters=local_iters, warmup=local_warmup)
        except Exception as e:  # noqa: BLE001
            print(f"  {vname:<28} failed: {e}", flush=True)
            continue
        r["n"] = n
        r["threads"] = threads
        rows.append(r)
        print(
            f"  {vname:<28}  median={r['median_ms']:>10.2f}ms  "
            f"min={r['min_ms']:>10.2f}ms  bw={r['bandwidth_gbps']:>5.2f} GB/s  "
            f"first={r['first_value']:.4f}",
            flush=True,
        )
    del S, x, npz
    gc.collect()
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[2000, 4000])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 0],
                    help="Thread counts to test. 0 = let BLAS decide (max).")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=2)
    args = ap.parse_args()

    print(f"Python {sys.version.split()[0]}, NumPy {np.__version__}")
    print(f"OMP_NUM_THREADS env: {os.environ.get('OMP_NUM_THREADS', '<unset>')}")

    all_rows = []
    for threads in args.threads:
        for n in args.n:
            t_arg = threads if threads > 0 else 9999  # 9999 = let BLAS decide
            all_rows.extend(run_for_size(n, t_arg, args.iters, args.warmup))

    print("\n=== summary (median ms across iters) ===")
    print(f"  {'N':>5}  {'threads':>7}  {'variant':<28}  {'median_ms':>10}  {'bw_gbps':>8}")
    for r in all_rows:
        print(
            f"  {r['n']:>5}  {r['threads']:>7}  {r['name']:<28}  {r['median_ms']:>10.2f}  {r['bandwidth_gbps']:>8.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
