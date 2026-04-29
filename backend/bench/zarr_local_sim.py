"""Simulate S3-style chunk-fetch latency on a local Zarr store.

Converts an existing weekly_cov_*.npz to a local Zarr group, then benches
mode=full and mode=approx queries through a wrapper store that injects
first-byte latency + a bandwidth cap on every chunk fetch — modelling
how Zarr+S3 would behave without needing real network or AWS account.

Run:
    cd backend
    uv run python -m bench.zarr_local_sim --n 1000 --weeks 104

Outputs a CSV under
experiments/2026-04-27-on-the-fly-risk/results/zarr_sim_<ts>.csv

Pre-req: weekly_cov_N{n}_W{w}.npz exists (build via build_weekly_cov.py).

Notes on what it simulates:
  - Per-GET first-byte latency (exponential around p50_ms; min 0.1 × p50)
  - Bandwidth cap (post-headers throughput, applied as time.sleep on bytes)
  - Each query is treated as cold-cache (no chunk reuse between iters).
    For warm-LRU behaviour the production code would wrap the store with
    a chunk cache; we don't model that here because the cold path is
    what tells us how big a cache we need.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gc
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import zarr
from zarr.storage import LocalStore, WrapperStore

HERE = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = HERE.parents[0] / "experiments" / "2026-04-27-on-the-fly-risk"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def npz_path(n: int, weeks: int) -> Path:
    return HERE / f"weekly_cov_N{n}_W{weeks}.npz"


def zarr_path(n: int, weeks: int) -> Path:
    return HERE / f"weekly_cov_N{n}_W{weeks}.zarr"


# ---------- LatencyStore -------------------------------------------------

class _Counters:
    """Shared mutable state — zarr's WrapperStore can clone the wrapper
    via with_read_only(), so counters live in a separate object that
    every clone references, not on the wrapper instance itself."""
    def __init__(self) -> None:
        self.fetches = 0
        self.bytes_fetched = 0
        self.network_seconds = 0.0

    def reset(self) -> None:
        self.fetches = 0
        self.bytes_fetched = 0
        self.network_seconds = 0.0


class LatencyStore(WrapperStore):
    """Wraps any zarr store, charging per-GET first-byte latency + a
    bandwidth ceiling. Rough-and-ready model of S3 / object-store
    behaviour. Counters track the network volume per query so we can
    sanity-check it matches the chunk math."""

    def __init__(self, store, p50_ms: float = 0.0, throughput_mbps: float = 0.0,
                 counters: _Counters | None = None):
        super().__init__(store)
        self.p50_ms = float(p50_ms)
        # 0 = unlimited
        self.throughput_mbps = float(throughput_mbps)
        self.counters = counters if counters is not None else _Counters()

    # zarr's WrapperStore.with_read_only() rebuilds via type(self)(store=...)
    # — preserve our latency knobs *and* the shared counters through that.
    def _with_store(self, store):
        return type(self)(
            store,
            p50_ms=self.p50_ms,
            throughput_mbps=self.throughput_mbps,
            counters=self.counters,
        )

    def reset_counters(self) -> None:
        self.counters.reset()

    @property
    def fetches(self) -> int:
        return self.counters.fetches

    @property
    def bytes_fetched(self) -> int:
        return self.counters.bytes_fetched

    @property
    def network_seconds(self) -> float:
        return self.counters.network_seconds

    async def _delay(self, n_bytes: int) -> None:
        if self.p50_ms > 0:
            await asyncio.sleep(self.p50_ms / 1000.0)
        if n_bytes and self.throughput_mbps > 0:
            seconds = n_bytes / (self.throughput_mbps * 1024 * 1024 / 8.0)
            await asyncio.sleep(seconds)
        self.counters.network_seconds += (self.p50_ms / 1000.0) + (
            n_bytes / (self.throughput_mbps * 1024 * 1024 / 8.0)
            if self.throughput_mbps > 0 else 0.0
        )

    async def get(self, key, prototype, byte_range=None):
        result = await self._store.get(key, prototype, byte_range)
        n_bytes = 0
        if result is not None:
            try:
                n_bytes = len(result.to_bytes())
            except Exception:
                n_bytes = result.nbytes if hasattr(result, "nbytes") else 0
            self.counters.fetches += 1
            self.counters.bytes_fetched += n_bytes
        await self._delay(n_bytes)
        return result

    async def get_partial_values(self, prototype, key_ranges):
        results = await self._store.get_partial_values(prototype, key_ranges)
        n_bytes = 0
        for r in results:
            if r is not None:
                try:
                    nb = len(r.to_bytes())
                except Exception:
                    nb = r.nbytes if hasattr(r, "nbytes") else 0
                n_bytes += nb
                self.counters.fetches += 1
                self.counters.bytes_fetched += nb
        await self._delay(n_bytes)
        return results


# ---------- Convert .npz -> Zarr -----------------------------------------

def ensure_zarr(n: int, weeks: int, force: bool = False) -> Path:
    src = npz_path(n, weeks)
    if not src.exists():
        raise FileNotFoundError(f"missing artefact: {src}. build it first.")
    dst = zarr_path(n, weeks)
    if dst.exists() and not force:
        return dst
    if dst.exists():
        shutil.rmtree(dst)
    print(f"converting {src.name} -> {dst.name} ...", flush=True)
    npz = np.load(src, allow_pickle=True)
    cov_full = npz["cov_full"]
    has_eig = "eig_V" in npz.files and "eig_D" in npz.files
    eig_V = npz["eig_V"] if has_eig else None
    eig_D = npz["eig_D"] if has_eig else None

    store = LocalStore(str(dst))
    root = zarr.create_group(store=store, overwrite=True)
    # Chunk by week — one chunk per Σ_t. For our access pattern (each
    # query reads a contiguous week range, walks the full N×N) this is
    # the only sensible choice.
    arr = root.create_array(
        "cov_full",
        shape=cov_full.shape,
        chunks=(1, cov_full.shape[1], cov_full.shape[2]),
        dtype=cov_full.dtype,
        compressors=None,
    )
    arr[:] = cov_full
    if eig_V is not None and eig_D is not None:
        arr_v = root.create_array(
            "eig_V",
            shape=eig_V.shape,
            chunks=(1, eig_V.shape[1], eig_V.shape[2]),
            dtype=eig_V.dtype,
            compressors=None,
        )
        arr_v[:] = eig_V
        arr_d = root.create_array(
            "eig_D",
            shape=eig_D.shape,
            chunks=(1, eig_D.shape[1]),
            dtype=eig_D.dtype,
            compressors=None,
        )
        arr_d[:] = eig_D
    print(f"  wrote {dst.name}  size={_du_mb(dst):.1f} MB", flush=True)
    return dst


def _du_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


# ---------- Compute primitives -------------------------------------------

def quadratic_full_streaming(cov: zarr.Array, x: np.ndarray, weeks_slice: slice) -> np.ndarray:
    """Per-week loop — touches one Σ_t at a time. Caps RAM usage at one
    matrix and forces a synchronous fetch per week."""
    out = []
    for t in range(weeks_slice.start or 0, weeks_slice.stop or cov.shape[0]):
        sigma_t = np.asarray(cov[t])
        out.append(float(x @ sigma_t @ x))
    return np.asarray(out, dtype=np.float64)


def quadratic_full_bulk(cov: zarr.Array, x: np.ndarray, weeks_slice: slice) -> np.ndarray:
    """Single zarr slice that materialises the requested week range as a
    contiguous numpy array, then runs the production batched matmul.
    Lets zarr's internal scheduler fetch chunks in parallel. Holds the
    full window in RAM — for large N this is the same memory cost as
    the local-mmap path; whether that's acceptable depends on the
    backend host's RAM."""
    s = weeks_slice.start or 0
    e = weeks_slice.stop or cov.shape[0]
    sub = np.asarray(cov[s:e])           # (W, N, N) — parallel chunk fetch
    y_all = sub @ x                       # (W, N)
    return (y_all * x).sum(axis=1)        # (W,)


def quadratic_approx_bulk(eig_V: zarr.Array, eig_D: zarr.Array, x: np.ndarray, weeks_slice: slice, k: int) -> np.ndarray:
    """Bulk read of (V, D) for the requested week range — same parallel
    chunk-fetch trick as quadratic_full_bulk. Holds (W, N, k) in RAM
    rather than (W, N, N), so memory is much smaller than the full
    case even at large N."""
    s = weeks_slice.start or 0
    e = weeks_slice.stop or eig_V.shape[0]
    V = np.asarray(eig_V[s:e, :, -k:])    # (W, N, k)
    D = np.asarray(eig_D[s:e, -k:])        # (W, k)
    y = np.einsum("wnk,n->wk", V, x, optimize=True)
    return np.einsum("wk,wk->w", D, y * y, optimize=True)


# ---------- Bench --------------------------------------------------------

@dataclass
class SimRow:
    n: int
    w: int
    scenario: str            # baseline | zarr-local | zarr-sim-<label>
    p50_ms: float
    throughput_mbps: float
    mode: str                # full | approx-k=K
    iter_idx: int
    elapsed_s: float
    fetches: int
    bytes_fetched_mb: float
    network_seconds: float


SCENARIOS = [
    # (label, p50_ms, throughput_mbps)
    ("local",        0.0,    0),       # zarr but no latency = pure on-disk
    ("minio_lh",     0.5,    10000),   # MinIO localhost: ~0.5ms FB, 10 Gbps
    ("aws_inreg",    5.0,    5000),    # S3 in same region: ~5ms FB, 5 Gbps
    ("aws_xreg",    80.0,    500),     # S3 cross-region: ~80ms FB, 500 Mbps
]


def bench_one(
    n: int, weeks: int, k: int, iters: int, warmup: int,
) -> list[SimRow]:
    npz = np.load(npz_path(n, weeks), allow_pickle=True)
    cov_in_ram = npz["cov_full"]
    has_eig = "eig_V" in npz.files
    if not has_eig:
        print("  [skip approx] artefact has no eig arrays", flush=True)
    eig_V_ram = npz["eig_V"] if has_eig else None
    eig_D_ram = npz["eig_D"] if has_eig else None
    rng = np.random.default_rng(17)
    x = rng.normal(0.0, 0.3, n).astype(np.float32)

    rows: list[SimRow] = []

    # --- BASELINE: in-RAM numpy (matches the production backend) -----
    def time_n(fn, iters: int, warmup: int) -> list[float]:
        for _ in range(warmup):
            fn()
        ts: list[float] = []
        for _ in range(iters):
            gc.collect()
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return ts

    def baseline_full() -> np.ndarray:
        y = cov_in_ram @ x
        return (y * x).sum(axis=1)

    def baseline_approx() -> np.ndarray:
        V = eig_V_ram[:, :, -k:]
        D = eig_D_ram[:, -k:]
        y = np.einsum("wnk,n->wk", V, x, optimize=True)
        return np.einsum("wk,wk->w", D, y * y, optimize=True)

    for ts in time_n(baseline_full, iters, warmup):
        rows.append(SimRow(n=n, w=weeks, scenario="baseline-numpy", p50_ms=0, throughput_mbps=0,
                           mode="full", iter_idx=len(rows), elapsed_s=ts,
                           fetches=0, bytes_fetched_mb=0, network_seconds=0))
    if has_eig:
        for ts in time_n(baseline_approx, iters, warmup):
            rows.append(SimRow(n=n, w=weeks, scenario="baseline-numpy", p50_ms=0, throughput_mbps=0,
                               mode=f"approx-k={k}", iter_idx=len(rows), elapsed_s=ts,
                               fetches=0, bytes_fetched_mb=0, network_seconds=0))

    # --- ZARR variants ----------------------------------------------
    zpath = ensure_zarr(n, weeks)
    weeks_slice = slice(0, weeks)

    for label, p50, bw in SCENARIOS:
        store = LatencyStore(LocalStore(str(zpath)), p50_ms=p50, throughput_mbps=bw)
        root = zarr.open_group(store=store, mode="r")
        cov = root["cov_full"]
        eig_V = root["eig_V"] if has_eig else None
        eig_D = root["eig_D"] if has_eig else None
        scenario = f"zarr-{label}"

        # FULL streaming — per-week loop; one fetch per week, one matrix in RAM.
        for it in range(iters):
            gc.collect()
            store.reset_counters()
            t0 = time.perf_counter()
            _ = quadratic_full_streaming(cov, x, weeks_slice)
            elapsed = time.perf_counter() - t0
            rows.append(SimRow(
                n=n, w=weeks, scenario=scenario, p50_ms=p50, throughput_mbps=bw,
                mode="full-stream", iter_idx=it, elapsed_s=elapsed,
                fetches=store.fetches,
                bytes_fetched_mb=store.bytes_fetched / (1024 * 1024),
                network_seconds=store.network_seconds,
            ))

        # FULL bulk — single zarr slice, parallel chunk fetch, batched matmul.
        for it in range(iters):
            gc.collect()
            store.reset_counters()
            t0 = time.perf_counter()
            _ = quadratic_full_bulk(cov, x, weeks_slice)
            elapsed = time.perf_counter() - t0
            rows.append(SimRow(
                n=n, w=weeks, scenario=scenario, p50_ms=p50, throughput_mbps=bw,
                mode="full-bulk", iter_idx=it, elapsed_s=elapsed,
                fetches=store.fetches,
                bytes_fetched_mb=store.bytes_fetched / (1024 * 1024),
                network_seconds=store.network_seconds,
            ))

        # APPROX bulk — single-slice fetch of (V, D) for the week range.
        if has_eig:
            for it in range(iters):
                gc.collect()
                store.reset_counters()
                t0 = time.perf_counter()
                _ = quadratic_approx_bulk(eig_V, eig_D, x, weeks_slice, k=k)
                elapsed = time.perf_counter() - t0
                rows.append(SimRow(
                    n=n, w=weeks, scenario=scenario, p50_ms=p50, throughput_mbps=bw,
                    mode=f"approx-bulk-k={k}", iter_idx=it, elapsed_s=elapsed,
                    fetches=store.fetches,
                    bytes_fetched_mb=store.bytes_fetched / (1024 * 1024),
                    network_seconds=store.network_seconds,
                ))

    return rows


# ---------- Driver -------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--k", type=int, default=30, help="approx mode k")
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.out is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%dT%H%M%S")
        args.out = RESULTS_DIR / f"zarr_sim_{ts}.csv"

    print(f"=== zarr local sim — N={args.n}, W={args.weeks}, k={args.k} ===", flush=True)
    rows = bench_one(args.n, args.weeks, args.k, args.iters, args.warmup)

    # Aggregate per (scenario, mode) and print.
    by_key: dict[tuple[str, str], list[SimRow]] = {}
    for r in rows:
        by_key.setdefault((r.scenario, r.mode), []).append(r)
    print("\nsummary (median over iters):")
    print(
        f"  {'scenario':<22}  {'mode':<14}  {'p50(s)':>8}  {'fetches':>7}  "
        f"{'MB pulled':>9}  {'net(s)':>7}"
    )
    for (sc, mode), rs in sorted(by_key.items()):
        ts_ = sorted(r.elapsed_s for r in rs)
        p50 = ts_[len(ts_) // 2]
        rep = rs[len(rs) // 2]
        print(
            f"  {sc:<22}  {mode:<14}  {p50:>7.3f}s  {rep.fetches:>7}  "
            f"{rep.bytes_fetched_mb:>8.1f}M  {rep.network_seconds:>6.3f}s",
            flush=True,
        )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
