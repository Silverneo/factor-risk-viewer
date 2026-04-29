"""Real-SDK companion to bench/zarr_local_sim.py — exercises the same
full-stream / full-bulk / approx-bulk patterns against a MinIO server
on localhost via s3fs + boto3 + zarr's FsspecStore. Validates that the
simulated numbers from zarr_local_sim track real-SDK numbers.

------------------------------------------------------------------------
SETUP (one-time, ~5 minutes):

1. Download the MinIO server binary. Pick one:

   # Git Bash / curl:
   mkdir -p backend/.tools
   curl -L -o backend/.tools/minio.exe \\
        https://dl.min.io/server/minio/release/windows-amd64/minio.exe

   # Or with PowerShell:
   Invoke-WebRequest -Uri https://dl.min.io/server/minio/release/windows-amd64/minio.exe `
                     -OutFile backend\\.tools\\minio.exe

   On macOS / Linux replace `windows-amd64/minio.exe` with the right
   build for your platform.

2. Start the MinIO server in a separate terminal:

   cd backend
   ./.tools/minio.exe server ./.minio-data --console-address :9001

   Default credentials: minioadmin / minioadmin
   API endpoint:        http://localhost:9000
   Web console:         http://localhost:9001  (optional)

3. Run this script — it will create the bucket if missing, upload the
   local Zarr artefact on first run, then bench:

   cd backend
   uv run python -m bench.zarr_minio_sim --n 1000 --iters 2

4. (optional) Stop MinIO with Ctrl+C in the server terminal. Data
   persists under backend/.minio-data/ — gitignored.

To override defaults:
  MINIO_ENDPOINT=http://localhost:9000     # API URL
  MINIO_ACCESS_KEY=minioadmin
  MINIO_SECRET_KEY=minioadmin
  MINIO_BUCKET=weekly-cov                  # bucket name
------------------------------------------------------------------------
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
import s3fs
import zarr
from zarr.storage import FsspecStore, LocalStore

# Reuse the local-sim's converter + compute primitives so the two
# benches stay numerically aligned.
from bench.zarr_local_sim import (
    ensure_zarr,
    npz_path,
    quadratic_full_streaming,
    quadratic_full_bulk,
    quadratic_approx_bulk,
)

HERE = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = HERE.parents[0] / "experiments" / "2026-04-27-on-the-fly-risk"
RESULTS_DIR = EXPERIMENT_DIR / "results"

DEFAULT_ENDPOINT = "http://localhost:9000"
DEFAULT_ACCESS_KEY = "minioadmin"
DEFAULT_SECRET_KEY = "minioadmin"
DEFAULT_BUCKET = "weekly-cov"


@dataclass
class MinioRow:
    n: int
    w: int
    scenario: str          # "minio-real"
    mode: str              # full-stream | full-bulk | approx-bulk-k=K
    iter_idx: int
    elapsed_s: float


def _make_fs(endpoint: str, access_key: str, secret_key: str) -> s3fs.S3FileSystem:
    """s3fs handle pointed at MinIO. We disable SSL/cert checking
    because the local server runs on plain HTTP."""
    return s3fs.S3FileSystem(
        key=access_key,
        secret=secret_key,
        client_kwargs={"endpoint_url": endpoint},
        use_ssl=endpoint.startswith("https"),
    )


def _check_minio(fs: s3fs.S3FileSystem) -> bool:
    """Probe whether MinIO is reachable. Returns False on connection
    error so we can print an actionable message rather than dumping a
    raw traceback."""
    try:
        fs.ls("/")  # list buckets
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[error] could not reach MinIO: {e}", file=sys.stderr)
        return False


def _ensure_bucket(fs: s3fs.S3FileSystem, bucket: str) -> None:
    if not fs.exists(bucket):
        print(f"  creating bucket: {bucket}", flush=True)
        fs.mkdir(bucket)


def _ensure_artefact_in_minio(
    fs: s3fs.S3FileSystem,
    bucket: str,
    n: int,
    weeks: int,
    force: bool = False,
) -> str:
    """Upload local weekly_cov_*.zarr to MinIO if not already there.
    Returns the s3 prefix of the artefact."""
    local = HERE / f"weekly_cov_N{n}_W{weeks}.zarr"
    if not local.exists():
        # Build it via the local-sim's converter (which reads npz).
        ensure_zarr(n, weeks)
    s3_prefix = f"{bucket}/weekly_cov_N{n}_W{weeks}.zarr"
    if not force and fs.exists(f"{s3_prefix}/zarr.json"):
        # Quick existence check — full integrity check would need a
        # listing, but the marker file is enough for our purposes.
        print(f"  artefact already in MinIO: s3://{s3_prefix}", flush=True)
        return s3_prefix
    print(f"  uploading {local.name} to s3://{s3_prefix} ...", flush=True)
    t0 = time.perf_counter()
    fs.put(str(local), s3_prefix, recursive=True)
    secs = time.perf_counter() - t0
    print(f"  upload complete in {secs:.1f}s", flush=True)
    return s3_prefix


def _open_remote_group(fs: s3fs.S3FileSystem, s3_prefix: str) -> zarr.Group:
    """Open the remote Zarr group via FsspecStore. The store mediates
    every chunk fetch through s3fs/aiobotocore — same code path as a
    real AWS S3 deployment."""
    store = FsspecStore(fs, path=s3_prefix)
    return zarr.open_group(store=store, mode="r")


def bench_minio(
    n: int,
    weeks: int,
    k: int,
    iters: int,
    warmup: int,
    fs: s3fs.S3FileSystem,
    s3_prefix: str,
) -> list[MinioRow]:
    root = _open_remote_group(fs, s3_prefix)
    cov = root["cov_full"]
    has_eig = "eig_V" in [str(p).split("/")[-1] for p in root]
    eig_V = root["eig_V"] if has_eig else None
    eig_D = root["eig_D"] if has_eig else None

    rng = np.random.default_rng(17)
    x = rng.normal(0.0, 0.3, n).astype(np.float32)
    weeks_slice = slice(0, weeks)

    rows: list[MinioRow] = []

    def time_it(fn, mode: str) -> None:
        for _ in range(warmup):
            _ = fn()
        for it in range(iters):
            gc.collect()
            t0 = time.perf_counter()
            _ = fn()
            elapsed = time.perf_counter() - t0
            rows.append(MinioRow(
                n=n, w=weeks, scenario="minio-real", mode=mode,
                iter_idx=it, elapsed_s=elapsed,
            ))
            print(f"  {mode:<22}  iter {it}  {elapsed * 1000:>9.1f} ms", flush=True)

    # NOTE: warmup matters more here than in the local sim, because
    # the first request through aiobotocore does extra TLS / connection
    # setup. We still report all iters; the median is what to compare.
    print("== full-stream ==", flush=True)
    time_it(lambda: quadratic_full_streaming(cov, x, weeks_slice), "full-stream")
    print("== full-bulk ==", flush=True)
    time_it(lambda: quadratic_full_bulk(cov, x, weeks_slice), "full-bulk")
    if has_eig:
        print(f"== approx-bulk-k={k} ==", flush=True)
        time_it(
            lambda: quadratic_approx_bulk(eig_V, eig_D, x, weeks_slice, k=k),
            f"approx-bulk-k={k}",
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("------------------------------------------------------------------------")[0].strip())
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--weeks", type=int, default=104)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--bucket", default=os.environ.get("MINIO_BUCKET", DEFAULT_BUCKET))
    ap.add_argument("--endpoint", default=os.environ.get("MINIO_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--access-key", default=os.environ.get("MINIO_ACCESS_KEY", DEFAULT_ACCESS_KEY))
    ap.add_argument("--secret-key", default=os.environ.get("MINIO_SECRET_KEY", DEFAULT_SECRET_KEY))
    ap.add_argument("--upload", action="store_true", help="Re-upload the artefact even if present.")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    fs = _make_fs(args.endpoint, args.access_key, args.secret_key)
    if not _check_minio(fs):
        print(
            "\nMinIO doesn't seem to be running.\n"
            f"  - is the server up at {args.endpoint}?\n"
            "  - see this script's docstring for one-time setup steps.\n"
            "  - or override --endpoint / MINIO_ENDPOINT env var.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"=== zarr-minio sim — N={args.n}, W={args.weeks}, k={args.k} ===", flush=True)
    print(f"  endpoint: {args.endpoint}", flush=True)
    _ensure_bucket(fs, args.bucket)
    s3_prefix = _ensure_artefact_in_minio(fs, args.bucket, args.n, args.weeks, force=args.upload)

    rows = bench_minio(args.n, args.weeks, args.k, args.iters, args.warmup, fs, s3_prefix)

    # Per-mode median summary.
    by_mode: dict[str, list[MinioRow]] = {}
    for r in rows:
        by_mode.setdefault(r.mode, []).append(r)
    print("\nsummary (median over iters):")
    print(f"  {'mode':<22}  {'p50_ms':>10}  {'min_ms':>10}")
    for mode, rs in sorted(by_mode.items()):
        ts = sorted(r.elapsed_s for r in rs)
        p50 = ts[len(ts) // 2]
        print(f"  {mode:<22}  {p50 * 1000:>9.1f}ms  {ts[0] * 1000:>9.1f}ms", flush=True)

    if args.out is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts_label = time.strftime("%Y%m%dT%H%M%S")
        args.out = RESULTS_DIR / f"zarr_minio_{ts_label}.csv"
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
