"""Real-SDK companion to factor_contrib_bench.py — exercise boto3 GETs
against a local MinIO server to validate the modeled S3 numbers.

Runs the same factor-contribution workload (K=3000, P=500) end-to-end
through plain boto3 (NO s3fs, NO smart caching layers) so the timings
correspond closely to what a Lambda / EC2 process would see talking to
real AWS S3.

Workflow per measured row:
  1. boto3 get_object("F.<format>")           -> raw bytes  (timed)
  2. boto3 get_object("X.<format>")           -> raw bytes  (timed, parallel)
  3. format-specific decode bytes -> ndarray  (timed)
  4. matmul + Hadamard + sum                  (timed)
Optionally:
  5. write contribution C back as <format> bytes
  6. boto3 put_object("C.<format>")
(steps 5-6 are off by default; toggle with --include-output)

Iteration model:
  - iters=3 by default. Iter 0 is "cold" (clean MinIO connection state,
    no OS page cache). Subsequent iters are "warm".
  - We do NOT clear OS caches between iters — on Windows that's not
    cheap, and warm timings are interesting in their own right.

------------------------------------------------------------------------
SETUP

If you don't already have MinIO running:

    # Download once:
    mkdir -p backend/.tools
    curl -L -o backend/.tools/minio.exe \
         https://dl.min.io/server/minio/release/windows-amd64/minio.exe

    # Start in a separate terminal:
    cd backend
    ./.tools/minio.exe server ./.minio-data --console-address :9001

    # Default API: http://localhost:9000   (creds: minioadmin/minioadmin)

Run:
    cd backend
    uv run python -m bench.factor_contrib_minio_bench

Override defaults via env vars: MINIO_ENDPOINT, MINIO_ACCESS_KEY,
MINIO_SECRET_KEY, MINIO_BUCKET.
------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import csv
import gc
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path

import boto3
import botocore
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = HERE.parents[0] / "experiments" / "2026-05-07-batch-factor-contribution-perf"
RESULTS_DIR = EXPERIMENT_DIR / "results"

DEFAULT_ENDPOINT = "http://localhost:9000"
DEFAULT_ACCESS_KEY = "minioadmin"
DEFAULT_SECRET_KEY = "minioadmin"
DEFAULT_BUCKET = "factor-contrib"

K, P = 3000, 500


def _make_s3_client(endpoint: str, access_key: str, secret_key: str):
    """Plain boto3 S3 client pointed at MinIO."""
    cfg = botocore.config.Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
        retries={"max_attempts": 1, "mode": "standard"},
        connect_timeout=5,
        read_timeout=60,
        max_pool_connections=20,
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        config=cfg,
    )


def _ensure_bucket(s3, bucket: str):
    try:
        s3.head_bucket(Bucket=bucket)
    except botocore.exceptions.ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in {"404", "NoSuchBucket", "NotFound"}:
            print(f"  creating bucket: {bucket}", flush=True)
            s3.create_bucket(Bucket=bucket)
        else:
            raise


# ---------------- data + serialization helpers ----------------

def make_F(dtype) -> np.ndarray:
    rng = np.random.default_rng(0)
    Q = rng.standard_normal((K, 60)).astype(dtype)
    diag = (0.3 + rng.random(K).astype(dtype)) ** 2
    return (Q @ Q.T + np.diag(diag)).astype(dtype)


def make_X(dtype) -> np.ndarray:
    rng = np.random.default_rng(1)
    return rng.standard_normal((K, P)).astype(dtype)


def encode_npy(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def decode_npy(b: bytes) -> np.ndarray:
    return np.load(io.BytesIO(b))


_IPC_SCHEMA = pa.schema([("data", pa.binary()), ("shape", pa.list_(pa.int64())), ("dtype", pa.string())])


def encode_arrow_ipc(arr: np.ndarray) -> bytes:
    sink = pa.BufferOutputStream()
    with ipc.new_stream(sink, _IPC_SCHEMA) as w:
        w.write_table(pa.table({"data": [arr.tobytes()], "shape": [list(arr.shape)], "dtype": [str(arr.dtype)]}, schema=_IPC_SCHEMA))
    return bytes(sink.getvalue())


def decode_arrow_ipc(b: bytes) -> np.ndarray:
    tbl = ipc.open_stream(pa.BufferReader(b)).read_all()
    raw = tbl.column("data")[0].as_py()
    shape = tuple(tbl.column("shape")[0].as_py())
    dt = np.dtype(tbl.column("dtype")[0].as_py())
    return np.frombuffer(raw, dtype=dt).reshape(shape)


def encode_parquet(arr: np.ndarray, dtype_name: str, name_prefix: str) -> bytes:
    pa_dtype = pa.float32() if dtype_name == "float32" else pa.float64()
    cols = {f"{name_prefix}{j:04d}": pa.array(arr[:, j], type=pa_dtype) for j in range(arr.shape[1])}
    sink = pa.BufferOutputStream()
    pq.write_table(pa.table(cols), sink, compression="snappy")
    return bytes(sink.getvalue())


def decode_parquet(b: bytes) -> np.ndarray:
    tbl = pq.read_table(pa.BufferReader(b))
    return np.column_stack([c.to_numpy(zero_copy_only=False) for c in tbl.columns])


# ---------------- bench primitives ----------------

@dataclass
class MinioRow:
    dtype: str
    fmt: str
    iter_idx: int
    iter_kind: str          # "cold" | "warm"
    F_bytes: int
    X_bytes: int
    get_F_ms: float         # parallel GET F
    get_X_ms: float         # parallel GET X
    get_total_ms: float     # max(get_F, get_X) — parallel wall-clock
    decode_ms: float
    compute_ms: float
    total_ms: float


def upload_artefacts(s3, bucket: str, dtype_name: str, dtype, fmt: str) -> tuple[str, str, int, int]:
    """Encode F + X in `fmt` + put_object to bucket. Returns (key_F, key_X, bytes_F, bytes_X)."""
    F = make_F(dtype)
    X = make_X(dtype)
    if fmt == "npy":
        bF, bX = encode_npy(F), encode_npy(X)
    elif fmt == "arrow_ipc":
        bF, bX = encode_arrow_ipc(F), encode_arrow_ipc(X)
    elif fmt == "parquet":
        bF, bX = encode_parquet(F, dtype_name, "f"), encode_parquet(X, dtype_name, "p")
    else:
        raise ValueError(f"unknown fmt {fmt}")
    keyF = f"F_{dtype_name}.{fmt}"
    keyX = f"X_{dtype_name}.{fmt}"
    s3.put_object(Bucket=bucket, Key=keyF, Body=bF)
    s3.put_object(Bucket=bucket, Key=keyX, Body=bX)
    del F, X
    gc.collect()
    return keyF, keyX, len(bF), len(bX)


def fetch_object(s3, bucket: str, key: str) -> tuple[bytes, float]:
    t0 = time.perf_counter()
    r = s3.get_object(Bucket=bucket, Key=key)
    body = r["Body"].read()
    return body, (time.perf_counter() - t0) * 1000


def compute_contrib(F: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (sigma2_F, C). C is the per-factor contribution matrix."""
    M = F @ X
    C = X * M
    return C.sum(axis=0), C


def select_decoder(fmt: str):
    return {"npy": decode_npy, "arrow_ipc": decode_arrow_ipc, "parquet": decode_parquet}[fmt]


def bench_one(
    s3, bucket: str, dtype_name: str, dtype, fmt: str, keyF: str, keyX: str,
    bF: int, bX: int, iters: int, parallel: bool,
) -> list[MinioRow]:
    decoder = select_decoder(fmt)
    rows: list[MinioRow] = []
    for it in range(iters):
        # 1+2: GET F and GET X (parallel by default — mimics real-world client)
        if parallel:
            with ThreadPoolExecutor(max_workers=2) as ex:
                future_F = ex.submit(fetch_object, s3, bucket, keyF)
                future_X = ex.submit(fetch_object, s3, bucket, keyX)
                t0 = time.perf_counter()
                bytes_F, t_F = future_F.result()
                bytes_X, t_X = future_X.result()
                t_get_total = (time.perf_counter() - t0) * 1000
        else:
            bytes_F, t_F = fetch_object(s3, bucket, keyF)
            bytes_X, t_X = fetch_object(s3, bucket, keyX)
            t_get_total = t_F + t_X

        # 3: decode
        t0 = time.perf_counter()
        F = decoder(bytes_F)
        X = decoder(bytes_X)
        t_decode = (time.perf_counter() - t0) * 1000

        # 4: compute
        t0 = time.perf_counter()
        sigma2_F, _C = compute_contrib(F, X)
        t_compute = (time.perf_counter() - t0) * 1000

        rows.append(MinioRow(
            dtype=dtype_name, fmt=fmt, iter_idx=it,
            iter_kind="cold" if it == 0 else "warm",
            F_bytes=bF, X_bytes=bX,
            get_F_ms=t_F, get_X_ms=t_X, get_total_ms=t_get_total,
            decode_ms=t_decode, compute_ms=t_compute,
            total_ms=t_get_total + t_decode + t_compute,
        ))
        # release big arrays before next iter so OS page cache state is honest
        del F, X, sigma2_F, _C
        gc.collect()
    return rows


def write_csv(rows: list[MinioRow], path: Path):
    if not rows:
        return
    fields = list(asdict(rows[0]).keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("MINIO_ENDPOINT", DEFAULT_ENDPOINT))
    ap.add_argument("--access-key", default=os.environ.get("MINIO_ACCESS_KEY", DEFAULT_ACCESS_KEY))
    ap.add_argument("--secret-key", default=os.environ.get("MINIO_SECRET_KEY", DEFAULT_SECRET_KEY))
    ap.add_argument("--bucket", default=os.environ.get("MINIO_BUCKET", DEFAULT_BUCKET))
    ap.add_argument("--dtypes", nargs="+", default=["float32", "float64"], choices=["float32", "float64"])
    ap.add_argument("--formats", nargs="+", default=["npy", "arrow_ipc", "parquet"],
                    choices=["npy", "arrow_ipc", "parquet"])
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--no-parallel", action="store_true",
                    help="Sequential GETs instead of parallel")
    ap.add_argument("--force-upload", action="store_true",
                    help="Re-upload artefacts even if they exist")
    ap.add_argument("--out", type=Path, default=RESULTS_DIR)
    args = ap.parse_args()

    print(f"endpoint    = {args.endpoint}")
    print(f"bucket      = {args.bucket}")
    print(f"dtypes      = {args.dtypes}")
    print(f"formats     = {args.formats}")
    print(f"iters       = {args.iters}")
    print(f"parallel    = {not args.no_parallel}")
    print()

    # Connect
    s3 = _make_s3_client(args.endpoint, args.access_key, args.secret_key)
    try:
        s3.list_buckets()
    except Exception as e:
        print(f"[error] could not reach MinIO at {args.endpoint}: {e}", file=sys.stderr)
        print("Hint: ensure 'minio.exe server ./.minio-data' is running.", file=sys.stderr)
        sys.exit(2)

    _ensure_bucket(s3, args.bucket)

    # Upload all combinations once (or skip if already present)
    print("=" * 80)
    print("Phase A — upload artefacts")
    print("=" * 80)
    catalog: dict[tuple[str, str], tuple[str, str, int, int]] = {}
    for dtype_name in args.dtypes:
        dtype = np.float32 if dtype_name == "float32" else np.float64
        for fmt in args.formats:
            keyF = f"F_{dtype_name}.{fmt}"
            keyX = f"X_{dtype_name}.{fmt}"
            need_upload = args.force_upload
            if not need_upload:
                try:
                    rF = s3.head_object(Bucket=args.bucket, Key=keyF)
                    rX = s3.head_object(Bucket=args.bucket, Key=keyX)
                    bF, bX = rF["ContentLength"], rX["ContentLength"]
                    print(f"  reuse  {dtype_name:<8} {fmt:<10}  F={bF/1e6:.1f} MB  X={bX/1e6:.1f} MB")
                    catalog[(dtype_name, fmt)] = (keyF, keyX, bF, bX)
                    continue
                except botocore.exceptions.ClientError:
                    need_upload = True
            print(f"  upload {dtype_name:<8} {fmt:<10}", flush=True)
            t0 = time.perf_counter()
            kF, kX, bF, bX = upload_artefacts(s3, args.bucket, dtype_name, dtype, fmt)
            print(f"     -> F={bF/1e6:.1f} MB  X={bX/1e6:.1f} MB   ({time.perf_counter() - t0:.1f}s)")
            catalog[(dtype_name, fmt)] = (kF, kX, bF, bX)

    # Bench
    print()
    print("=" * 80)
    print("Phase B — bench reads via boto3")
    print("=" * 80)
    print(f"{'dtype':<8} {'fmt':<10} {'iter':<5} {'getF':>7} {'getX':>7} {'getTot':>7} "
          f"{'decode':>7} {'compute':>8} {'total':>8}  bytes(MB)")
    all_rows: list[MinioRow] = []
    for dtype_name in args.dtypes:
        dtype = np.float32 if dtype_name == "float32" else np.float64
        for fmt in args.formats:
            keyF, keyX, bF, bX = catalog[(dtype_name, fmt)]
            rows = bench_one(s3, args.bucket, dtype_name, dtype, fmt,
                             keyF, keyX, bF, bX, iters=args.iters,
                             parallel=not args.no_parallel)
            for r in rows:
                print(f"{r.dtype:<8} {r.fmt:<10} {r.iter_kind:<5} "
                      f"{r.get_F_ms:7.1f} {r.get_X_ms:7.1f} {r.get_total_ms:7.1f} "
                      f"{r.decode_ms:7.1f} {r.compute_ms:8.1f} {r.total_ms:8.1f}  "
                      f"{(r.F_bytes + r.X_bytes)/1e6:.1f}")
            all_rows.extend(rows)

    out_path = args.out / "minio_boto3.csv"
    write_csv(all_rows, out_path)
    print()
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()
