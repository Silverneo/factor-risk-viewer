"""Bench: batch factor risk contribution for many portfolios.

Setup:

  - F   factor covariance, shape (K, K), K = 3000
  - X   factor exposures,  shape (K, P), P sweeps {100, 500, 1000, 5000}
  - i   per-portfolio idiosyncratic variance, shape (P,) — scalar add

Compute path:

  M = F @ X                  # (K, P)  — one GEMM
  C = X * M                  # (K, P)  — Hadamard
  sigma2_F = C.sum(axis=0)   # (P,)
  sigma2_total = sigma2_F + idio_var

Three sweeps, one CSV each:

  1. compute_kernel.csv   — matmul + reduction, batched vs per-portfolio,
                            float32 vs float64
  2. serialization.csv    — npy / Arrow IPC / Parquet, write + read,
                            for both F and X at both dtypes
  3. end_to_end.csv       — read F + X from disk → compute → write C,
                            then derived columns for S3 transfer at three
                            bandwidth tiers

Re-run:

  cd backend
  uv run python -m bench.factor_contrib_bench \
      --out ../experiments/2026-05-07-batch-factor-contribution-perf/results
"""

from __future__ import annotations

import argparse
import csv
import gc
import io
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def median_time(fn, iters: int = 5, warmup: int = 1) -> float:
    """Return median wall-clock time in seconds across `iters` runs."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2]


def make_psd_factor_cov(K: int, dtype, seed: int = 0) -> np.ndarray:
    """Synthetic factor cov: low-rank + diagonal so it's positive definite
    and roughly resembles a real Barra-style cov spectrum."""
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((K, 60)).astype(dtype)
    diag = (0.3 + rng.random(K).astype(dtype)) ** 2
    F = Q @ Q.T + np.diag(diag)
    return F.astype(dtype)


def make_exposures(K: int, P: int, dtype, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((K, P)).astype(dtype)


@dataclass
class ComputeRow:
    K: int
    P: int
    dtype: str
    mode: str           # "batched" or "loop"
    p50_ms: float
    gflops: float       # 2·K²·P / time
    bytes_touched: int  # F + X + M
    gb_per_s: float     # bytes_touched / time / 1e9


def bench_compute(K: int, Ps: list[int], iters: int) -> list[ComputeRow]:
    rows: list[ComputeRow] = []
    for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
        F = make_psd_factor_cov(K, dtype)
        for P in Ps:
            X = make_exposures(K, P, dtype)

            # Batched: single GEMM + Hadamard + sum
            def batched():
                M = F @ X
                C = X * M
                _ = C.sum(axis=0)

            t = median_time(batched, iters=iters)
            flops = 2.0 * K * K * P
            bytes_touched = (F.nbytes + X.nbytes + X.nbytes)  # F read, X read, M written
            rows.append(ComputeRow(
                K=K, P=P, dtype=dtype_name, mode="batched",
                p50_ms=t * 1000,
                gflops=flops / t / 1e9,
                bytes_touched=bytes_touched,
                gb_per_s=bytes_touched / t / 1e9,
            ))

            # Loop: F @ x_p, P times
            def loop():
                out = np.empty(P, dtype=dtype)
                for p in range(P):
                    m = F @ X[:, p]
                    out[p] = X[:, p] @ m

            t = median_time(loop, iters=max(1, iters // 2 if P >= 1000 else iters))
            rows.append(ComputeRow(
                K=K, P=P, dtype=dtype_name, mode="loop",
                p50_ms=t * 1000,
                gflops=flops / t / 1e9,
                bytes_touched=bytes_touched,
                gb_per_s=bytes_touched / t / 1e9,
            ))

            del X
            gc.collect()
        del F
        gc.collect()
    return rows


@dataclass
class SerializationRow:
    array: str          # "F" or "X"
    K: int
    P: int              # for X; 0 for F
    dtype: str
    fmt: str            # "npy", "arrow_ipc", "parquet"
    bytes_on_disk: int
    write_ms: float
    read_ms: float
    read_gb_per_s: float  # bytes_on_disk / read_time / 1e9 (the relevant one for S3)


def _bench_serialize(name: str, arr: np.ndarray, dtype_name: str, K: int, P: int, tmpdir: Path, iters: int) -> list[SerializationRow]:
    rows: list[SerializationRow] = []

    # 1. .npy
    npy_path = tmpdir / f"{name}_{dtype_name}.npy"
    def write_npy():
        np.save(npy_path, arr)
    def read_npy():
        np.load(npy_path)
    write_npy()
    bytes_on_disk = npy_path.stat().st_size
    write_t = median_time(write_npy, iters=iters)
    read_t = median_time(read_npy, iters=iters)
    rows.append(SerializationRow(
        array=name, K=K, P=P, dtype=dtype_name, fmt="npy",
        bytes_on_disk=bytes_on_disk,
        write_ms=write_t * 1000, read_ms=read_t * 1000,
        read_gb_per_s=bytes_on_disk / read_t / 1e9,
    ))

    # 2. Arrow IPC. Treat the matrix as one FixedSizeListArray per row (cheap)
    #    or simpler: a single FixedSizeBinaryArray storing all bytes. We use
    #    the simplest: a Table with one FixedSizeListArray column representing
    #    the rows. But the most idiomatic is to store as a single column of
    #    list<float>; pyarrow lacks a 2D primitive. The closest *fast* path is
    #    to represent the matrix as a contiguous buffer in a FixedSizeBinary
    #    field, which is what zero-copy bytes ↔ np looks like in practice.
    #    We use that path here.
    arrow_path = tmpdir / f"{name}_{dtype_name}.arrow"
    raw_bytes = arr.tobytes()
    schema = pa.schema([("data", pa.binary()), ("shape", pa.list_(pa.int64())), ("dtype", pa.string())])

    def write_arrow():
        tbl = pa.table({
            "data": [raw_bytes],
            "shape": [list(arr.shape)],
            "dtype": [str(arr.dtype)],
        }, schema=schema)
        with pa.OSFile(str(arrow_path), "wb") as f:
            with ipc.new_stream(f, schema) as writer:
                writer.write_table(tbl)

    def read_arrow():
        with pa.memory_map(str(arrow_path), "r") as f:
            reader = ipc.open_stream(f)
            tbl = reader.read_all()
        b = tbl.column("data")[0].as_py()
        shape = tuple(tbl.column("shape")[0].as_py())
        dtype = np.dtype(tbl.column("dtype")[0].as_py())
        np.frombuffer(b, dtype=dtype).reshape(shape)

    write_arrow()
    bytes_on_disk = arrow_path.stat().st_size
    write_t = median_time(write_arrow, iters=iters)
    read_t = median_time(read_arrow, iters=iters)
    rows.append(SerializationRow(
        array=name, K=K, P=P, dtype=dtype_name, fmt="arrow_ipc",
        bytes_on_disk=bytes_on_disk,
        write_ms=write_t * 1000, read_ms=read_t * 1000,
        read_gb_per_s=bytes_on_disk / read_t / 1e9,
    ))

    # 3. Parquet — store column-oriented (each portfolio = column for X, each
    #    row of F = column for F). pyarrow Parquet is the most likely format
    #    if the data is exported from a "real" data engine.
    parquet_path = tmpdir / f"{name}_{dtype_name}.parquet"
    pa_dtype = pa.float32() if dtype_name == "float32" else pa.float64()
    if name == "X":
        # Columns = portfolios
        cols = {f"p{j:04d}": pa.array(arr[:, j], type=pa_dtype) for j in range(arr.shape[1])}
    else:
        # F: columns = factors (square)
        cols = {f"f{j:04d}": pa.array(arr[:, j], type=pa_dtype) for j in range(arr.shape[1])}
    parquet_table = pa.table(cols)

    def write_parquet():
        pq.write_table(parquet_table, parquet_path, compression="snappy")
    def read_parquet():
        t = pq.read_table(parquet_path)
        # Reconstruct ndarray (this is the cost real consumers pay)
        np.column_stack([c.to_numpy(zero_copy_only=False) for c in t.columns])

    write_parquet()
    bytes_on_disk = parquet_path.stat().st_size
    write_t = median_time(write_parquet, iters=max(1, iters // 2))
    read_t = median_time(read_parquet, iters=max(1, iters // 2))
    rows.append(SerializationRow(
        array=name, K=K, P=P, dtype=dtype_name, fmt="parquet",
        bytes_on_disk=bytes_on_disk,
        write_ms=write_t * 1000, read_ms=read_t * 1000,
        read_gb_per_s=bytes_on_disk / read_t / 1e9,
    ))

    return rows


def bench_serialization(K: int, P: int, tmpdir: Path, iters: int) -> list[SerializationRow]:
    rows: list[SerializationRow] = []
    for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
        F = make_psd_factor_cov(K, dtype)
        X = make_exposures(K, P, dtype)
        rows.extend(_bench_serialize("F", F, dtype_name, K, 0, tmpdir, iters))
        rows.extend(_bench_serialize("X", X, dtype_name, K, P, tmpdir, iters))
        del F, X
        gc.collect()
    return rows


@dataclass
class EndToEndRow:
    K: int
    P: int
    dtype: str
    fmt: str
    F_bytes: int
    X_bytes: int
    read_ms: float       # F + X read + decode
    compute_ms: float    # batched matmul path
    total_ms: float      # read + compute
    # Network model — added by post-processing
    s3_inreg_total_ms: float = 0.0
    s3_xreg_total_ms: float = 0.0
    s3_offcloud_total_ms: float = 0.0


def bench_end_to_end(K: int, P: int, tmpdir: Path, iters: int) -> list[EndToEndRow]:
    rows: list[EndToEndRow] = []
    for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
        F = make_psd_factor_cov(K, dtype)
        X = make_exposures(K, P, dtype)

        # Pre-write all formats
        npy_F = tmpdir / f"F_{dtype_name}.npy"
        npy_X = tmpdir / f"X_{dtype_name}.npy"
        np.save(npy_F, F); np.save(npy_X, X)

        def npy_read():
            return np.load(npy_F), np.load(npy_X)

        # Arrow IPC paths use the same writer as above
        def _write_ipc(arr, path):
            schema = pa.schema([("data", pa.binary()), ("shape", pa.list_(pa.int64())), ("dtype", pa.string())])
            tbl = pa.table({"data": [arr.tobytes()], "shape": [list(arr.shape)], "dtype": [str(arr.dtype)]}, schema=schema)
            with pa.OSFile(str(path), "wb") as f:
                with ipc.new_stream(f, schema) as w:
                    w.write_table(tbl)

        def _read_ipc(path):
            with pa.memory_map(str(path), "r") as f:
                tbl = ipc.open_stream(f).read_all()
            b = tbl.column("data")[0].as_py()
            shape = tuple(tbl.column("shape")[0].as_py())
            d = np.dtype(tbl.column("dtype")[0].as_py())
            return np.frombuffer(b, dtype=d).reshape(shape)

        ipc_F = tmpdir / f"F_{dtype_name}.arrow"
        ipc_X = tmpdir / f"X_{dtype_name}.arrow"
        _write_ipc(F, ipc_F); _write_ipc(X, ipc_X)

        def ipc_read():
            return _read_ipc(ipc_F), _read_ipc(ipc_X)

        # Parquet
        pa_dtype = pa.float32() if dtype_name == "float32" else pa.float64()
        pq_F = tmpdir / f"F_{dtype_name}.parquet"
        pq_X = tmpdir / f"X_{dtype_name}.parquet"
        pq.write_table(pa.table({f"f{j:04d}": pa.array(F[:, j], type=pa_dtype) for j in range(K)}), pq_F)
        pq.write_table(pa.table({f"p{j:04d}": pa.array(X[:, j], type=pa_dtype) for j in range(P)}), pq_X)

        def pq_read():
            t1 = pq.read_table(pq_F)
            t2 = pq.read_table(pq_X)
            F2 = np.column_stack([c.to_numpy(zero_copy_only=False) for c in t1.columns])
            X2 = np.column_stack([c.to_numpy(zero_copy_only=False) for c in t2.columns])
            return F2, X2

        # Compute is the same
        def compute_on(F2, X2):
            M = F2 @ X2
            C = X2 * M
            return C.sum(axis=0)

        for fmt, reader, fpath, xpath in [
            ("npy", npy_read, npy_F, npy_X),
            ("arrow_ipc", ipc_read, ipc_F, ipc_X),
            ("parquet", pq_read, pq_F, pq_X),
        ]:
            f_bytes = fpath.stat().st_size
            x_bytes = xpath.stat().st_size

            # Time only the read step
            it = max(1, iters // 2 if fmt == "parquet" else iters)
            read_t = median_time(reader, iters=it)
            # Time only the compute step (using freshly-loaded arrays)
            F2, X2 = reader()
            compute_t = median_time(lambda: compute_on(F2, X2), iters=iters)

            rows.append(EndToEndRow(
                K=K, P=P, dtype=dtype_name, fmt=fmt,
                F_bytes=f_bytes, X_bytes=x_bytes,
                read_ms=read_t * 1000,
                compute_ms=compute_t * 1000,
                total_ms=(read_t + compute_t) * 1000,
            ))

        del F, X
        gc.collect()
    return rows


# -------- Network model (paper, applied to end_to_end rows) ----------------

# Bandwidth tiers (sustained, sequential), and first-byte penalty per request.
# Numbers calibrated to the prior 2026-04-27-on-the-fly-risk Phase 4/5 work
# and AWS published guidance.
NET_TIERS = {
    "s3_inreg":   {"gbps": 5.0,   "fb_ms": 5.0,   "n_requests": 2},
    "s3_xreg":    {"gbps": 0.5,   "fb_ms": 80.0,  "n_requests": 2},
    "s3_offcloud":{"gbps": 0.1,   "fb_ms": 80.0,  "n_requests": 2},
}


def transfer_ms(total_bytes: int, gbps: float, fb_ms: float, n_requests: int) -> float:
    """Estimate transfer wall-clock for `total_bytes` over a `gbps`-bandwidth
    link with `fb_ms` first-byte latency per request."""
    bps = gbps * 1e9 / 8
    seconds = total_bytes / bps
    return seconds * 1000 + fb_ms * n_requests


def annotate_network(rows: list[EndToEndRow]) -> list[EndToEndRow]:
    for r in rows:
        total_bytes = r.F_bytes + r.X_bytes
        for tier_key, tier in NET_TIERS.items():
            xfer = transfer_ms(total_bytes, tier["gbps"], tier["fb_ms"], tier["n_requests"])
            setattr(r, f"{tier_key}_total_ms", xfer + r.compute_ms)
    return rows


def write_csv(rows, path: Path):
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
    ap.add_argument("--K", type=int, default=3000)
    ap.add_argument("--P", type=int, nargs="+", default=[100, 500, 1000, 5000])
    ap.add_argument("--P-end-to-end", type=int, default=500)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tmpdir", type=Path, default=Path("./_factor_contrib_tmp"))
    ap.add_argument("--skip", type=str, nargs="*", default=[],
                    help="Phases to skip: compute, serial, e2e")
    args = ap.parse_args()

    args.tmpdir.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)

    if "compute" not in args.skip:
        print(f"[1/3] compute kernel sweep K={args.K} P={args.P}")
        rows = bench_compute(args.K, args.P, args.iters)
        write_csv(rows, args.out / "compute_kernel.csv")
        for r in rows:
            print(f"   K={r.K} P={r.P:>4} {r.dtype} {r.mode:<8} "
                  f"{r.p50_ms:8.2f} ms  {r.gflops:7.1f} GFLOPS  "
                  f"{r.gb_per_s:5.1f} GB/s touch")

    if "serial" not in args.skip:
        print(f"[2/3] serialization sweep K={args.K} P={args.P_end_to_end}")
        rows = bench_serialization(args.K, args.P_end_to_end, args.tmpdir, args.iters)
        write_csv(rows, args.out / "serialization.csv")
        for r in rows:
            print(f"   {r.array} {r.dtype} {r.fmt:<10} "
                  f"{r.bytes_on_disk/1e6:7.2f} MB  "
                  f"write {r.write_ms:7.1f} ms  read {r.read_ms:7.1f} ms  "
                  f"({r.read_gb_per_s:.2f} GB/s)")

    if "e2e" not in args.skip:
        print(f"[3/3] end-to-end K={args.K} P={args.P_end_to_end}")
        rows = bench_end_to_end(args.K, args.P_end_to_end, args.tmpdir, args.iters)
        rows = annotate_network(rows)
        write_csv(rows, args.out / "end_to_end.csv")
        for r in rows:
            print(f"   {r.dtype} {r.fmt:<10} "
                  f"read {r.read_ms:6.1f}  compute {r.compute_ms:6.1f}  "
                  f"total {r.total_ms:6.1f}  "
                  f"in-reg+compute {r.s3_inreg_total_ms:6.1f}  "
                  f"x-reg+compute {r.s3_xreg_total_ms:6.1f}  "
                  f"off-cloud+compute {r.s3_offcloud_total_ms:6.1f}")


if __name__ == "__main__":
    main()
