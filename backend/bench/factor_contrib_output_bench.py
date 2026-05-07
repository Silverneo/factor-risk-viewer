"""Output-side cost: per-factor contribution matrix vs sum-only.

Two questions:
  1. Does keeping C = X*M (per-factor contribution) cost any extra
     compute over (X*M).sum(0)?
  2. What does serializing + shipping C cost in each format?

C has the same shape and bytes as X (K x P), so we can re-use
serialization measurements from factor_contrib_bench.py.

Run from backend/:
  uv run python -m bench.factor_contrib_output_bench
"""
from __future__ import annotations
import gc
import io
import time
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq


def t(fn, n=5, warmup=1):
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        s.append(time.perf_counter() - t0)
    s.sort()
    return s[len(s) // 2]


K, P = 3000, 500
rng = np.random.default_rng(0)


def make_F(dtype):
    Q = rng.standard_normal((K, 60)).astype(dtype)
    diag = (0.3 + rng.random(K).astype(dtype)) ** 2
    return (Q @ Q.T + np.diag(diag)).astype(dtype)


print("=" * 80)
print("Phase 1 — compute path: sum-only vs keep-C vs einsum-fused-sum")
print("=" * 80)

for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
    F = make_F(dtype)
    X = rng.standard_normal((K, P)).astype(dtype)

    def sum_only():
        M = F @ X
        return (X * M).sum(0)

    def keep_C():
        M = F @ X
        C = X * M
        return C.sum(0), C

    def einsum_fused():
        M = F @ X
        return np.einsum("ij,ij->j", X, M)

    sum_ms = t(sum_only, n=5) * 1000
    keep_ms = t(keep_C, n=5) * 1000
    einsum_ms = t(einsum_fused, n=5) * 1000
    print(f"  {dtype_name}:  sum-only {sum_ms:6.2f} ms   "
          f"keep-C {keep_ms:6.2f} ms   "
          f"einsum-fused {einsum_ms:6.2f} ms")
    del F, X
    gc.collect()


print()
print("=" * 80)
print("Phase 2 — output serialization: write C in each format (in-RAM bytes)")
print("=" * 80)

results = []
for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
    F = make_F(dtype)
    X = rng.standard_normal((K, P)).astype(dtype)
    M = F @ X
    C = X * M  # this IS the per-factor contribution matrix

    pa_dtype = pa.float32() if dtype_name == "float32" else pa.float64()

    # 1. .npy
    def npy_to_buf():
        b = io.BytesIO()
        np.save(b, C)
        return b.getvalue()
    npy_size = len(npy_to_buf())
    npy_ms = t(npy_to_buf, n=5) * 1000

    # 2. Arrow IPC (raw blob path)
    schema = pa.schema([("data", pa.binary()), ("shape", pa.list_(pa.int64())), ("dtype", pa.string())])
    def ipc_to_buf():
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, schema) as w:
            w.write_table(pa.table({"data": [C.tobytes()], "shape": [list(C.shape)], "dtype": [str(C.dtype)]}, schema=schema))
        return sink.getvalue()
    ipc_size = ipc_to_buf().size
    ipc_ms = t(ipc_to_buf, n=5) * 1000

    # 3. Parquet (column-per-portfolio)
    def parquet_to_buf():
        buf = io.BytesIO()
        pq.write_table(pa.table({f"p{j:04d}": pa.array(C[:, j], type=pa_dtype) for j in range(P)}), buf)
        return buf.getvalue()
    pq_size = len(parquet_to_buf())
    pq_ms = t(parquet_to_buf, n=3) * 1000

    print(f"  {dtype_name}:  npy {npy_ms:6.1f} ms ({npy_size/1e6:.1f} MB)   "
          f"arrow_ipc {ipc_ms:6.1f} ms ({ipc_size/1e6:.1f} MB)   "
          f"parquet {pq_ms:6.1f} ms ({pq_size/1e6:.1f} MB)")
    results.append((dtype_name, npy_ms, npy_size, ipc_ms, ipc_size, pq_ms, pq_size))
    del F, X, M, C
    gc.collect()


print()
print("=" * 80)
print("Phase 3 — modeled S3 upload of C (write back to bucket, in-region 5 Gbps)")
print("=" * 80)

# In-region S3 upload at 5 Gbps, single PUT request, ~5 ms first-byte
def s3_inreg_upload_ms(bytes_):
    return bytes_ / (5e9 / 8) * 1000 + 5

for dtype_name, npy_ms, npy_size, ipc_ms, ipc_size, pq_ms, pq_size in results:
    print(f"  {dtype_name}:")
    for fmt, ser_ms, n_bytes in [("npy", npy_ms, npy_size),
                                  ("arrow_ipc", ipc_ms, ipc_size),
                                  ("parquet", pq_ms, pq_size)]:
        upload_ms = s3_inreg_upload_ms(n_bytes)
        total = ser_ms + upload_ms
        print(f"    {fmt:<10} serialize {ser_ms:6.1f} ms + S3 upload {upload_ms:6.1f} ms = {total:6.1f} ms total")
