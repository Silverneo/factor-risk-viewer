"""Pure decode cost (no disk, no compute) for npy / Arrow IPC / Parquet.

Lets us decompose `local_read = disk_io + decode` into its parts so the
S3 model can be:

  s3_total = network_transfer + decode + compute

without conflating the disk-bandwidth term in `local_read` with the CPU
decode term.

Run from backend/:
  uv run python -m bench.factor_contrib_decode_only

Supplements factor_contrib_bench.py — informs the correction in
2026-05-07-batch-factor-contribution-perf/report.md (the original
end_to_end S3 model added xfer+compute and missed parquet's decode cost).
"""
from __future__ import annotations
import io
import time
import gc
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

for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
    Q = rng.standard_normal((K, 60)).astype(dtype)
    diag = (0.3 + rng.random(K).astype(dtype)) ** 2
    F = (Q @ Q.T + np.diag(diag)).astype(dtype)
    X = rng.standard_normal((K, P)).astype(dtype)

    # ---- Parquet: write to in-memory bytes, then time pure decode
    pa_dtype = pa.float32() if dtype_name == "float32" else pa.float64()
    pq_F_buf = io.BytesIO()
    pq_X_buf = io.BytesIO()
    pq.write_table(pa.table({f"f{j:04d}": pa.array(F[:, j], type=pa_dtype) for j in range(K)}), pq_F_buf)
    pq.write_table(pa.table({f"p{j:04d}": pa.array(X[:, j], type=pa_dtype) for j in range(P)}), pq_X_buf)
    pq_F_bytes = pq_F_buf.getvalue()
    pq_X_bytes = pq_X_buf.getvalue()

    def parquet_decode():
        tF = pq.read_table(pa.BufferReader(pq_F_bytes))
        tX = pq.read_table(pa.BufferReader(pq_X_bytes))
        F2 = np.column_stack([c.to_numpy(zero_copy_only=False) for c in tF.columns])
        X2 = np.column_stack([c.to_numpy(zero_copy_only=False) for c in tX.columns])
        return F2, X2

    pq_decode_ms = t(parquet_decode, n=5) * 1000

    # ---- npy: write to bytes, then time pure decode
    npy_F_buf = io.BytesIO(); np.save(npy_F_buf, F); npy_F_bytes = npy_F_buf.getvalue()
    npy_X_buf = io.BytesIO(); np.save(npy_X_buf, X); npy_X_bytes = npy_X_buf.getvalue()

    def npy_decode():
        F2 = np.load(io.BytesIO(npy_F_bytes))
        X2 = np.load(io.BytesIO(npy_X_bytes))
        return F2, X2

    npy_decode_ms = t(npy_decode, n=5) * 1000

    # ---- Arrow IPC for completeness
    schema = pa.schema([("data", pa.binary()), ("shape", pa.list_(pa.int64())), ("dtype", pa.string())])

    def _ipc_bytes(arr):
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, schema) as w:
            w.write_table(pa.table({"data": [arr.tobytes()], "shape": [list(arr.shape)], "dtype": [str(arr.dtype)]}, schema=schema))
        return sink.getvalue()

    ipc_F_buf = _ipc_bytes(F)
    ipc_X_buf = _ipc_bytes(X)

    def ipc_decode():
        tF = ipc.open_stream(pa.BufferReader(ipc_F_buf)).read_all()
        tX = ipc.open_stream(pa.BufferReader(ipc_X_buf)).read_all()
        bF = tF.column("data")[0].as_py()
        bX = tX.column("data")[0].as_py()
        F2 = np.frombuffer(bF, dtype=np.dtype(tF.column("dtype")[0].as_py())).reshape(tuple(tF.column("shape")[0].as_py()))
        X2 = np.frombuffer(bX, dtype=np.dtype(tX.column("dtype")[0].as_py())).reshape(tuple(tX.column("shape")[0].as_py()))
        return F2, X2

    ipc_decode_ms = t(ipc_decode, n=5) * 1000

    print(f"{dtype_name}:  npy decode {npy_decode_ms:6.1f} ms   "
          f"arrow_ipc decode {ipc_decode_ms:6.1f} ms   "
          f"parquet decode {pq_decode_ms:6.1f} ms   "
          f"(F+X bytes  npy={len(npy_F_bytes)+len(npy_X_bytes):>10,d}  "
          f"ipc={len(ipc_F_buf)+len(ipc_X_buf):>10,d}  "
          f"parquet={len(pq_F_bytes)+len(pq_X_bytes):>10,d})")
    del F, X, Q, diag
    gc.collect()
