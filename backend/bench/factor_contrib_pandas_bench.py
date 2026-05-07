"""Pandas vs numpy on the same factor-contribution workload.

K = 3000 factors, P = 500 portfolios. Three things measured:

  1. Compute  (matmul + Hadamard + sum)
       a. all-numpy
       b. all-pandas with aligned indices (the realistic pandas path)
       c. pandas -> .to_numpy() at the boundary, then numpy
       d. pandas with mismatched index ordering (forces reindex)
  2. Read     (parquet -> ndarray vs parquet -> DataFrame)
  3. Write    (ndarray -> parquet vs DataFrame -> parquet, also feather)

Run from backend/:
  uv run python -m bench.factor_contrib_pandas_bench
"""
from __future__ import annotations
import gc
import io
import time
import numpy as np
import pandas as pd
import pyarrow as pa
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
print(f"pandas {pd.__version__}, numpy {np.__version__}, pyarrow {pa.__version__}")
print(f"K={K} P={P}")
print()

rng = np.random.default_rng(0)
factor_names = [f"f{i:04d}" for i in range(K)]
portfolio_names = [f"p{j:04d}" for j in range(P)]

for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
    print("=" * 80)
    print(f"  dtype = {dtype_name}")
    print("=" * 80)

    # ---- Build the data ----
    Q = rng.standard_normal((K, 60)).astype(dtype)
    diag = (0.3 + rng.random(K).astype(dtype)) ** 2
    F = (Q @ Q.T + np.diag(diag)).astype(dtype)
    X = rng.standard_normal((K, P)).astype(dtype)

    F_df = pd.DataFrame(F, index=factor_names, columns=factor_names)
    X_df = pd.DataFrame(X, index=factor_names, columns=portfolio_names)
    # Reorder X's index to force reindex on Hadamard
    shuffled = factor_names[1:] + factor_names[:1]
    X_df_misaligned = X_df.loc[shuffled]

    # ============ 1. COMPUTE ============
    print("\nCompute path (matmul + Hadamard + sum):")

    def all_numpy():
        M = F @ X
        C = X * M
        return C.sum(0)

    def all_pandas_aligned():
        M = F_df @ X_df
        C = X_df * M
        return C.sum(axis=0)

    def pandas_to_numpy_boundary():
        F_arr = F_df.to_numpy()
        X_arr = X_df.to_numpy()
        M = F_arr @ X_arr
        C = X_arr * M
        return C.sum(0)

    def pandas_misaligned():
        # X has wrong row order — pandas will reindex
        M = F_df @ X_df_misaligned
        C = X_df_misaligned * M
        return C.sum(axis=0)

    np_ms = t(all_numpy, n=5) * 1000
    pdal_ms = t(all_pandas_aligned, n=5) * 1000
    pd2np_ms = t(pandas_to_numpy_boundary, n=5) * 1000
    pdmis_ms = t(pandas_misaligned, n=3) * 1000

    print(f"   all-numpy                  {np_ms:7.2f} ms  (1.0x baseline)")
    print(f"   all-pandas (aligned)       {pdal_ms:7.2f} ms  ({pdal_ms/np_ms:.2f}x numpy)")
    print(f"   pandas -> .to_numpy() -> np  {pd2np_ms:7.2f} ms  ({pd2np_ms/np_ms:.2f}x numpy)")
    print(f"   pandas (misaligned index)  {pdmis_ms:7.2f} ms  ({pdmis_ms/np_ms:.2f}x numpy)")

    # ============ 2. READ ============
    print("\nRead path (parquet on disk -> in-memory):")

    # Pre-write a parquet file for F (the larger payload)
    tmpdir = (lambda p: (p.mkdir(exist_ok=True), p)[1])(__import__("pathlib").Path("./_pandas_bench_tmp"))
    pq_path = tmpdir / f"F_{dtype_name}.parquet"
    F_df.to_parquet(pq_path, engine="pyarrow", compression="snappy")
    pq_size = pq_path.stat().st_size

    def read_parquet_to_numpy():
        tbl = pq.read_table(str(pq_path))
        return np.column_stack([c.to_numpy(zero_copy_only=False) for c in tbl.columns])

    def read_parquet_to_pandas():
        return pd.read_parquet(str(pq_path), engine="pyarrow")

    np_read_ms = t(read_parquet_to_numpy, n=3) * 1000
    pd_read_ms = t(read_parquet_to_pandas, n=3) * 1000
    print(f"   F parquet on disk: {pq_size/1e6:5.1f} MB")
    print(f"   parquet -> numpy ndarray   {np_read_ms:7.2f} ms")
    print(f"   parquet -> pandas DataFrame {pd_read_ms:7.2f} ms  ({pd_read_ms/np_read_ms:.2f}x numpy)")

    # Also: read into pandas then convert
    def read_parquet_to_pandas_then_numpy():
        df = pd.read_parquet(str(pq_path), engine="pyarrow")
        return df.to_numpy()
    pd_then_np_ms = t(read_parquet_to_pandas_then_numpy, n=3) * 1000
    print(f"   parquet -> pandas -> .to_numpy() {pd_then_np_ms:7.2f} ms  ({pd_then_np_ms/np_read_ms:.2f}x numpy-direct)")

    # ============ 3. WRITE ============
    print("\nWrite path (in-memory -> parquet bytes):")

    def write_numpy_via_pa(arr):
        # numpy ndarray -> column-per-portfolio parquet (the path used in earlier benches)
        pa_dtype = pa.float32() if dtype_name == "float32" else pa.float64()
        cols = {f"f{j:04d}": pa.array(arr[:, j], type=pa_dtype) for j in range(arr.shape[1])}
        sink = pa.BufferOutputStream()
        pq.write_table(pa.table(cols), sink, compression="snappy")
        return sink.getvalue()

    def write_pandas_to_parquet(df):
        buf = io.BytesIO()
        df.to_parquet(buf, engine="pyarrow", compression="snappy")
        return buf.getvalue()

    def write_pandas_to_feather(df):
        buf = io.BytesIO()
        df.to_feather(buf)
        return buf.getvalue()

    def write_pandas_to_pickle(df):
        return df.to_pickle(None)  # returns bytes when path is None? actually no — use BytesIO
    def write_pandas_to_pickle_via_buf(df):
        buf = io.BytesIO()
        df.to_pickle(buf)
        return buf.getvalue()

    np_write_ms = t(lambda: write_numpy_via_pa(F), n=3) * 1000
    pd_pq_ms = t(lambda: write_pandas_to_parquet(F_df), n=3) * 1000
    pd_ft_ms = t(lambda: write_pandas_to_feather(F_df), n=3) * 1000
    pd_pkl_ms = t(lambda: write_pandas_to_pickle_via_buf(F_df), n=3) * 1000

    np_pq_bytes = len(write_numpy_via_pa(F))
    pd_pq_bytes = len(write_pandas_to_parquet(F_df))
    pd_ft_bytes = len(write_pandas_to_feather(F_df))
    pd_pkl_bytes = len(write_pandas_to_pickle_via_buf(F_df))

    print(f"   numpy -> parquet  (pa.table column-per-factor)  "
          f"{np_write_ms:7.1f} ms  {np_pq_bytes/1e6:5.1f} MB")
    print(f"   pandas -> parquet (df.to_parquet)               "
          f"{pd_pq_ms:7.1f} ms  {pd_pq_bytes/1e6:5.1f} MB  ({pd_pq_ms/np_write_ms:.2f}x numpy)")
    print(f"   pandas -> feather (df.to_feather, Arrow IPC)    "
          f"{pd_ft_ms:7.1f} ms  {pd_ft_bytes/1e6:5.1f} MB")
    print(f"   pandas -> pickle  (df.to_pickle)                "
          f"{pd_pkl_ms:7.1f} ms  {pd_pkl_bytes/1e6:5.1f} MB")

    print()
    del F, X, F_df, X_df, X_df_misaligned, Q, diag
    gc.collect()
