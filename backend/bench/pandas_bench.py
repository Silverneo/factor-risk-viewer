"""Pandas conversion timing per format.

For each (format, n), convert the encoded wire bytes to a long-form pandas
DataFrame with columns [factor_a, factor_b, value]. Times only the conversion
step (decode + reshape), not encode.

Output: bench/results/pandas_bench.json
"""
from __future__ import annotations

import io
import json
import statistics
import struct
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc

from bench import formats
from bench.server_bench import synth, SIZES, TRIALS

RESULTS_DIR = (
    Path(__file__).parents[2]
    / "experiments"
    / "2026-04-26-covariance-payload-formats"
    / "results"
)


def to_long_df(ids: list[str], matrix: np.ndarray) -> pd.DataFrame:
    n = len(ids)
    ai, aj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return pd.DataFrame(
        {
            "factor_a": np.take(ids, ai.ravel()),
            "factor_b": np.take(ids, aj.ravel()),
            "value": matrix.ravel(),
        }
    )


def convert_a(payload: bytes) -> pd.DataFrame:
    body = json.loads(payload)
    ids = body["factor_ids"]
    df = pd.DataFrame(body["cells"])
    df["factor_a"] = np.take(ids, df["i"].values)
    df["factor_b"] = np.take(ids, df["j"].values)
    return df[["factor_a", "factor_b", "v"]].rename(columns={"v": "value"})


def convert_b(payload: bytes) -> pd.DataFrame:
    body = json.loads(payload)
    ids = body["factor_ids"]
    matrix = np.asarray(body["matrix"], dtype=np.float32)
    return to_long_df(ids, matrix)


def convert_d(payload: bytes) -> pd.DataFrame:
    reader = ipc.open_stream(io.BytesIO(payload))
    table = reader.read_all()
    ids = json.loads(reader.schema.metadata[b"factor_ids"])
    n = len(ids)
    flat = table.column("values").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    matrix = flat.reshape(n, n).astype(np.float32)
    return to_long_df(ids, matrix)


def convert_e(payload: bytes) -> pd.DataFrame:
    (sidecar_len,) = struct.unpack_from("<I", payload, 0)
    ids = json.loads(payload[4 : 4 + sidecar_len])["factor_ids"]
    (n,) = struct.unpack_from("<I", payload, 4 + sidecar_len)
    body_offset = 4 + sidecar_len + 4
    matrix = np.frombuffer(payload, dtype=np.float32, count=n * n, offset=body_offset).reshape(n, n)
    return to_long_df(ids, matrix)


CONVERTERS = {"A": convert_a, "B": convert_b, "D": convert_d, "E": convert_e}


def time_convert(fn, payload: bytes) -> float:
    samples = []
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        fn(payload)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for n in SIZES:
        print(f"size n={n}")
        ids, matrix = synth(n)
        for code, spec in formats.REGISTRY.items():
            payload, _ = spec.encode(ids, matrix)
            ms = time_convert(CONVERTERS[code], payload)
            print(f"  [{code}] {spec.name:32s}  convert={ms:7.1f} ms")
            rows.append({"n": n, "format": code, "convert_ms": ms})
    out = RESULTS_DIR / "pandas_bench.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
