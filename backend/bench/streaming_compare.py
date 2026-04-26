"""Side experiment: does FastAPI StreamingResponse help on format A?

Compares two encoder paths for format A at n=3600:
  - buffered: build the whole bytes object in memory, return Response
  - streaming: yield JSON tokens incrementally — never materialize the
    full payload in Python heap

Measures server peak memory (tracemalloc) + total wall time.
This is a server-side measurement only; the browser-side V8 string
ceiling is unchanged regardless of how the bytes arrived.
"""
from __future__ import annotations

import gc
import json
import time
import tracemalloc
from typing import Iterator

import numpy as np

from bench.server_bench import synth


def encode_a_buffered(ids: list[str], matrix: np.ndarray) -> bytes:
    n = len(ids)
    rows = [
        {"i": int(i), "j": int(j), "v": float(matrix[i, j])}
        for i in range(n)
        for j in range(n)
    ]
    body = {"factor_ids": ids, "cells": rows}
    return json.dumps(body).encode("utf-8")


def stream_a(ids: list[str], matrix: np.ndarray) -> Iterator[bytes]:
    """Yield format A as JSON tokens. Never builds the full list in memory."""
    n = len(ids)
    yield b'{"factor_ids":'
    yield json.dumps(ids).encode("utf-8")
    yield b',"cells":['
    first = True
    for i in range(n):
        row = matrix[i]
        for j in range(n):
            prefix = b"" if first else b","
            first = False
            cell = b'{"i":%d,"j":%d,"v":%s}' % (i, j, repr(float(row[j])).encode("ascii"))
            yield prefix + cell
    yield b"]}"


def measure_buffered(ids, matrix):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    payload = encode_a_buffered(ids, matrix)
    encode_ms = (time.perf_counter() - t0) * 1000
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "path": "buffered",
        "encode_ms": encode_ms,
        "peak_python_mb": peak / 1e6,
        "payload_bytes": len(payload),
    }


def measure_streaming(ids, matrix):
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    total = 0
    chunks = 0
    first_chunk_ms = None
    peak_during = 0
    for chunk in stream_a(ids, matrix):
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - t0) * 1000
        total += len(chunk)
        chunks += 1
        # Sample peak during streaming, not just at the end.
        if chunks % 100_000 == 0:
            cur, _ = tracemalloc.get_traced_memory()
            if cur > peak_during:
                peak_during = cur
    encode_ms = (time.perf_counter() - t0) * 1000
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "path": "streaming",
        "encode_ms": encode_ms,
        "peak_python_mb": peak / 1e6,
        "first_chunk_ms": first_chunk_ms,
        "payload_bytes": total,
        "chunks": chunks,
    }


def main():
    print("Building synthetic n=3600 matrix...")
    ids, matrix = synth(3600)

    print("\n--- buffered (current behaviour) ---")
    r = measure_buffered(ids, matrix)
    print(f"  encode wall time   : {r['encode_ms']:9.0f} ms")
    print(f"  peak Python heap   : {r['peak_python_mb']:9.1f} MB")
    print(f"  payload size       : {r['payload_bytes']/1e6:9.1f} MB")
    print(f"  TTFB (server side) : {r['encode_ms']:9.0f} ms (full body must be built first)")

    print("\n--- streaming (yield each cell) ---")
    r = measure_streaming(ids, matrix)
    print(f"  encode wall time   : {r['encode_ms']:9.0f} ms")
    print(f"  peak Python heap   : {r['peak_python_mb']:9.1f} MB")
    print(f"  payload size       : {r['payload_bytes']/1e6:9.1f} MB")
    print(f"  TTFB (server side) : {r['first_chunk_ms']:9.1f} ms (first chunk yielded)")
    print(f"  chunks emitted     : {r['chunks']:>9}")


if __name__ == "__main__":
    main()
