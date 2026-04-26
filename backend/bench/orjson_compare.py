"""Side experiment: does orjson reduce server-side encode time for formats A and B?

Compares 4 paths at each size:
  A.stdlib    : json.dumps over a list of 12.96M dicts
  A.orjson    : orjson.dumps over the same list of dicts
  B.stdlib    : json.dumps over matrix.tolist()
  B.orjson_np : orjson.dumps with OPT_SERIALIZE_NUMPY (skips .tolist())

Output: bench/results/orjson_compare.json (printed too)
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np
import orjson

from bench.server_bench import synth, SIZES, TRIALS

RESULTS_DIR = Path(__file__).parent / "results"


def encode_a_stdlib(ids, matrix):
    n = len(ids)
    rows = [{"i": int(i), "j": int(j), "v": float(matrix[i, j])} for i in range(n) for j in range(n)]
    return json.dumps({"factor_ids": ids, "cells": rows}).encode("utf-8")


def encode_a_orjson(ids, matrix):
    n = len(ids)
    rows = [{"i": int(i), "j": int(j), "v": float(matrix[i, j])} for i in range(n) for j in range(n)]
    return orjson.dumps({"factor_ids": ids, "cells": rows})


def encode_b_stdlib(ids, matrix):
    return json.dumps({"factor_ids": ids, "matrix": matrix.tolist()}).encode("utf-8")


def encode_b_orjson_np(ids, matrix):
    # orjson can serialize numpy arrays directly when OPT_SERIALIZE_NUMPY is set —
    # this skips the expensive .tolist() conversion.
    return orjson.dumps({"factor_ids": ids, "matrix": matrix}, option=orjson.OPT_SERIALIZE_NUMPY)


VARIANTS = {
    "A.stdlib":    encode_a_stdlib,
    "A.orjson":    encode_a_orjson,
    "B.stdlib":    encode_b_stdlib,
    "B.orjson_np": encode_b_orjson_np,
}


def time_encode(fn, ids, matrix):
    samples = []
    payload = b""
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        payload = fn(ids, matrix)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), payload


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for n in SIZES:
        print(f"\nsize n={n} ({n*n:,} cells)")
        ids, matrix = synth(n)
        for label, fn in VARIANTS.items():
            ms, payload = time_encode(fn, ids, matrix)
            print(f"  {label:14s}  encode={ms:9.1f} ms  raw={len(payload)/1e6:7.2f} MB")
            rows.append({"n": n, "variant": label, "encode_ms": ms, "raw_bytes": len(payload)})
    out = RESULTS_DIR / "orjson_compare.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
