"""Server-side timing for covariance payload formats.

For each (format, n) in {A,B,D,E} x {200, 1000, 3600}:
  - encode 3 times, take median wall-clock ms
  - record raw payload bytes
  - record gzip(level=6) bytes

Output: bench/results/server_bench.json
"""
from __future__ import annotations

import gzip
import json
import statistics
import time
from pathlib import Path

import numpy as np

from bench import formats

SIZES = [200, 1000, 3600]
TRIALS = 3
RESULTS_DIR = Path(__file__).parent / "results"


def synth(n: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(n)  # deterministic per size
    ids = [f"F_{i}" for i in range(n)]
    raw = rng.standard_normal((n, n)).astype(np.float32)
    sym = (raw + raw.T) / 2.0
    np.fill_diagonal(sym, 1.0)
    return ids, sym


def time_encode(spec: formats.FormatSpec, ids, matrix) -> tuple[float, bytes]:
    samples = []
    payload = b""
    for _ in range(TRIALS):
        t0 = time.perf_counter()
        payload, _ = spec.encode(ids, matrix)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples), payload


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    rows = []
    for n in SIZES:
        print(f"size n={n} ({n*n:,} cells)")
        ids, matrix = synth(n)
        for code, spec in formats.REGISTRY.items():
            encode_ms, payload = time_encode(spec, ids, matrix)
            raw_bytes = len(payload)
            gz_bytes = len(gzip.compress(payload, compresslevel=6))
            print(
                f"  [{code}] {spec.name:32s}  encode={encode_ms:7.1f} ms  "
                f"raw={raw_bytes/1e6:7.2f} MB  gz={gz_bytes/1e6:7.2f} MB"
            )
            rows.append(
                {
                    "n": n,
                    "format": code,
                    "encode_ms": encode_ms,
                    "raw_bytes": raw_bytes,
                    "gzip_bytes": gz_bytes,
                }
            )
    out = RESULTS_DIR / "server_bench.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
