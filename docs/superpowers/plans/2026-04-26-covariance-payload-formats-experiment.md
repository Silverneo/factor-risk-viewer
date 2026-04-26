# Covariance Payload Formats Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmark harness that compares 4 covariance payload formats (list-of-dict JSON, 2D nested array JSON, Apache Arrow IPC, raw Float32Array + JSON sidecar) across 3 size points (200/1000/3600 factors) on encode time, wire size (raw + gzip), client-side decode time, JS heap delta, AG Grid scroll FPS, and pandas conversion time. Produce a written report at `docs/experiments/2026-04-26-covariance-payload-formats.md`.

**Architecture:** Backend has a new `bench/` module with one `formats.py` containing all 4 encoders + decoders + a registry. A standalone `server_bench.py` script measures encode/size; a `pandas_bench.py` measures conversion. A new gated endpoint `GET /api/_bench/covariance` reuses `formats.py` to serve test payloads to the browser. Frontend adds a `'bench'` view to the existing `AppView` switch and a new `CovarianceBench.tsx` component that fetches each (format, size) tuple, measures decode + heap, runs an AG Grid scroll FPS sub-test on n=3600, and renders results in a table with a "download JSON" button.

**Tech Stack:** Python 3.11 / FastAPI / DuckDB / NumPy / PyArrow on the backend; React 19 / Vite / AG Grid Enterprise on the frontend. New frontend dep: `apache-arrow`. Tests use `pytest`.

**Spec:** `docs/superpowers/specs/2026-04-26-covariance-payload-formats-experiment-design.md`

---

## File map

**Created:**
- `backend/bench/__init__.py` — empty marker
- `backend/bench/formats.py` — encoders, decoders, registry
- `backend/bench/test_formats.py` — round-trip equality test
- `backend/bench/server_bench.py` — server-side timing script
- `backend/bench/pandas_bench.py` — pandas conversion timing script
- `backend/bench/results/` — output directory for JSON results (created at runtime)
- `frontend/src/bench/CovarianceBench.tsx` — frontend bench page
- `docs/experiments/2026-04-26-covariance-payload-formats.md` — final report

**Modified:**
- `backend/api.py` — add gated `/api/_bench/covariance` endpoint
- `backend/pyproject.toml` — add `pytest` dev dep
- `frontend/src/App.tsx` — extend `AppView` union with `'bench'`, add tab + render branch
- `frontend/package.json` — add `apache-arrow` dep

---

## Task 1: Backend bench module scaffold

**Files:**
- Create: `backend/bench/__init__.py`
- Modify: `backend/pyproject.toml` (add pytest)

- [ ] **Step 1: Create empty package marker**

```bash
mkdir -p backend/bench/results
touch backend/bench/__init__.py
```

- [ ] **Step 2: Add pytest as a dev dep**

Modify `backend/pyproject.toml` — append after the `dependencies` block:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
]
```

- [ ] **Step 3: Sync deps**

Run from `backend/`:
```bash
uv sync --group dev
```
Expected: pytest installed, no errors.

- [ ] **Step 4: Commit**

```bash
git add backend/bench/__init__.py backend/pyproject.toml backend/uv.lock
git commit -m "bench: scaffold backend/bench module + pytest dev dep"
```

---

## Task 2: Format A — list-of-dict JSON

**Files:**
- Create: `backend/bench/formats.py`
- Create: `backend/bench/test_formats.py`

- [ ] **Step 1: Write the failing test**

Create `backend/bench/test_formats.py`:

```python
import numpy as np
import pytest

from bench import formats


def make_input(n: int) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(42)
    ids = [f"F_{i}" for i in range(n)]
    raw = rng.standard_normal((n, n)).astype(np.float32)
    sym = (raw + raw.T) / 2.0  # symmetric, like a real cov matrix
    np.fill_diagonal(sym, 1.0)
    return ids, sym


@pytest.mark.parametrize("fmt", ["A"])
def test_roundtrip(fmt: str):
    ids, matrix = make_input(20)
    payload, content_type = formats.REGISTRY[fmt].encode(ids, matrix)
    assert isinstance(payload, (bytes, bytearray))
    assert isinstance(content_type, str) and content_type
    decoded_ids, decoded_matrix = formats.REGISTRY[fmt].decode(payload)
    assert decoded_ids == ids
    np.testing.assert_allclose(decoded_matrix, matrix, rtol=0, atol=1e-6)
```

- [ ] **Step 2: Run the test, verify it fails**

Run from `backend/`:
```bash
uv run pytest bench/test_formats.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'bench.formats'`.

- [ ] **Step 3: Implement `formats.py` with format A only**

Create `backend/bench/formats.py`:

```python
"""Covariance payload format encoders/decoders for the benchmark harness.

Each format implements `encode(ids, matrix) -> (bytes, content_type)` and
`decode(bytes) -> (ids, matrix)`. The decoder exists only for round-trip
correctness testing — production code never calls it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class FormatSpec:
    name: str
    description: str
    encode: Callable[[list[str], np.ndarray], tuple[bytes, str]]
    decode: Callable[[bytes], tuple[list[str], np.ndarray]]


# ---------- Format A: list-of-dict JSON --------------------------------------

def _encode_a(ids: list[str], matrix: np.ndarray) -> tuple[bytes, str]:
    n = len(ids)
    rows = [
        {"i": int(i), "j": int(j), "v": float(matrix[i, j])}
        for i in range(n)
        for j in range(n)
    ]
    body = {"factor_ids": ids, "cells": rows}
    return json.dumps(body).encode("utf-8"), "application/json"


def _decode_a(payload: bytes) -> tuple[list[str], np.ndarray]:
    body = json.loads(payload)
    ids = body["factor_ids"]
    n = len(ids)
    matrix = np.zeros((n, n), dtype=np.float32)
    for cell in body["cells"]:
        matrix[cell["i"], cell["j"]] = cell["v"]
    return ids, matrix


REGISTRY: dict[str, FormatSpec] = {
    "A": FormatSpec(
        name="list-of-dict JSON",
        description="Row-shaped JSON: [{i, j, v}, ...]",
        encode=_encode_a,
        decode=_decode_a,
    ),
}
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
uv run pytest bench/test_formats.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/bench/formats.py backend/bench/test_formats.py
git commit -m "bench: add format A (list-of-dict JSON) encoder + roundtrip test"
```

---

## Task 3: Format B — 2D nested array JSON

**Files:**
- Modify: `backend/bench/formats.py`
- Modify: `backend/bench/test_formats.py`

- [ ] **Step 1: Extend the parametrized test**

In `backend/bench/test_formats.py`, change the parametrize line:

```python
@pytest.mark.parametrize("fmt", ["A", "B"])
```

- [ ] **Step 2: Run the test, verify it fails for B**

```bash
uv run pytest bench/test_formats.py -v
```
Expected: B fails with `KeyError: 'B'`.

- [ ] **Step 3: Implement format B**

Append to `backend/bench/formats.py`, before the `REGISTRY` assignment:

```python
# ---------- Format B: 2D nested array JSON -----------------------------------

def _encode_b(ids: list[str], matrix: np.ndarray) -> tuple[bytes, str]:
    body = {"factor_ids": ids, "matrix": matrix.tolist()}
    return json.dumps(body).encode("utf-8"), "application/json"


def _decode_b(payload: bytes) -> tuple[list[str], np.ndarray]:
    body = json.loads(payload)
    return body["factor_ids"], np.asarray(body["matrix"], dtype=np.float32)
```

And add to `REGISTRY`:

```python
    "B": FormatSpec(
        name="2D nested array JSON",
        description="{factor_ids, matrix: [[...]]}",
        encode=_encode_b,
        decode=_decode_b,
    ),
```

- [ ] **Step 4: Run the test, verify it passes**

```bash
uv run pytest bench/test_formats.py -v
```
Expected: A and B both PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/bench/formats.py backend/bench/test_formats.py
git commit -m "bench: add format B (2D nested array JSON)"
```

---

## Task 4: Format D — Apache Arrow IPC stream

**Files:**
- Modify: `backend/bench/formats.py`
- Modify: `backend/bench/test_formats.py`

- [ ] **Step 1: Extend the parametrized test**

```python
@pytest.mark.parametrize("fmt", ["A", "B", "D"])
```

- [ ] **Step 2: Run, verify D fails**

```bash
uv run pytest bench/test_formats.py -v
```
Expected: D fails with `KeyError: 'D'`.

- [ ] **Step 3: Implement format D**

Append to `formats.py`:

```python
# ---------- Format D: Apache Arrow IPC stream --------------------------------
# Schema: factor_ids in stream metadata; one record batch per row of the
# matrix, with one Float32 column per matrix row. This keeps the schema flat
# and lets the JS reader iterate row-by-row.

import io
import pyarrow as pa
import pyarrow.ipc as ipc


def _encode_d(ids: list[str], matrix: np.ndarray) -> tuple[bytes, str]:
    n = len(ids)
    # Single record batch: n rows, 2 columns: row_idx (int32), values (list<float32>)
    arr_idx = pa.array(np.arange(n, dtype=np.int32))
    arr_values = pa.array(
        [matrix[i].tolist() for i in range(n)],
        type=pa.list_(pa.float32(), n),
    )
    schema = pa.schema(
        [("row_idx", pa.int32()), ("values", pa.list_(pa.float32(), n))],
        metadata={b"factor_ids": json.dumps(ids).encode("utf-8")},
    )
    batch = pa.record_batch([arr_idx, arr_values], schema=schema)
    sink = io.BytesIO()
    with ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue(), "application/vnd.apache.arrow.stream"


def _decode_d(payload: bytes) -> tuple[list[str], np.ndarray]:
    reader = ipc.open_stream(io.BytesIO(payload))
    table = reader.read_all()
    ids_meta = reader.schema.metadata[b"factor_ids"]
    ids = json.loads(ids_meta)
    n = len(ids)
    # values column is FixedSizeListArray; flatten to 2D
    flat = table.column("values").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    matrix = flat.reshape(n, n).astype(np.float32)
    return ids, matrix
```

Add to `REGISTRY`:

```python
    "D": FormatSpec(
        name="Apache Arrow IPC stream",
        description="Binary columnar; ids in schema metadata; FixedSizeList<float32> rows",
        encode=_encode_d,
        decode=_decode_d,
    ),
```

- [ ] **Step 4: Run, verify all three pass**

```bash
uv run pytest bench/test_formats.py -v
```
Expected: A, B, D all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/bench/formats.py backend/bench/test_formats.py
git commit -m "bench: add format D (Apache Arrow IPC stream)"
```

---

## Task 5: Format E — raw Float32Array + JSON sidecar

**Files:**
- Modify: `backend/bench/formats.py`
- Modify: `backend/bench/test_formats.py`

- [ ] **Step 1: Extend test**

```python
@pytest.mark.parametrize("fmt", ["A", "B", "D", "E"])
```

- [ ] **Step 2: Run, verify E fails**

Same `pytest` command. Expected: E missing.

- [ ] **Step 3: Implement format E**

Wire format: 4-byte little-endian uint32 N (sidecar length) + N-byte UTF-8 JSON `{"factor_ids":[...]}` + 4-byte uint32 (matrix side length, sanity) + raw little-endian Float32 buffer of size n*n*4 bytes.

Append to `formats.py`:

```python
# ---------- Format E: raw Float32Array + JSON sidecar ------------------------
# Wire layout (all little-endian):
#   u32 sidecar_len
#   sidecar_len bytes: utf-8 JSON {"factor_ids": [...]}
#   u32 n  (matrix side length, sanity check)
#   n*n*4 bytes: row-major float32 matrix

import struct


def _encode_e(ids: list[str], matrix: np.ndarray) -> tuple[bytes, str]:
    sidecar = json.dumps({"factor_ids": ids}).encode("utf-8")
    n = len(ids)
    header = struct.pack("<I", len(sidecar)) + sidecar + struct.pack("<I", n)
    body = matrix.astype(np.float32, copy=False).tobytes(order="C")
    return header + body, "application/octet-stream"


def _decode_e(payload: bytes) -> tuple[list[str], np.ndarray]:
    (sidecar_len,) = struct.unpack_from("<I", payload, 0)
    sidecar = json.loads(payload[4 : 4 + sidecar_len])
    ids = sidecar["factor_ids"]
    (n,) = struct.unpack_from("<I", payload, 4 + sidecar_len)
    body_offset = 4 + sidecar_len + 4
    matrix = np.frombuffer(payload, dtype=np.float32, count=n * n, offset=body_offset)
    return ids, matrix.reshape(n, n).copy()
```

Add to `REGISTRY`:

```python
    "E": FormatSpec(
        name="Float32Array + JSON sidecar",
        description="Lower-bound binary: u32 sidecar_len, JSON {ids}, u32 n, row-major float32",
        encode=_encode_e,
        decode=_decode_e,
    ),
```

- [ ] **Step 4: Run all four**

```bash
uv run pytest bench/test_formats.py -v
```
Expected: A, B, D, E all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/bench/formats.py backend/bench/test_formats.py
git commit -m "bench: add format E (Float32Array + JSON sidecar)"
```

---

## Task 6: Server-side benchmark script

**Files:**
- Create: `backend/bench/server_bench.py`

- [ ] **Step 1: Write the script**

Create `backend/bench/server_bench.py`:

```python
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
```

- [ ] **Step 2: Smoke-run on small sizes only first**

Edit `SIZES = [200]` temporarily, then:
```bash
cd backend && uv run python -m bench.server_bench
```
Expected: prints 4 rows (A/B/D/E at n=200), writes JSON. Numbers will be small.

- [ ] **Step 3: Restore full sizes**

Set `SIZES = [200, 1000, 3600]` again.

- [ ] **Step 4: Commit**

```bash
git add backend/bench/server_bench.py
git commit -m "bench: server-side encode/size benchmark script"
```

---

## Task 7: Pandas conversion benchmark

**Files:**
- Create: `backend/bench/pandas_bench.py`

- [ ] **Step 1: Write the script**

Create `backend/bench/pandas_bench.py`:

```python
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

RESULTS_DIR = Path(__file__).parent / "results"


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
```

- [ ] **Step 2: Smoke-run**

```bash
cd backend && uv run python -m bench.pandas_bench
```
Expected: prints 4 rows × 3 sizes = 12 lines, writes JSON.

- [ ] **Step 3: Commit**

```bash
git add backend/bench/pandas_bench.py
git commit -m "bench: pandas conversion timing per format"
```

---

## Task 8: Gated bench endpoint in api.py

**Files:**
- Modify: `backend/api.py`

- [ ] **Step 1: Add the endpoint**

Append to `backend/api.py` after the existing covariance endpoints (after the function ending around line 462):

```python
# ---------- Bench endpoint (gated) -------------------------------------------
# Serves synthetic covariance payloads in 4 formats for the frontend bench
# harness. Gated by env var FACTOR_RISK_BENCH=1 because the app has no auth.

import os as _os

if _os.environ.get("FACTOR_RISK_BENCH") == "1":
    from bench import formats as _bench_formats  # type: ignore
    from bench.server_bench import synth as _bench_synth  # type: ignore

    @app.get("/api/_bench/covariance")
    def _bench_covariance(format: str, n: int) -> Response:
        if format not in _bench_formats.REGISTRY:
            raise HTTPException(400, f"unknown format {format!r}")
        if n not in (200, 1000, 3600):
            raise HTTPException(400, f"n must be one of 200/1000/3600, got {n}")
        ids, matrix = _bench_synth(n)
        spec = _bench_formats.REGISTRY[format]
        payload, content_type = spec.encode(ids, matrix)
        return Response(content=payload, media_type=content_type)
```

- [ ] **Step 2: Restart backend with the flag**

In whichever shell runs uvicorn:
```bash
cd backend && FACTOR_RISK_BENCH=1 uv run uvicorn api:app --port 8000 --log-level warning
```

- [ ] **Step 3: Smoke-test each format**

In another shell:
```bash
curl -s -o /dev/null -w "%{http_code} %{size_download}\n" "http://localhost:8000/api/_bench/covariance?format=A&n=200"
curl -s -o /dev/null -w "%{http_code} %{size_download}\n" "http://localhost:8000/api/_bench/covariance?format=B&n=200"
curl -s -o /dev/null -w "%{http_code} %{size_download}\n" "http://localhost:8000/api/_bench/covariance?format=D&n=200"
curl -s -o /dev/null -w "%{http_code} %{size_download}\n" "http://localhost:8000/api/_bench/covariance?format=E&n=200"
```
Expected: all `200 <bytes>`. Sizes: A largest, E smallest, D and B in between.

- [ ] **Step 4: Verify gating**

Restart without the env var; the route should 404:
```bash
curl -s -o /dev/null -w "%{http_code}\n" "http://localhost:8000/api/_bench/covariance?format=A&n=200"
```
Expected: `404`.

Re-enable for the rest of the work:
```bash
FACTOR_RISK_BENCH=1 uv run uvicorn api:app --port 8000 --log-level warning
```

- [ ] **Step 5: Commit**

```bash
git add backend/api.py
git commit -m "bench: gated /api/_bench/covariance endpoint serving 4 formats"
```

---

## Task 9: Frontend dep — apache-arrow

**Files:**
- Modify: `frontend/package.json` and `frontend/package-lock.json` (or yarn/pnpm lockfile)

- [ ] **Step 1: Install**

From `frontend/`:
```bash
npm install apache-arrow@^17
```
(Note: `npm` is not on the default bash PATH. From bash, prepend `export PATH="/c/Users/CMZHA/AppData/Roaming/fnm/node-versions/v24.15.0/installation:$PATH"` first.)

Expected: `apache-arrow` appears in `dependencies`.

- [ ] **Step 2: Verify import works**

Quick smoke test from `frontend/`:
```bash
node -e "console.log(Object.keys(require('apache-arrow')).slice(0,5))"
```
Expected: prints an array of arrow exports (e.g. `[ 'Schema', 'Field', 'Type', ... ]`).

- [ ] **Step 3: Commit**

```bash
git add frontend/package.json frontend/package-lock.json
git commit -m "bench: add apache-arrow frontend dep"
```

---

## Task 10: Frontend bench page scaffold

**Files:**
- Create: `frontend/src/bench/CovarianceBench.tsx`
- Modify: `frontend/src/App.tsx` (lines around the `AppView` union and the view-switch render)

- [ ] **Step 1: Create the bench component skeleton**

Create `frontend/src/bench/CovarianceBench.tsx`:

```tsx
import { useState } from 'react'

const API = import.meta.env.VITE_API ?? 'http://localhost:8000'
const FORMATS = ['A', 'B', 'D', 'E'] as const
const SIZES = [200, 1000, 3600] as const

type Row = {
  n: number
  format: string
  decodeMs: number
  heapDeltaMb: number | null
  payloadBytes: number
}

export function CovarianceBench() {
  const [running, setRunning] = useState(false)
  const [rows, setRows] = useState<Row[]>([])

  async function run() {
    setRunning(true)
    setRows([])
    const collected: Row[] = []
    for (const n of SIZES) {
      for (const fmt of FORMATS) {
        const url = `${API}/api/_bench/covariance?format=${fmt}&n=${n}`
        const heapBefore = (performance as any).memory?.usedJSHeapSize ?? null
        const t0 = performance.now()
        const buf = await fetch(url).then(r => r.arrayBuffer())
        // decode-step measurement happens in Task 11; for now just record sizes.
        const decodeMs = performance.now() - t0
        const heapAfter = (performance as any).memory?.usedJSHeapSize ?? null
        const heapDeltaMb =
          heapBefore != null && heapAfter != null
            ? (heapAfter - heapBefore) / 1e6
            : null
        collected.push({
          n,
          format: fmt,
          decodeMs,
          heapDeltaMb,
          payloadBytes: buf.byteLength,
        })
        setRows([...collected])
      }
    }
    setRunning(false)
  }

  function download() {
    const blob = new Blob([JSON.stringify(rows, null, 2)], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'frontend_bench.json'
    a.click()
  }

  return (
    <div style={{ padding: 16, fontFamily: 'monospace' }}>
      <h2>Covariance payload bench</h2>
      <button onClick={run} disabled={running}>
        {running ? 'Running…' : 'Run'}
      </button>{' '}
      <button onClick={download} disabled={rows.length === 0}>
        Download JSON
      </button>
      <table style={{ marginTop: 12, borderCollapse: 'collapse' }}>
        <thead>
          <tr>
            {['n', 'format', 'fetch+decode ms', 'heap Δ MB', 'payload bytes'].map(h => (
              <th key={h} style={{ padding: '4px 8px', textAlign: 'left', borderBottom: '1px solid #ccc' }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, idx) => (
            <tr key={idx}>
              <td style={{ padding: '2px 8px' }}>{r.n}</td>
              <td style={{ padding: '2px 8px' }}>{r.format}</td>
              <td style={{ padding: '2px 8px' }}>{r.decodeMs.toFixed(1)}</td>
              <td style={{ padding: '2px 8px' }}>{r.heapDeltaMb?.toFixed(2) ?? '—'}</td>
              <td style={{ padding: '2px 8px' }}>{r.payloadBytes.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
```

- [ ] **Step 2: Add `'bench'` to the AppView union**

In `frontend/src/App.tsx`, line 67, change:
```ts
type AppView = 'risk' | 'covariance' | 'timeseries'
```
to:
```ts
type AppView = 'risk' | 'covariance' | 'timeseries' | 'bench'
```

- [ ] **Step 3: Add an import for the bench page**

Near the top of `App.tsx` with the other imports, add:
```ts
import { CovarianceBench } from './bench/CovarianceBench'
```

- [ ] **Step 4: Add a Bench tab**

Find the existing tab block around lines 1885-1893 (the `appView === 'covariance'` tab). Immediately after the `</button>` for the covariance tab, add:

```tsx
<button
  aria-selected={appView === 'bench'}
  className={appView === 'bench' ? 'active' : ''}
  onClick={() => setAppView('bench')}
>
  Bench
</button>
```

- [ ] **Step 5: Render the bench view**

Around line 2071 where the existing `appView === 'covariance' && (...)` block lives, add a sibling block:

```tsx
{appView === 'bench' && <CovarianceBench />}
```

- [ ] **Step 6: Smoke-test in browser**

Make sure both backend (with `FACTOR_RISK_BENCH=1`) and frontend (`npm run dev`) are running. Open the app, click the Bench tab, click Run. Expected: rows fill in over a few seconds; n=3600 might take ~30s for format A.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/bench/CovarianceBench.tsx frontend/src/App.tsx
git commit -m "bench: add /bench view scaffold with fetch + payload size measurement"
```

---

## Task 11: Real per-format decode-step measurement

The Task 10 scaffold lumps fetch + parse together. Now split out the parse step so we measure the format-specific decode cost properly.

**Files:**
- Modify: `frontend/src/bench/CovarianceBench.tsx`

- [ ] **Step 1: Add per-format decode functions**

In `CovarianceBench.tsx`, above the component definition, add:

```ts
import { tableFromIPC } from 'apache-arrow'

type Decoded = { ids: string[]; matrix: Float32Array; n: number }

async function decodeA(buf: ArrayBuffer): Promise<Decoded> {
  const body = JSON.parse(new TextDecoder().decode(buf))
  const ids: string[] = body.factor_ids
  const n = ids.length
  const matrix = new Float32Array(n * n)
  for (const cell of body.cells) matrix[cell.i * n + cell.j] = cell.v
  return { ids, matrix, n }
}

async function decodeB(buf: ArrayBuffer): Promise<Decoded> {
  const body = JSON.parse(new TextDecoder().decode(buf))
  const ids: string[] = body.factor_ids
  const n = ids.length
  const matrix = new Float32Array(n * n)
  for (let i = 0; i < n; i++) {
    const row = body.matrix[i]
    for (let j = 0; j < n; j++) matrix[i * n + j] = row[j]
  }
  return { ids, matrix, n }
}

async function decodeD(buf: ArrayBuffer): Promise<Decoded> {
  const table = tableFromIPC(new Uint8Array(buf))
  const meta = table.schema.metadata.get('factor_ids')!
  const ids: string[] = JSON.parse(meta)
  const n = ids.length
  // FixedSizeList<Float32> -> contiguous float32 buffer
  const valuesCol = table.getChild('values')!
  const data = valuesCol.data[0]
  // child Float32Array buffer covers n*n floats
  const child = data.children[0]
  const floats = child.values as Float32Array
  // copy into a plain Float32Array (matches other formats' shape)
  return { ids, matrix: new Float32Array(floats.buffer, floats.byteOffset, n * n), n }
}

async function decodeE(buf: ArrayBuffer): Promise<Decoded> {
  const view = new DataView(buf)
  const sidecarLen = view.getUint32(0, true)
  const sidecar = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, sidecarLen)))
  const ids: string[] = sidecar.factor_ids
  const n = view.getUint32(4 + sidecarLen, true)
  const bodyOffset = 4 + sidecarLen + 4
  return { ids, matrix: new Float32Array(buf, bodyOffset, n * n), n }
}

const DECODERS: Record<string, (buf: ArrayBuffer) => Promise<Decoded>> = {
  A: decodeA, B: decodeB, D: decodeD, E: decodeE,
}
```

- [ ] **Step 2: Update the run loop to measure decode separately**

In the `run` function, replace the fetch+decode block with:

```ts
const heapBefore = (performance as any).memory?.usedJSHeapSize ?? null
const buf = await fetch(url).then(r => r.arrayBuffer())
// Take median of 3 decode trials. Each trial re-decodes from the same buffer.
const samples: number[] = []
let decoded: Decoded | null = null
for (let trial = 0; trial < 3; trial++) {
  const t0 = performance.now()
  decoded = await DECODERS[fmt](buf)
  samples.push(performance.now() - t0)
}
samples.sort((a, b) => a - b)
const decodeMs = samples[1]
// keep `decoded` reachable so heap measurement reflects decoded form
;(window as any).__lastDecoded = decoded
const heapAfter = (performance as any).memory?.usedJSHeapSize ?? null
```

And update the column header from `'fetch+decode ms'` to `'decode ms (median 3)'`.

- [ ] **Step 3: Browser smoke test**

Reload the bench page, click Run. Expected: decode ms differs noticeably between formats; format A is slowest, E is fastest.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/bench/CovarianceBench.tsx
git commit -m "bench: per-format decode functions + median-of-3 timing"
```

---

## Task 12: AG Grid scroll FPS sub-test

**Files:**
- Modify: `frontend/src/bench/CovarianceBench.tsx`

The sub-test loads the n=3600 matrix once in format D and once in format B, mounts an `<AgGridReact>` with `valueGetter` reading the decoded matrix, and runs a programmatic horizontal scroll while counting frames.

- [ ] **Step 1: Add the sub-test UI + logic**

At the top of `CovarianceBench.tsx`, add:

```tsx
import { AgGridReact } from 'ag-grid-react'
import 'ag-grid-community/styles/ag-grid.css'
import 'ag-grid-community/styles/ag-theme-quartz.css'
import { ModuleRegistry, AllCommunityModule } from 'ag-grid-community'
import { useRef } from 'react'

ModuleRegistry.registerModules([AllCommunityModule])
```

Inside `CovarianceBench`, add a second state slice and a sub-test function:

```tsx
const [fpsResults, setFpsResults] = useState<{ format: string; fps: number }[]>([])
const [activeMatrix, setActiveMatrix] = useState<Decoded | null>(null)
const gridRef = useRef<AgGridReact>(null)

async function runFpsTest() {
  setFpsResults([])
  const out: { format: string; fps: number }[] = []
  for (const fmt of ['B', 'D'] as const) {
    const buf = await fetch(`${API}/api/_bench/covariance?format=${fmt}&n=3600`).then(r => r.arrayBuffer())
    const decoded = await DECODERS[fmt](buf)
    setActiveMatrix(decoded)
    // wait one frame for AG Grid to mount
    await new Promise(r => requestAnimationFrame(r))
    await new Promise(r => setTimeout(r, 200))
    const fps = await measureScrollFps(gridRef.current!, decoded.n)
    out.push({ format: fmt, fps })
    setFpsResults([...out])
  }
}

async function measureScrollFps(grid: AgGridReact, n: number): Promise<number> {
  const api = grid.api
  if (!api) return 0
  let frames = 0
  let stop = false
  const tick = () => {
    if (stop) return
    frames++
    requestAnimationFrame(tick)
  }
  requestAnimationFrame(tick)
  // pan across columns
  const start = performance.now()
  for (let c = 0; c < n; c += 10) {
    api.ensureColumnVisible(`c${c}`)
    await new Promise(r => setTimeout(r, 16))
    if (performance.now() - start > 4000) break
  }
  stop = true
  const elapsedSec = (performance.now() - start) / 1000
  return frames / elapsedSec
}
```

- [ ] **Step 2: Mount the AG Grid for the FPS test**

Add to the JSX returned from `CovarianceBench`, below the existing table:

```tsx
<h3 style={{ marginTop: 24 }}>AG Grid scroll FPS (n=3600)</h3>
<button onClick={runFpsTest}>Run FPS test</button>
<table style={{ marginTop: 8 }}>
  <thead>
    <tr><th style={{ padding: '4px 8px' }}>format</th><th style={{ padding: '4px 8px' }}>FPS</th></tr>
  </thead>
  <tbody>
    {fpsResults.map(r => (
      <tr key={r.format}>
        <td style={{ padding: '2px 8px' }}>{r.format}</td>
        <td style={{ padding: '2px 8px' }}>{r.fps.toFixed(1)}</td>
      </tr>
    ))}
  </tbody>
</table>
{activeMatrix && (
  <div className="ag-theme-quartz" style={{ height: 400, width: '100%', marginTop: 12 }}>
    <AgGridReact
      ref={gridRef}
      rowData={Array.from({ length: activeMatrix.n }, (_, i) => ({ __row: i }))}
      columnDefs={Array.from({ length: activeMatrix.n }, (_, j) => ({
        colId: `c${j}`,
        headerName: String(j),
        width: 64,
        valueGetter: (p: any) => {
          const i = p.data.__row
          return activeMatrix.matrix[i * activeMatrix.n + j]
        },
      }))}
    />
  </div>
)}
```

- [ ] **Step 3: Smoke-test the FPS sub-test**

In the browser, click "Run FPS test". Expected: a 3600-column AG Grid mounts, the column scroller pans from left to right over a few seconds, and one FPS number per format is printed. The two numbers should be in the same ballpark (both formats produce identical Float32Array under the hood, so the cell-access cost is the same — the difference, if any, comes from initial mount cost and one-time decode behaviour).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/bench/CovarianceBench.tsx
git commit -m "bench: AG Grid scroll FPS sub-test for formats B and D at n=3600"
```

---

## Task 13: Run benchmarks and collect results

This task produces no code — it executes the harness and stashes the JSON outputs.

**Files:**
- Create (output): `backend/bench/results/server_bench.json`
- Create (output): `backend/bench/results/pandas_bench.json`
- Create (output): `backend/bench/results/frontend_bench.json`
- Create (output): `backend/bench/results/fps.json` (manually transcribed from the UI table)

- [ ] **Step 1: Run server bench**

```bash
cd backend && uv run python -m bench.server_bench
```
Expected: writes `bench/results/server_bench.json` with 12 rows.

- [ ] **Step 2: Run pandas bench**

```bash
cd backend && uv run python -m bench.pandas_bench
```
Expected: writes `bench/results/pandas_bench.json` with 12 rows.

- [ ] **Step 3: Run frontend bench**

In a Chrome tab on the running app, navigate to the Bench tab and click Run. When done, click "Download JSON" and save the file as `backend/bench/results/frontend_bench.json`.

Expected: 12 rows. Watch out for n=3600 + format A — it may take 20-30s and allocate hundreds of MB. If the tab crashes, that is a finding; record it as `decodeMs: null, note: "OOM"`.

- [ ] **Step 4: Run AG Grid FPS sub-test**

Click "Run FPS test" on the same page. Manually copy the two FPS numbers into a JSON file:

```bash
cat > backend/bench/results/fps.json <<'EOF'
[
  {"format": "B", "fps": <number>},
  {"format": "D", "fps": <number>}
]
EOF
```

- [ ] **Step 5: Commit results**

```bash
git add backend/bench/results/
git commit -m "bench: collected results from all four benchmark runs"
```

---

## Task 14: Write the report

**Files:**
- Create: `docs/experiments/2026-04-26-covariance-payload-formats.md`

- [ ] **Step 1: Read all four results files**

Inspect:
```bash
cat backend/bench/results/server_bench.json
cat backend/bench/results/pandas_bench.json
cat backend/bench/results/frontend_bench.json
cat backend/bench/results/fps.json
```

- [ ] **Step 2: Write the report**

Create `docs/experiments/2026-04-26-covariance-payload-formats.md` with this exact section structure (fill values from the JSON):

```markdown
# Covariance Payload Formats — Results

Date: 2026-04-26
Spec: `docs/superpowers/specs/2026-04-26-covariance-payload-formats-experiment-design.md`

## Setup

One paragraph: machine, browser, dataset shape (synthetic random symmetric float32, deterministic seed per size), trials = 3, median reported, gzip level 6.

## Wire size (raw / gzipped, MB)

| Format | n=200 | n=1000 | n=3600 |
|--------|-------|--------|--------|
| A — list-of-dict | _raw_ / _gz_ | _raw_ / _gz_ | _raw_ / _gz_ |
| B — 2D array     | _raw_ / _gz_ | _raw_ / _gz_ | _raw_ / _gz_ |
| D — Arrow IPC    | _raw_ / _gz_ | _raw_ / _gz_ | _raw_ / _gz_ |
| E — Float32+JSON | _raw_ / _gz_ | _raw_ / _gz_ | _raw_ / _gz_ |

## Server encode time (ms, median of 3)

(same shape table)

## Client decode time + heap delta

| Format | n=200 ms / MB | n=1000 ms / MB | n=3600 ms / MB |
|--------|--------------|----------------|----------------|

## AG Grid scroll FPS at n=3600

| Format | FPS |
|--------|-----|
| B baseline | _x_ |
| D winner   | _x_ |

## Pandas conversion (ms)

(same shape table)

Inline note: LOC of each canonical converter (from `bench/pandas_bench.py`):
- A: ~5 lines
- B: ~3 lines
- D: ~5 lines (3 if you accept `pa.ipc.open_stream(buf).read_pandas()`)
- E: ~6 lines (manual struct.unpack)

## Findings

(4-6 bullets — must touch each of these themes:)
- JS heap blowup of list-of-dict at the largest size, vs flat typed-array footprint
- JSON.parse blocks the main thread, none of the JSON variants stream
- Transferable / Web Worker boundary: only typed arrays + Arrow buffers transfer zero-copy
- AG Grid valueGetter per-cell access cost: typed-array index << object property lookup << Arrow vector accessor with `.get()`
- Bundle delta of `apache-arrow` (measure with `npm run build` before/after if not already known)
- Debug ergonomics asymmetry: Network-tab grep vs needing a binary viewer

## Recommendation

One paragraph: which format for `/api/covariance/subset` going forward, which for bulk download, whether the existing 2D-array choice should change. Justify against the numbers above.
```

- [ ] **Step 3: Commit report**

```bash
git add docs/experiments/2026-04-26-covariance-payload-formats.md
git commit -m "bench: covariance payload formats experiment report"
```

---

## Self-review notes

Spec coverage: all 8 spec sections (Goal, Scope, Correctness gate, Code organisation, Deliverable shape, Open questions, Format choice for AG Grid wire-in) map to tasks. Round-trip correctness gate is enforced in tasks 2-5 via `test_formats.py`. AG Grid wire-in uses format D as specified; if results show D > 2× worse than E, the report Recommendation section calls this out (no plan changes needed — it's a documentation outcome).

Dropped from spec scope (acknowledged in spec text, no task): brotli, MessagePack/CBOR/NDJSON, range requests, ETag, cross-browser, server-side memory.

Synthesis vs real data: spec recommended synthetic random matrices for stability; plan uses `np.random.default_rng(n)` with size as seed for deterministic per-size payloads.
