"""Covariance payload format encoders/decoders for the benchmark harness.

Each format implements `encode(ids, matrix) -> (bytes, content_type)` and
`decode(bytes) -> (ids, matrix)`. The decoder exists only for round-trip
correctness testing — production code never calls it.
"""
from __future__ import annotations

import io
import json
import struct
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


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


# ---------- Format B: 2D nested array JSON -----------------------------------

def _encode_b(ids: list[str], matrix: np.ndarray) -> tuple[bytes, str]:
    body = {"factor_ids": ids, "matrix": matrix.tolist()}
    return json.dumps(body).encode("utf-8"), "application/json"


def _decode_b(payload: bytes) -> tuple[list[str], np.ndarray]:
    body = json.loads(payload)
    return body["factor_ids"], np.asarray(body["matrix"], dtype=np.float32)


# ---------- Format D: Apache Arrow IPC stream --------------------------------
# Schema: factor_ids in stream metadata; one record batch with
# FixedSizeList<float32>[n] per matrix row.

def _encode_d(ids: list[str], matrix: np.ndarray) -> tuple[bytes, str]:
    n = len(ids)
    flat = matrix.astype(np.float32, copy=False).reshape(-1)
    values_array = pa.FixedSizeListArray.from_arrays(pa.array(flat), n)
    idx_array = pa.array(np.arange(n, dtype=np.int32))
    schema = pa.schema(
        [("row_idx", pa.int32()), ("values", pa.list_(pa.float32(), n))],
        metadata={b"factor_ids": json.dumps(ids).encode("utf-8")},
    )
    batch = pa.record_batch([idx_array, values_array], schema=schema)
    sink = io.BytesIO()
    with ipc.new_stream(sink, schema) as writer:
        writer.write_batch(batch)
    return sink.getvalue(), "application/vnd.apache.arrow.stream"


def _decode_d(payload: bytes) -> tuple[list[str], np.ndarray]:
    reader = ipc.open_stream(io.BytesIO(payload))
    table = reader.read_all()
    ids = json.loads(reader.schema.metadata[b"factor_ids"])
    n = len(ids)
    flat = table.column("values").combine_chunks().flatten().to_numpy(zero_copy_only=False)
    matrix = flat.reshape(n, n).astype(np.float32)
    return ids, matrix


# ---------- Format E: raw Float32Array + JSON sidecar ------------------------
# Wire layout (all little-endian):
#   u32 sidecar_len
#   sidecar_len bytes: utf-8 JSON {"factor_ids": [...]}
#   u32 n  (matrix side length, sanity check)
#   n*n*4 bytes: row-major float32 matrix

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


REGISTRY: dict[str, FormatSpec] = {
    "A": FormatSpec(
        name="list-of-dict JSON",
        description="Row-shaped JSON: [{i, j, v}, ...]",
        encode=_encode_a,
        decode=_decode_a,
    ),
    "B": FormatSpec(
        name="2D nested array JSON",
        description="{factor_ids, matrix: [[...]]}",
        encode=_encode_b,
        decode=_decode_b,
    ),
    "D": FormatSpec(
        name="Apache Arrow IPC stream",
        description="Binary columnar; ids in schema metadata; FixedSizeList<float32> rows",
        encode=_encode_d,
        decode=_decode_d,
    ),
    "E": FormatSpec(
        name="Float32Array + JSON sidecar",
        description="Lower-bound binary: u32 sidecar_len, JSON {ids}, u32 n, row-major float32",
        encode=_encode_e,
        decode=_decode_e,
    ),
}
