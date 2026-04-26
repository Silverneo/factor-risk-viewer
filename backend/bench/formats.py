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


# ---------- Format B: 2D nested array JSON -----------------------------------

def _encode_b(ids: list[str], matrix: np.ndarray) -> tuple[bytes, str]:
    body = {"factor_ids": ids, "matrix": matrix.tolist()}
    return json.dumps(body).encode("utf-8"), "application/json"


def _decode_b(payload: bytes) -> tuple[list[str], np.ndarray]:
    body = json.loads(payload)
    return body["factor_ids"], np.asarray(body["matrix"], dtype=np.float32)


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
}
