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


@pytest.mark.parametrize("fmt", ["A", "B"])
def test_roundtrip(fmt: str):
    ids, matrix = make_input(20)
    payload, content_type = formats.REGISTRY[fmt].encode(ids, matrix)
    assert isinstance(payload, (bytes, bytearray))
    assert isinstance(content_type, str) and content_type
    decoded_ids, decoded_matrix = formats.REGISTRY[fmt].decode(payload)
    assert decoded_ids == ids
    np.testing.assert_allclose(decoded_matrix, matrix, rtol=0, atol=1e-6)
