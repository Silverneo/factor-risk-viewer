import numpy as np
import pytest

from bench import formats
from bench.server_bench import synth


@pytest.mark.parametrize("fmt", list(formats.REGISTRY.keys()))
def test_roundtrip(fmt: str):
    ids, matrix = synth(20)
    payload, content_type = formats.REGISTRY[fmt].encode(ids, matrix)
    assert isinstance(payload, (bytes, bytearray))
    assert isinstance(content_type, str) and content_type
    decoded_ids, decoded_matrix = formats.REGISTRY[fmt].decode(payload)
    assert decoded_ids == ids
    np.testing.assert_allclose(decoded_matrix, matrix, rtol=0, atol=1e-6)
