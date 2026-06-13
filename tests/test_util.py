"""Tests for blocking and filter-halo helpers."""
import numpy as np
import pytest

from bioimage_py.sources import as_source
from bioimage_py.util import derive_block_shape, get_blocking, normalize_halo, sigma_to_halo, to_roi


def test_to_roi():
    blocking = get_blocking((10, 7), (4, 3))
    block = blocking.get_block(0)
    assert to_roi(block) == (slice(0, 4), slice(0, 3))


def test_get_blocking_counts():
    blocking = get_blocking((10, 7), (4, 3))
    assert int(blocking.number_of_blocks) == 9


def test_get_blocking_with_roi():
    blocking = get_blocking((20, 20), (5, 5), roi=(slice(0, 10), slice(0, 10)))
    assert int(blocking.number_of_blocks) == 4


def test_derive_block_shape_explicit(rng):
    src = as_source(rng.random((8, 8)))
    assert derive_block_shape(src, (4, 4)) == (4, 4)


def test_derive_block_shape_from_chunks(zarr_factory, rng):
    z = zarr_factory(rng.random((8, 8)).astype("float32"), chunks=(4, 2))
    assert derive_block_shape(as_source(z), None) == (4, 2)


def test_derive_block_shape_unchunked_raises(rng):
    src = as_source(rng.random((8, 8)))
    with pytest.raises(ValueError, match="block_shape is required"):
        derive_block_shape(src, None)


def test_sigma_to_halo():
    assert sigma_to_halo(2.0, 0) == 2 * int(np.ceil(3.0 * 2.0 + 0.5))
    assert sigma_to_halo([1.0, 2.0], 1) == [2 * int(np.ceil(3.0 * s + 0.5 * 1 + 0.5)) for s in (1.0, 2.0)]


def test_normalize_halo():
    assert normalize_halo(3, 2) == [3, 3]
    assert normalize_halo([1, 2], 2) == [1, 2]
    with pytest.raises(ValueError):
        normalize_halo([1, 2, 3], 2)
