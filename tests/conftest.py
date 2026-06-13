"""Shared pytest fixtures."""
import numpy as np
import pytest
import zarr


@pytest.fixture
def rng():
    """A seeded random generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def zarr_factory(tmp_path):
    """Return a factory that writes arrays to fresh on-disk zarr arrays.

    Usage: ``z = zarr_factory(array, chunks)`` or ``z = zarr_factory(shape=..., chunks=...,
    dtype=..., fill=...)`` for an empty (optionally filled) output array.
    """
    counter = {"i": 0}

    def _make(array=None, chunks=None, *, shape=None, dtype=None, fill=None):
        counter["i"] += 1
        path = str(tmp_path / f"arr_{counter['i']}.zarr")
        if array is not None:
            shape, dtype = array.shape, array.dtype
        z = zarr.open_array(path, mode="w", shape=shape, chunks=tuple(chunks), dtype=dtype)
        if array is not None:
            z[:] = array
        elif fill is not None:
            z[:] = fill
        return z

    return _make
