"""Shared pytest fixtures."""
import os
import shutil
import uuid

import numpy as np
import pytest
import zarr

# Root for shared-filesystem test data, used by the slurm tests. Must be visible to compute
# nodes (node-local /tmp / pytest's tmp_path are not), so it is opted into via an env var.
_SHARED_ROOT = os.environ.get("BIOIMAGE_PY_SHARED_TMP")


def _write_zarr(path, array=None, chunks=None, *, shape=None, dtype=None, fill=None):
    """Create a fresh on-disk zarr array, optionally filled from ``array`` or ``fill``."""
    if array is not None:
        shape, dtype = array.shape, array.dtype
    z = zarr.open_array(path, mode="w", shape=shape, chunks=tuple(chunks), dtype=dtype)
    if array is not None:
        z[:] = array
    elif fill is not None:
        z[:] = fill
    return z


@pytest.fixture
def rng():
    """A seeded random generator for reproducible test data."""
    return np.random.default_rng(42)


@pytest.fixture
def shared_tmp_path():
    """A per-test directory on the shared filesystem (the shared-FS analogue of tmp_path)."""
    base = os.path.join(_SHARED_ROOT, f"bp_test_{uuid.uuid4().hex[:12]}")
    os.makedirs(base, exist_ok=True)
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


@pytest.fixture
def shared_zarr_factory(shared_tmp_path):
    """Like :func:`zarr_factory` but writes the arrays under the shared filesystem."""
    counter = {"i": 0}

    def _make(array=None, chunks=None, *, shape=None, dtype=None, fill=None):
        counter["i"] += 1
        path = os.path.join(shared_tmp_path, f"arr_{counter['i']}.zarr")
        return _write_zarr(path, array, chunks, shape=shape, dtype=dtype, fill=fill)

    return _make


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
        return _write_zarr(path, array, chunks, shape=shape, dtype=dtype, fill=fill)

    return _make
