"""Headline parity tests: direct == local(1) == local(N) == subprocess(N)."""
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp


def test_max_parity(zarr_factory, rng):
    a = rng.random((37, 41)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    expected = float(a.max())

    assert np.isclose(bp.stats.max(a), expected)  # direct
    assert np.isclose(bp.stats.max(z, num_workers=1, block_shape=(16, 16)), expected)
    assert np.isclose(bp.stats.max(z, num_workers=4, block_shape=(16, 16)), expected)
    assert np.isclose(
        bp.stats.max(z, num_workers=3, block_shape=(16, 16), job_type="subprocess"), expected
    )


def test_max_numpy_subprocess_raises(rng):
    a = rng.random((20, 20)).astype("float32")
    with pytest.raises(ValueError, match="numpy"):
        bp.stats.max(a, num_workers=2, block_shape=(8, 8), job_type="subprocess")


def test_gaussian_parity(zarr_factory, rng):
    a = rng.random((40, 48)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    ref = bic.filters.gaussian_smoothing(a, 2.0)

    # direct (in place on a copy via numpy source)
    direct = bp.filters.gaussian_smoothing(a.copy(), 2.0)
    np.testing.assert_allclose(direct[(slice(None), slice(None))], ref, atol=1e-4)

    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="float32", fill=0.0)
        bp.filters.gaussian_smoothing(z, 2.0, output=out, block_shape=(16, 16),
                                      num_workers=nw, job_type=job)
        np.testing.assert_allclose(out[:], ref, atol=1e-4,
                                   err_msg=f"mismatch for nw={nw} job={job}")
