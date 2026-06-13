"""Tests for block-wise connected-component labeling."""
import numpy as np
import pytest

import bioimage_cpp as bic
import bioimage_py as bp


def assert_same_partition(a, b):
    """Assert two label arrays induce the same partition (equal up to relabeling)."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    assert np.array_equal(a == 0, b == 0), "background mismatch"
    fg = a != 0
    pairs = np.unique(np.stack([a[fg], b[fg]], axis=1), axis=0)
    assert len(np.unique(pairs[:, 0])) == len(pairs), "a-label split across b-labels"
    assert len(np.unique(pairs[:, 1])) == len(pairs), "b-label split across a-labels"


@pytest.mark.parametrize("shape,block_shape", [
    ((37, 41), (16, 16)),
    ((64, 64), (13, 17)),     # non-divisible
    ((20, 23, 19), (8, 8, 8)),
])
@pytest.mark.parametrize("nw,job_type", [(1, "local"), (4, "local"), (3, "subprocess")])
def test_label_parity(shape, block_shape, nw, job_type, zarr_factory, rng):
    binary = (rng.random(shape) > 0.6).astype("uint8")
    ref = bic.segmentation.label(binary.astype(bool), connectivity=1)

    zb = zarr_factory(binary, chunks=block_shape)
    out = zarr_factory(shape=shape, chunks=block_shape, dtype="uint64", fill=0)
    bp.segmentation.label(zb, out, block_shape=block_shape, num_workers=nw, job_type=job_type)
    result = out[:]

    assert_same_partition(result, ref)
    # Labels must be compact: max == number of components == reference component count.
    n = int(result.max())
    assert n == len(np.unique(result[result != 0])) == int(ref.max())


def test_label_threshold(zarr_factory, rng):
    a = rng.random((40, 40)).astype("float32")
    ref = bic.segmentation.label(a > 0.5, connectivity=1)
    z = zarr_factory(a, chunks=(16, 16))
    out = zarr_factory(shape=a.shape, chunks=(16, 16), dtype="uint64", fill=0)
    bp.segmentation.label(z, out, threshold=0.5, block_shape=(16, 16), num_workers=4)
    assert_same_partition(out[:], ref)


def test_label_mask(zarr_factory, rng):
    binary = (rng.random((40, 40)) > 0.4).astype("uint8")
    mask = np.zeros((40, 40), dtype="uint8")
    mask[8:32, 8:32] = 1
    ref = bic.segmentation.label((binary.astype(bool) & mask.astype(bool)), connectivity=1)
    zb = zarr_factory(binary, chunks=(16, 16))
    out = zarr_factory(shape=binary.shape, chunks=(16, 16), dtype="uint64", fill=0)
    bp.segmentation.label(zb, out, block_shape=(16, 16), num_workers=4, mask=mask)
    result = out[:]
    assert np.all(result[~mask.astype(bool)] == 0)
    assert_same_partition(result, ref)


def test_high_connectivity_blockwise_raises(zarr_factory, rng):
    zb = zarr_factory((rng.random((32, 32)) > 0.5).astype("uint8"), chunks=(16, 16))
    out = zarr_factory(shape=(32, 32), chunks=(16, 16), dtype="uint64", fill=0)
    with pytest.raises(NotImplementedError, match="connectivity=1"):
        bp.segmentation.label(zb, out, connectivity=2, block_shape=(16, 16), num_workers=2)


def test_direct_full_connectivity(rng):
    binary = (rng.random((30, 30)) > 0.5).astype("uint8")
    ref = bic.segmentation.label(binary.astype(bool), connectivity=2)
    out = np.zeros((30, 30), dtype="uint64")
    bp.segmentation.label(binary, out, connectivity=2)
    assert_same_partition(out, ref)


def test_output_optional_local(zarr_factory, rng):
    binary = (rng.random((40, 40)) > 0.6).astype("uint8")
    ref = bic.segmentation.label(binary.astype(bool), connectivity=1)
    zb = zarr_factory(binary, chunks=(16, 16))
    # No output -> a uint64 numpy array is allocated and returned.
    result = bp.segmentation.label(zb, block_shape=(16, 16), num_workers=4)
    assert isinstance(result, np.ndarray) and result.dtype == np.dtype("uint64")
    assert_same_partition(result, ref)


def test_output_required_distributed(zarr_factory, rng):
    zb = zarr_factory((rng.random((32, 32)) > 0.5).astype("uint8"), chunks=(16, 16))
    with pytest.raises(ValueError, match="required for distributed execution"):
        bp.segmentation.label(zb, block_shape=(16, 16), num_workers=2, job_type="subprocess")


def test_output_dtype_validated(rng):
    binary = (rng.random((16, 16)) > 0.5).astype("uint8")
    out = np.zeros((16, 16), dtype="uint32")
    with pytest.raises(ValueError, match="uint64"):
        bp.segmentation.label(binary, out, block_shape=(8, 8), num_workers=2)


@pytest.mark.parametrize("fill", [0.0, 1.0])
def test_label_empty_and_full(zarr_factory, fill):
    binary = np.full((24, 24), fill, dtype="uint8")
    ref = bic.segmentation.label(binary.astype(bool), connectivity=1)
    zb = zarr_factory(binary, chunks=(8, 8))
    out = zarr_factory(shape=(24, 24), chunks=(8, 8), dtype="uint64", fill=0)
    bp.segmentation.label(zb, out, block_shape=(8, 8), num_workers=4)
    assert_same_partition(out[:], ref)
