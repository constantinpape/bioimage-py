"""Correctness tests for the morphology op, validated against skimage.measure.regionprops."""
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter
from skimage.measure import label as sklabel
from skimage.measure import regionprops

import bioimage_py as bp


def _make_segmentation(shape, rng):
    """Build a labeled volume of irregular blobs (smoothed noise -> threshold -> connected components)."""
    field = gaussian_filter(rng.random(shape), sigma=2.0)
    binary = field > field.mean() + field.std()
    return sklabel(binary).astype("uint64")


def _check_against_regionprops(seg):
    """Assert morphology(seg) matches regionprops on every label (size, com, slice-ready bbox)."""
    ndim = seg.ndim
    axes = ["y", "x"] if ndim == 2 else ["z", "y", "x"]
    df = bp.morphology.morphology(seg).set_index("label")
    props = regionprops(seg.astype("int64"))

    assert 0 not in df.index.to_numpy()
    assert len(df) == len(props)
    for rp in props:
        row = df.loc[rp.label]
        assert int(row["size"]) == rp.area
        com = np.array([row[f"com_{a}"] for a in axes])
        assert np.allclose(com, rp.centroid)
        bb_min = np.array([int(row[f"bb_min_{a}"]) for a in axes])
        bb_max = np.array([int(row[f"bb_max_{a}"]) for a in axes])
        assert np.array_equal(bb_min, [s.start for s in rp.slice])
        assert np.array_equal(bb_max, [s.stop for s in rp.slice])  # our bb_max is the exclusive stop
        # The bounding box is slice-ready: slicing it back out recovers exactly the object.
        box = seg[tuple(slice(lo, hi) for lo, hi in zip(bb_min, bb_max))]
        assert int((box == rp.label).sum()) == rp.area


def test_morphology_3d(rng):
    _check_against_regionprops(_make_segmentation((30, 40, 50), rng))


def test_morphology_2d(rng):
    _check_against_regionprops(_make_segmentation((64, 80), rng))


def test_morphology_blocked_matches_direct(zarr_factory, rng):
    seg = _make_segmentation((40, 48), rng)
    z = zarr_factory(seg, chunks=(16, 16))
    direct = bp.morphology.morphology(seg)
    blocked = bp.morphology.morphology(z, num_workers=4, block_shape=(16, 16))
    pd.testing.assert_frame_equal(direct, blocked)


def test_morphology_mask(rng):
    seg = _make_segmentation((40, 48), rng)
    mask = np.zeros(seg.shape, dtype=bool)
    mask[:20] = True
    df = bp.morphology.morphology(seg, num_workers=2, block_shape=(16, 16), mask=mask).set_index("label")
    # Compare to regionprops on the masked segmentation (out-of-mask voxels become background).
    props = regionprops(np.where(mask, seg, 0).astype("int64"))
    assert len(df) == len(props)
    for rp in props:
        assert int(df.loc[rp.label, "size"]) == rp.area


def test_morphology_empty():
    df = bp.morphology.morphology(np.zeros((10, 12), dtype="uint64"))
    assert len(df) == 0
    assert list(df.columns) == [
        "label", "size", "com_y", "com_x", "bb_min_y", "bb_min_x", "bb_max_y", "bb_max_x"
    ]


def test_morphology_requires_integer():
    with pytest.raises(ValueError, match="integer"):
        bp.morphology.morphology(np.zeros((5, 5), dtype="float32"))
