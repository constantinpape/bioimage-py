"""Tests for the generic per-item Runner.map across local and subprocess backends."""
import numpy as np
import pytest

import bioimage_py as bp
from bioimage_py.runner import RunnerError
from bioimage_py.sources import as_source
from bioimage_py.sources.dispatch import from_spec


def _square(i):
    """Top-level (picklable) per-item function."""
    return i * i


def _make_reader(spec, offset):
    """Return a closure that reopens a source by spec and reads index i (+ offset)."""
    def read(i):
        src = from_spec(spec)
        return int(np.asarray(src[(slice(i, i + 1),)])[0]) + offset
    return read


def _make_boom(bad):
    """Return a function that raises for one index."""
    def fn(i):
        if i == bad:
            raise ValueError(f"boom at {i}")
        return i
    return fn


def test_map_local_orders_results():
    runner = bp.get_runner("local")
    assert runner.map(_square, 6, num_workers=3) == [0, 1, 4, 9, 16, 25]
    # explicit item_ids are honored and results follow their order
    assert runner.map(_square, item_ids=[5, 2, 0], num_workers=2) == [25, 4, 0]


def test_map_parity_local_subprocess(zarr_factory):
    a = np.arange(12, dtype="int64")
    z = zarr_factory(a, chunks=(5,))
    spec = as_source(z).to_spec()
    expected = [int(a[i]) + 100 for i in range(12)]
    for nw, job in [(1, "local"), (4, "local"), (3, "subprocess")]:
        got = bp.get_runner(job).map(_make_reader(spec, 100), 12, num_workers=nw, name="map")
        assert got == expected, f"mismatch for nw={nw} job={job}: {got}"


def test_map_requires_n_or_ids():
    with pytest.raises(ValueError, match="n_items or item_ids"):
        bp.get_runner("local").map(_square)


def test_map_failure_reports_id():
    with pytest.raises(RunnerError) as excinfo:
        bp.get_runner("local").map(_make_boom(3), 6, num_workers=2)
    assert 3 in excinfo.value.failed_block_ids
