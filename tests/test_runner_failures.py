"""Failure handling: RunnerError, preserved temp folder, and block_ids re-run."""
import os

import pytest

from bioimage_py.runner import RunnerError, get_runner


def _make_flaky(marker_path):
    """Per-block function that fails the first-processed corner block exactly once.

    The failure is recorded via a marker file so that re-running the failed block
    succeeds (simulating a transient failure that re-run fixes).
    """
    def fn(block, inputs, outputs, mask):
        is_corner = all(int(b) == 0 for b in block.begin)
        if is_corner and not os.path.exists(marker_path):
            with open(marker_path, "w") as f:
                f.write("failed once")
            raise RuntimeError("transient boom")
        return int(block.begin[0])

    return fn


@pytest.mark.parametrize("job_type", ["local", "subprocess"])
def test_failure_then_rerun(job_type, zarr_factory, rng, tmp_path):
    a = rng.random((32, 32)).astype("float32")
    z = zarr_factory(a, chunks=(16, 16))
    marker = str(tmp_path / "marker.txt")
    fn = _make_flaky(marker)
    runner = get_runner(job_type)

    with pytest.raises(RunnerError) as excinfo:
        runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True, name="flaky")
    err = excinfo.value
    assert err.failed_block_ids, "failed block ids should be reported"

    if job_type == "subprocess":
        assert err.tmp_folder is not None and os.path.isdir(err.tmp_folder)
        assert os.path.exists(os.path.join(err.tmp_folder, "source.py"))

    # Re-running the reported failed blocks now succeeds (marker exists).
    results = runner.run(fn, [z], block_shape=(16, 16), num_workers=4, has_return_val=True,
                         block_ids=err.failed_block_ids, name="flaky-rerun")
    assert all(r is not None for r in results)
