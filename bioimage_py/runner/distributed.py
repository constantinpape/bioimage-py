"""Distributed runners: a shared protocol base, the subprocess runner, and a slurm stub.

The protocol (cloudpickled payload + generated per-task work lists + result/sentinel files)
is shared so that :class:`SubprocessRunner` (here) and the future ``SlurmRunner`` differ
only in how tasks are launched and awaited.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent import futures
from typing import Any, List, Optional, Sequence, Tuple

import cloudpickle
from bioimage_cpp.utils import Blocking
from tqdm import tqdm

from ..sources.base import Source
from ..util import ComputeFn
from .base import Runner, RunnerError


def _partition(block_ids: Sequence[int], n_tasks: int) -> List[List[int]]:
    """Split ``block_ids`` into ``n_tasks`` contiguous, near-equal groups."""
    block_ids = list(block_ids)
    n = len(block_ids)
    base, extra = divmod(n, n_tasks)
    tasks, start = [], 0
    for t in range(n_tasks):
        size = base + (1 if t < extra else 0)
        tasks.append(block_ids[start:start + size])
        start += size
    return tasks


class _DistributedRunner(Runner):
    """Base for runners that ship the computation to separate worker processes."""

    @staticmethod
    def _require_reopenable(inputs: Sequence[Source], outputs: Sequence[Source],
                            mask: Optional[Source]) -> None:
        """Validate that every source is file-backed (reopenable on a worker).

        Args:
            inputs: The input sources.
            outputs: The output sources.
            mask: The mask source, or ``None``.

        Raises:
            ValueError: If any source cannot be reopened (e.g. an in-memory numpy array).
        """
        roles = [("input", inputs), ("output", outputs)]
        if mask is not None:
            roles.append(("mask", [mask]))
        for role, sources in roles:
            for source in sources:
                try:
                    source.to_spec()
                except ValueError as error:
                    raise ValueError(
                        f"Distributed execution requires file-backed {role} arrays (zarr/n5). {error}"
                    ) from error

    def _execute(
        self,
        *,
        function: ComputeFn,
        inputs: Sequence[Source],
        outputs: Sequence[Source],
        mask: Optional[Source],
        blocking: Blocking,
        block_ids: Sequence[int],
        halo: Optional[Sequence[int]],
        has_return_val: bool,
        num_workers: int,
        name: str,
        shape: Tuple[int, ...],
        block_shape: Tuple[int, ...],
        roi: Optional[Tuple[slice, ...]],
    ) -> List[Any]:
        # Validate up front that every source can be reopened on a worker (file-backed).
        self._require_reopenable(inputs, outputs, mask)
        tmp = tempfile.mkdtemp(prefix="bioimage_py_", dir=self.config.tmp_root)
        for sub in ("blocks", "results", "success", "error"):
            os.makedirs(os.path.join(tmp, sub), exist_ok=True)

        # Build and write the cloudpickle payload. to_spec() raises here for numpy inputs,
        # which is the actionable "numpy is local-only" failure for distributed backends.
        payload = {
            "function": function,
            "input_specs": [s.to_spec() for s in inputs],
            "output_specs": [s.to_spec() for s in outputs],
            "mask_spec": mask.to_spec() if mask is not None else None,
            "shape": tuple(shape),
            "block_shape": tuple(block_shape),
            "roi": roi,
            "halo": None if halo is None else [int(h) for h in halo],
            "has_return_val": bool(has_return_val),
            "python": tuple(sys.version_info[:2]),
        }
        with open(os.path.join(tmp, "payload.pkl"), "wb") as f:
            cloudpickle.dump(payload, f)

        # Human-readable debug artifact (never used for correctness).
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            source = f"# source unavailable for {getattr(function, '__name__', function)!r}\n"
        with open(os.path.join(tmp, "source.py"), "w") as f:
            f.write(source)

        n_tasks = max(1, min(int(num_workers), len(block_ids)))
        tasks = _partition(block_ids, n_tasks)
        for task_id, ids in enumerate(tasks):
            with open(os.path.join(tmp, "blocks", f"{task_id}.json"), "w") as f:
                json.dump([int(b) for b in ids], f)

        self._launch_and_wait(tmp, n_tasks, num_workers, name)

        # Ground truth for success is the per-task sentinel file, not the launcher's status.
        failed_tasks = [t for t in range(n_tasks)
                        if not os.path.exists(os.path.join(tmp, "success", f"{t}.success"))]
        if failed_tasks:
            failed_block_ids = sorted(int(b) for t in failed_tasks for b in tasks[t])
            raise RunnerError(self._failure_message(tmp, failed_tasks, name),
                              failed_block_ids=failed_block_ids, tmp_folder=tmp)

        results = self._collect(tmp, n_tasks, block_ids) if has_return_val else [None] * len(block_ids)
        shutil.rmtree(tmp, ignore_errors=True)
        return results

    @staticmethod
    def _collect(tmp: str, n_tasks: int, block_ids: Sequence[int]) -> List[Any]:
        """Load and order the per-task results by block id."""
        result_by_block = {}
        for task_id in range(n_tasks):
            with open(os.path.join(tmp, "results", f"{task_id}.pkl"), "rb") as f:
                for bid, res in cloudpickle.load(f):
                    result_by_block[int(bid)] = res
        return [result_by_block[int(b)] for b in block_ids]

    @staticmethod
    def _failure_message(tmp: str, failed_tasks: Sequence[int], name: str) -> str:
        """Build an error message naming the preserved temp folder and first error."""
        first = None
        err_path = os.path.join(tmp, "error", f"{failed_tasks[0]}.txt")
        if os.path.exists(err_path):
            with open(err_path) as f:
                first = f.read().strip().splitlines()[-1] if f else None
        return (
            f"{len(failed_tasks)} task(s) failed in '{name or 'run'}'. "
            f"Temp folder preserved for debugging: {tmp}. First error: {first!r}"
        )

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str) -> None:
        """Launch the worker tasks and block until they have all finished."""
        raise NotImplementedError


class SubprocessRunner(_DistributedRunner):
    """Distributed runner that launches each task as a local subprocess.

    Exercises the full distributed protocol (cloudpickle payload, generated harness,
    result/sentinel files, ``block_ids`` re-run) without a scheduler.
    """

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str) -> None:
        """Run each task as a local subprocess, up to ``num_workers`` concurrently."""
        python = self.config.python_executable or sys.executable
        cmd_base = [python, "-m", "bioimage_py.runner._harness", tmp]

        def _run_task(task_id: int):
            # Output is discarded; failures are reported via the harness's error file.
            return subprocess.run(cmd_base + [str(task_id)], capture_output=True, text=True)

        with futures.ThreadPoolExecutor(max(1, int(num_workers))) as tp:
            list(tqdm(tp.map(_run_task, range(n_tasks)), total=n_tasks,
                      desc=name or None, disable=not name))


class SlurmRunner(_DistributedRunner):
    """Stub for the slurm backend. To be implemented next session on the cluster."""

    def run(self, *args, **kwargs):  # type: ignore[override]
        """Not implemented yet; the slurm backend lands next session."""
        raise NotImplementedError("The slurm backend is not implemented yet (next session).")

    def _launch_and_wait(self, tmp: str, n_tasks: int, num_workers: int, name: str) -> None:
        # Next session: render an sbatch array script (one task per array index) using
        # SlurmConfig, submit with `sbatch`, poll terminal states with `sacct` (not squeue),
        # treat the per-task sentinel files as ground truth, and write a manifest enabling
        # reattach if the orchestrating process dies. The harness and result/sentinel
        # protocol are reused unchanged from the base class.
        raise NotImplementedError("The slurm backend is not implemented yet (next session).")
