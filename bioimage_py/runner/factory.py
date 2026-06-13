"""Factory for obtaining a runner by job type."""
from __future__ import annotations

from typing import Optional

from .base import LocalRunner, Runner
from .config import RunnerConfig
from .distributed import SlurmRunner, SubprocessRunner

_RUNNERS = {
    "local": LocalRunner,
    "subprocess": SubprocessRunner,
    "slurm": SlurmRunner,
}


def get_runner(job_type: str, config: Optional[RunnerConfig] = None) -> Runner:
    """Return a runner for the given job type.

    Args:
        job_type: One of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        config: Optional runner configuration.

    Returns:
        A :class:`~bioimage_py.runner.base.Runner` instance.

    Raises:
        ValueError: If ``job_type`` is unknown.
    """
    try:
        cls = _RUNNERS[job_type.lower()]
    except KeyError:
        raise ValueError(f"Unknown job_type {job_type!r}; expected one of {sorted(_RUNNERS)}.")
    return cls(config)
