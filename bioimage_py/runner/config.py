"""Runner configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RunnerConfig:
    """Base configuration shared by all runners.

    Attributes:
        poll_interval: Seconds between status polls (distributed runners).
        tmp_root: Root directory for job temp folders. ``None`` uses the system default.
            For distributed jobs this must be on a shared filesystem.
        python_executable: Interpreter used to launch worker tasks. ``None`` uses the
            current interpreter (``sys.executable``).
    """

    poll_interval: float = 10.0
    tmp_root: Optional[str] = None
    python_executable: Optional[str] = None


@dataclass
class SlurmConfig(RunnerConfig):
    """Configuration for the (not-yet-implemented) slurm runner.

    Inherits ``poll_interval``, ``tmp_root`` and ``python_executable`` from
    :class:`RunnerConfig`.

    Attributes:
        partition: The slurm partition to submit to.
        time: The per-task time limit (slurm time format, e.g. ``"01:00:00"``).
        mem: The per-task memory limit (e.g. ``"8G"``).
        cpus_per_task: Number of CPUs requested per task.
        gpus: Number of GPUs requested per task.
        account: The accounting project to charge.
        qos: The quality-of-service to request.
        constraint: A node feature constraint.
        shebang: How to set up the environment in the generated job script (shebang /
            activation). The package and all dependencies must be importable on the node.
    """

    partition: Optional[str] = None
    time: Optional[str] = None
    mem: Optional[str] = None
    cpus_per_task: int = 1
    gpus: int = 0
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    # How to set up the environment in the generated job script (shebang / activation).
    # The package and all dependencies must be importable on the compute node.
    shebang: Optional[str] = None
