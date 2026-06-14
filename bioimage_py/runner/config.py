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
    """Configuration for the slurm runner.

    Inherits ``poll_interval``, ``tmp_root`` and ``python_executable`` from
    :class:`RunnerConfig`. For slurm, ``tmp_root`` is **required** and must point at a
    shared filesystem visible to all compute nodes (not node-local ``/tmp``), and
    ``num_workers`` (passed to the op / ``run``) is interpreted as the array throttle — the
    maximum number of tasks allowed to run concurrently — independently of how many tasks
    the work is partitioned into.

    Attributes:
        partition: The slurm partition to submit to.
        time: The per-task time limit (slurm time format, e.g. ``"01:00:00"``).
        mem: The per-task memory limit (e.g. ``"8G"``).
        cpus_per_task: Number of CPUs requested per task.
        gpus: Number of GPUs requested per task (emitted as ``--gpus`` only when > 0).
        account: The accounting project to charge.
        qos: The quality-of-service to request.
        constraint: A node feature constraint.
        shebang: Optional environment setup for the generated job script. If given, its
            first line must be an interpreter line (starting with ``#!``) which is placed at
            the top of the script; any remaining lines are emitted as an activation preamble
            *after* the ``#SBATCH`` directives (so the directives are still honoured). The
            preamble is for making the package importable on the node (e.g. ``module load``
            / ``LD_LIBRARY_PATH`` exports), not for choosing the interpreter: the worker is
            always launched with the absolute ``python_executable`` (defaulting to the
            submitting ``sys.executable``). ``None`` uses ``#!/bin/bash`` and that absolute
            interpreter, which needs no activation when the env lives on a shared
            filesystem. Example::

                shebang = "#!/bin/bash\\nmodule load gcc\\nexport LD_LIBRARY_PATH=...:$LD_LIBRARY_PATH"

        max_array_size: Override for the maximum number of array tasks per job. ``None``
            queries the cluster's ``MaxArraySize`` (falling back to a safe default). A run
            partitioned into more tasks than this is rejected up front with a clear error.
        latency_wait: Seconds to wait for a finished task's ``.success`` sentinel to become
            visible on a shared (NFS) filesystem before giving up on it. A task that the
            scheduler reports ``COMPLETED`` wrote its sentinel, but the orchestrating node's
            attribute cache can lag the compute node by up to the mount's ``acdirmax``
            (typically 60 s); this must comfortably exceed that. It only bounds the wait on a
            ``COMPLETED``-but-not-yet-visible task — a task is resolved the moment its
            sentinel appears, so a generous value does not slow down successful runs.
    """

    partition: Optional[str] = None
    time: Optional[str] = None
    mem: Optional[str] = None
    cpus_per_task: int = 1
    gpus: int = 0
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    shebang: Optional[str] = None
    max_array_size: Optional[int] = None
    latency_wait: float = 120.0
