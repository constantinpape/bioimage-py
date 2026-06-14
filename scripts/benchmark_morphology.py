#!/usr/bin/env python
"""Benchmark the two-pass morphology workflow as a function of the object count.

The first pass (:func:`bioimage_py.morphology.morphology`) is a block-wise reduction whose cost
scales with the *volume*; the second pass (:func:`bioimage_py.morphology.regionprops`) does
per-object work, so its cost scales with the *number of objects*. This script times the two passes
separately over a sweep of object counts and reports the per-object cost, which is the quantity that
exposes a per-object scaling problem.

Examples:
    # Default local sweep over a grid of labeled cubes (exact, controllable object counts):
    python scripts/benchmark_morphology.py

    # Custom counts, also time the marching-cubes surface area, and write a CSV:
    python scripts/benchmark_morphology.py --counts 200 1000 5000 20000 --surface both --csv out.csv

    # Realistic concave blobs (exercises the EDT centroid fallback); count set by the volume size:
    python scripts/benchmark_morphology.py --mode blobs --shape 192 192 192

    # Distributed (subprocess) backend with 8 workers:
    python scripts/benchmark_morphology.py --job-type subprocess --num-workers 8
"""
from __future__ import annotations

import argparse
import itertools
import math
import tempfile
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

import bioimage_py as bp


def make_grid(n_objects: int, cube: int = 4, gap: int = 3, ndim: int = 3) -> np.ndarray:
    """Build a labeled volume of exactly ``n_objects`` non-touching cubes on a regular grid.

    Args:
        n_objects: The exact number of labeled cubes to place.
        cube: The edge length (in voxels) of each cube.
        gap: The background gap between adjacent cubes (>= 1 keeps them separate objects).
        ndim: The number of spatial dimensions.

    Returns:
        A ``uint64`` label volume with labels ``1 .. n_objects``.
    """
    per_axis = int(math.ceil(n_objects ** (1.0 / ndim)))
    pitch = cube + gap
    extent = per_axis * pitch + gap
    seg = np.zeros((extent,) * ndim, dtype="uint64")
    label = 0
    for idx in itertools.product(range(per_axis), repeat=ndim):
        label += 1
        if label > n_objects:
            break
        sl = tuple(slice(gap + i * pitch, gap + i * pitch + cube) for i in idx)
        seg[sl] = label
    return seg


def make_blobs(shape: Sequence[int], rng: np.random.Generator, sigma: float = 1.6,
               thresh_factor: float = 0.8) -> np.ndarray:
    """Build a labeled volume of irregular blobs (smoothed noise -> threshold -> components)."""
    from scipy.ndimage import gaussian_filter
    from skimage.measure import label as sklabel
    field = gaussian_filter(rng.random(tuple(shape)), sigma=sigma)
    return sklabel(field > field.mean() + thresh_factor * field.std()).astype("uint64")


def _as_input(seg: np.ndarray, job_type: str) -> object:
    """Return the op input: the numpy array for ``local``, else a file-backed zarr array."""
    if job_type == "local":
        return seg
    import zarr
    path = tempfile.mkdtemp(prefix="bench_morph_") + "/seg.zarr"
    chunks = tuple(min(64, s) for s in seg.shape)
    z = zarr.open(path, mode="w", shape=seg.shape, chunks=chunks, dtype=seg.dtype)
    z[:] = seg
    return z


def _time(fn) -> Tuple[object, float]:
    """Call ``fn`` and return ``(result, elapsed_seconds)``."""
    t0 = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - t0


def run_one(seg: np.ndarray, *, resolution: Sequence[float], compute_surface: bool,
            job_type: str, num_workers: int, block_shape: Optional[Tuple[int, ...]]) -> dict:
    """Time pass 1 (morphology) and pass 2 (regionprops) for a single segmentation."""
    inp = _as_input(seg, job_type)
    n_fg = int((seg != 0).sum())

    table, morph_s = _time(lambda: bp.morphology.morphology(
        inp, num_workers=num_workers, block_shape=block_shape, job_type=job_type))
    n_obj = len(table)

    df, rprops_s = _time(lambda: bp.morphology.regionprops(
        inp, table, resolution=resolution, compute_surface=compute_surface,
        num_workers=num_workers, job_type=job_type))
    assert len(df) == n_obj

    return {
        "n_obj": n_obj, "shape": tuple(int(s) for s in seg.shape), "n_fg": n_fg,
        "morph_s": morph_s, "rprops_s": rprops_s,
        "morph_us_per_obj": 1e6 * morph_s / max(n_obj, 1),
        "rprops_us_per_obj": 1e6 * rprops_s / max(n_obj, 1),
        "surface": compute_surface,
    }


def _print_table(rows: List[dict]) -> None:
    """Print the collected timing rows as an aligned table."""
    hdr = ("n_obj", "shape", "n_fg", "surf", "morph_s", "rprops_s",
           "morph_us/obj", "rprops_us/obj")
    print(f"{hdr[0]:>8} {hdr[1]:>16} {hdr[2]:>10} {hdr[3]:>5} {hdr[4]:>9} {hdr[5]:>9} "
          f"{hdr[6]:>13} {hdr[7]:>14}")
    for r in rows:
        print(f"{r['n_obj']:>8} {str(r['shape']):>16} {r['n_fg']:>10} "
              f"{('on' if r['surface'] else 'off'):>5} {r['morph_s']:>9.3f} {r['rprops_s']:>9.3f} "
              f"{r['morph_us_per_obj']:>13.1f} {r['rprops_us_per_obj']:>14.1f}")


def _write_csv(rows: List[dict], path: str) -> None:
    """Write the timing rows to a CSV file."""
    import csv
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    """Parse arguments, run the sweep, and report timings."""
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("grid", "blobs"), default="grid",
                   help="grid: exact object counts via labeled cubes; blobs: irregular concave blobs.")
    p.add_argument("--counts", type=int, nargs="+", default=[200, 1000, 5000, 20000],
                   help="Object counts to sweep (grid mode).")
    p.add_argument("--shape", type=int, nargs="+", default=[192, 192, 192],
                   help="Volume shape (blobs mode).")
    p.add_argument("--resolution", type=float, nargs="+", default=None,
                   help="Per-axis voxel size (defaults to ones).")
    p.add_argument("--surface", choices=("off", "on", "both"), default="off",
                   help="Whether to time the marching-cubes surface area.")
    p.add_argument("--job-type", default="local", choices=("local", "subprocess"))
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--block-shape", type=int, nargs="+", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default=None, help="Optional path to also write the timings as CSV.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    surfaces = {"off": [False], "on": [True], "both": [False, True]}[args.surface]
    block_shape = tuple(args.block_shape) if args.block_shape else None

    if args.mode == "grid":
        segs = [(c, make_grid(c)) for c in args.counts]
    else:
        segs = [(None, make_blobs(args.shape, rng))]

    rows: List[dict] = []
    for requested, seg in segs:
        ndim = seg.ndim
        resolution = tuple(args.resolution) if args.resolution else tuple(1.0 for _ in range(ndim))
        for surface in surfaces:
            row = run_one(seg, resolution=resolution, compute_surface=surface,
                          job_type=args.job_type, num_workers=args.num_workers,
                          block_shape=block_shape)
            tag = "" if requested is None else f" (requested {requested})"
            print(f"done: {row['n_obj']} objects, surface={'on' if surface else 'off'}{tag}")
            rows.append(row)

    print()
    _print_table(rows)
    if args.csv:
        _write_csv(rows, args.csv)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
