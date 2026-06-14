"""Advanced per-object morphology: scikit-image regionprops + surface area and centroid correction.

A *second pass* on top of :func:`bioimage_py.morphology.morphology`: for each labeled object it crops the
sub-volume by the precomputed bounding box, masks ``== label``, and computes scikit-image regionprops
(in physical units via ``spacing``) plus a marching-cubes ``surface_area`` (3D) and a corrected centroid
(the center-of-mass if it lies inside the object, else the deepest-interior voxel via the Euclidean
distance transform). The regionprops ``area`` already gives the physical volume under ``spacing``. Work
is mapped one task per object with the generic
:meth:`bioimage_py.runner.base.Runner.map`, so it runs identically across ``local`` / ``subprocess`` /
``slurm``; for distributed backends the base table is serialized to a temp file the workers read.
"""
from __future__ import annotations

import functools
import os
import tempfile
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import bioimage_cpp as bic
import numpy as np

from ..runner import get_runner
from ..runner.config import RunnerConfig
from ..sources import Source, SourceLike, as_source
from .morphology import _axis_names

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["regionprops"]

# Non-positional shape descriptors: unambiguous under cropping (no local/global frame issue).
_DEFAULT_PROPERTIES = ("area", "equivalent_diameter_area", "extent", "solidity",
                       "euler_number", "axis_major_length", "axis_minor_length")

# Per-worker caches so each task reopens the segmentation / reads the table once, not per object.
_SEG_CACHE: Dict[Any, Source] = {}
_TABLE_CACHE: Dict[str, "pd.DataFrame"] = {}


def _check_skimage() -> None:
    """Ensure scikit-image is importable and new enough to support the ``spacing`` argument."""
    try:
        import inspect

        import skimage
        from skimage.measure import regionprops_table
    except ImportError as exc:  # pragma: no cover - exercised only without the dependency.
        raise ImportError(
            "bioimage_py.morphology.regionprops requires scikit-image (>= 0.20); "
            "install it (e.g. pip install 'scikit-image>=0.20')."
        ) from exc
    if "spacing" not in inspect.signature(regionprops_table).parameters:
        raise RuntimeError(
            "bioimage_py.morphology.regionprops needs scikit-image >= 0.20 for the 'spacing' "
            f"argument; found {skimage.__version__}."
        )


def _read_table(path: str) -> "pd.DataFrame":
    """Read a serialized table from a ``.csv`` / ``.xlsx`` path."""
    import pandas as pd
    lower = str(path).lower()
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path!r} (use .csv or .xlsx/.xls).")


def _load_table(table: Union[str, "pd.DataFrame"]) -> "pd.DataFrame":
    """Return ``table`` as a DataFrame (pass-through, or read from a path)."""
    import pandas as pd
    if isinstance(table, pd.DataFrame):
        return table
    return _read_table(str(table))


def _required_columns(axes: Sequence[str]) -> List[str]:
    """The base-morphology columns this op consumes."""
    return (["label"] + [f"com_{a}" for a in axes]
            + [f"bb_min_{a}" for a in axes] + [f"bb_max_{a}" for a in axes])


def _resolve_seg(seg: Union[Source, Any]) -> Source:
    """Return an opened segmentation source, reopening (and caching) a `SourceSpec` if needed."""
    if isinstance(seg, Source):
        return seg
    key = (seg.kind, seg.path, seg.internal_path)
    src = _SEG_CACHE.get(key)
    if src is None:
        from ..sources.dispatch import from_spec
        src = from_spec(seg)
        _SEG_CACHE[key] = src
    return src


def _resolve_table(table: Union[str, "pd.DataFrame"]) -> "pd.DataFrame":
    """Return the table DataFrame, reading (and caching) it if a path was passed."""
    if isinstance(table, str):
        df = _TABLE_CACHE.get(table)
        if df is None:
            df = _read_table(table)
            _TABLE_CACHE[table] = df
        return df
    return table


def _surface_area(mask: np.ndarray, resolution: Sequence[float]) -> float:
    """Physical surface area (nm² etc.) of a 3D mask via marching cubes; 0.0 if degenerate."""
    from skimage.measure import marching_cubes, mesh_surface_area
    if not mask.any():
        return 0.0
    # Pad by 1 so border voxels produce closed faces; spacing makes the area physical.
    padded = np.pad(mask, 1).astype("float32")
    try:
        verts, faces, _, _ = marching_cubes(padded, level=0.5, spacing=tuple(resolution))
        return float(mesh_surface_area(verts, faces))
    except (RuntimeError, ValueError):
        # Objects too thin/small for a closed surface.
        return 0.0


def _corrected_centroid(mask: np.ndarray, com_global_vox: np.ndarray, origin: np.ndarray,
                        resolution: Sequence[float]) -> np.ndarray:
    """Return the corrected centroid in physical units.

    Keeps the center-of-mass when it lands inside the object; otherwise uses the deepest-interior
    voxel (argmax of the Euclidean distance transform), then scales to physical units.
    """
    spacing = np.asarray(resolution, dtype="float64")
    if not mask.any():
        return com_global_vox * spacing
    com_local = com_global_vox - origin
    rounded = np.round(com_local).astype(int)
    inside = (np.all(rounded >= 0) and np.all(rounded < np.asarray(mask.shape))
              and bool(mask[tuple(rounded)]))
    if inside:
        centroid_vox = com_global_vox
    else:
        # Deepest-interior point: argmax of the (anisotropic) exact EDT from bioimage_cpp.
        dt = bic.distance.distance_transform(mask.astype("uint8"), sampling=tuple(resolution))
        idx = np.unravel_index(int(np.argmax(dt)), mask.shape)
        centroid_vox = np.asarray(idx, dtype="float64") + origin
    return centroid_vox * spacing


def _object_features(index: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
    """Compute the feature row for a single object (the per-item function handed to ``runner.map``)."""
    from skimage.measure import regionprops_table
    seg = _resolve_seg(ctx["seg"])
    df = _resolve_table(ctx["table"])
    axes, ndim = ctx["axes"], ctx["ndim"]
    resolution, properties = ctx["resolution"], ctx["properties"]

    row = df.iloc[index]
    label = int(row["label"])
    com = np.array([float(row[f"com_{a}"]) for a in axes], dtype="float64")
    bb_min = np.array([int(row[f"bb_min_{a}"]) for a in axes], dtype="int64")
    bb_max = np.array([int(row[f"bb_max_{a}"]) for a in axes], dtype="int64")

    crop = np.asarray(seg[tuple(slice(int(lo), int(hi)) for lo, hi in zip(bb_min, bb_max))])
    mask = crop == label
    n_voxels = int(mask.sum())

    result: Dict[str, Any] = {"label": label, "n_voxels": n_voxels}

    # scikit-image regionprops in physical units (spacing). One synthetic region (label 1); the
    # real label is kept separately. A degenerate object that breaks a property yields NaNs.
    if n_voxels > 0:
        try:
            feats = regionprops_table(mask.astype("uint8"), properties=list(properties),
                                      spacing=tuple(resolution))
            for key, vals in feats.items():
                if key != "label":
                    result[key] = float(vals[0])
        except Exception:  # noqa: BLE001 - degenerate object: report NaNs, do not fail the task.
            for key in properties:
                if key != "label":
                    result[key] = float("nan")
    else:
        for key in properties:
            if key != "label":
                result[key] = float("nan")

    if ctx["compute_surface"] and ndim == 3:
        result["surface_area"] = _surface_area(mask, resolution)

    centroid = _corrected_centroid(mask, com, bb_min.astype("float64"), resolution)
    for a, ax in enumerate(axes):
        result[f"centroid_{ax}"] = float(centroid[a])
    for a, ax in enumerate(axes):
        result[f"bb_min_{ax}"] = int(bb_min[a])
        result[f"bb_max_{ax}"] = int(bb_max[a])
    return result


def _order_columns(out: "pd.DataFrame", axes: Sequence[str], compute_surface: bool,
                   ndim: int) -> List[str]:
    """Stable column order: identity, geometry, surface, then the regionprops-derived columns."""
    leading = (["label", "n_voxels"] + [f"centroid_{a}" for a in axes]
               + [f"bb_min_{a}" for a in axes] + [f"bb_max_{a}" for a in axes])
    if compute_surface and ndim == 3:
        leading.append("surface_area")
    leading = [c for c in leading if c in out.columns]
    rest = [c for c in out.columns if c not in leading]
    return leading + rest


def _write_table(out: "pd.DataFrame", output_path: str) -> None:
    """Write the result table to a ``.csv`` / ``.xlsx`` path (creating parent dirs)."""
    parent = os.path.dirname(os.path.abspath(str(output_path)))
    if parent:
        os.makedirs(parent, exist_ok=True)
    lower = str(output_path).lower()
    if lower.endswith((".xlsx", ".xls")):
        out.to_excel(output_path, index=False)
    elif lower.endswith(".csv"):
        out.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {output_path!r} (use .csv or .xlsx/.xls).")


def regionprops(
    input: SourceLike,
    table: Union[str, "pd.DataFrame"],
    *,
    resolution: Optional[Sequence[float]] = None,
    properties: Optional[Sequence[str]] = None,
    compute_surface: bool = True,
    output_path: Optional[str] = None,
    num_workers: int = 1,
    job_type: str = "local",
    job_config: Optional[RunnerConfig] = None,
    pre_cleanup: Optional[Callable[[str], None]] = None,
) -> "pd.DataFrame":
    """Compute per-object morphology features for a labeled volume, one task per object.

    For each object listed in ``table`` (the output of :func:`morphology`), the sub-volume is cropped by
    its bounding box, masked to the label, and described with scikit-image regionprops (in physical units
    via ``spacing=resolution``; the ``area`` property is therefore the physical volume) plus a
    marching-cubes ``surface_area`` (3D only) and a corrected centroid (the center-of-mass when it lies
    inside the object, otherwise the deepest-interior voxel — the argmax of the Euclidean distance
    transform).

    Args:
        input: The labeled segmentation (a numpy/zarr/n5 array or a `Source`); integer-typed. For the
            ``subprocess``/``slurm`` backends it must be file-backed (zarr/n5).
        table: The base morphology table — a pandas DataFrame or a path to a ``.csv`` / ``.xlsx`` file.
            Must contain ``label``, ``com_<axis>``, ``bb_min_<axis>`` and ``bb_max_<axis>`` (``bb_max``
            is the exclusive slice stop, as produced by :func:`morphology`).
        resolution: Per-axis physical voxel size in array (e.g. z, y, x) order. Defaults to ones (voxel
            units).
        properties: scikit-image regionprops property names. Defaults to a curated non-positional set;
            adding positional properties (e.g. ``centroid``/``bbox``) yields crop-local values.
        compute_surface: Whether to add ``surface_area`` (3D inputs only).
        output_path: Optional ``.csv`` / ``.xlsx`` path to also write the result to.
        num_workers: Number of parallel workers (threads for ``local``, tasks for distributed backends).
        job_type: Execution backend: one of ``"local"``, ``"subprocess"`` or ``"slurm"``.
        job_config: Backend configuration (a `RunnerConfig` / `SlurmConfig`).
        pre_cleanup: Optional ``pre_cleanup(tmp_folder)`` callback invoked on the orchestrating
            process with the job temp folder right before it is deleted (distributed backends only).
            Use it to read out the per-task timing files under ``tmp_folder/timings/`` before cleanup.
            Ignored for the ``local`` backend (no temp folder).

    Returns:
        A pandas DataFrame with one row per object, sorted by ``label``: ``label``, ``n_voxels`` (raw
        voxel count), ``centroid_<axis>`` (corrected, physical), ``bb_min_<axis>``/``bb_max_<axis>``
        (global voxels), ``surface_area`` (3D), and the requested regionprops columns (with the default
        ``properties``, ``area`` is the physical volume).
    """
    import pandas as pd

    _check_skimage()
    src = as_source(input)
    if not np.issubdtype(np.dtype(src.dtype), np.integer):
        raise ValueError(f"regionprops expects an integer label image, got dtype {src.dtype}.")
    ndim = src.ndim
    axes = _axis_names(ndim)

    if resolution is None:
        resolution = tuple(1.0 for _ in range(ndim))
    else:
        resolution = tuple(float(r) for r in resolution)
        if len(resolution) != ndim:
            raise ValueError(f"resolution {resolution} does not match the input ndim {ndim}.")
    properties = list(_DEFAULT_PROPERTIES if properties is None else properties)

    df = _load_table(table)
    missing = [c for c in _required_columns(axes) if c not in df.columns]
    if missing:
        raise ValueError(
            f"table is missing required columns {missing}; pass the output of "
            "bioimage_py.morphology.morphology (label / com_* / bb_min_* / bb_max_*)."
        )

    n = len(df)
    if n == 0:
        empty = ["label", "n_voxels"] + [f"centroid_{a}" for a in axes] \
            + [f"bb_min_{a}" for a in axes] + [f"bb_max_{a}" for a in axes] \
            + (["surface_area"] if (compute_surface and ndim == 3) else []) \
            + [p for p in properties if p != "label"]
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in empty})

    seg_arg: Any = src
    table_arg: Any = df
    tmp_table: Optional[str] = None
    try:
        if job_type != "local":
            try:
                seg_arg = src.to_spec()
            except ValueError as err:
                raise ValueError(
                    f"Distributed regionprops requires a file-backed (zarr/n5) segmentation. {err}"
                ) from err
            # Serialize the (column-subset) table to a temp file the workers read (shared FS for slurm).
            tmp_root = job_config.tmp_root if job_config is not None else None
            fd, tmp_table = tempfile.mkstemp(prefix="bioimage_py_morph_", suffix=".csv", dir=tmp_root)
            os.close(fd)
            df[_required_columns(axes)].to_csv(tmp_table, index=False)
            table_arg = tmp_table

        ctx = {
            "seg": seg_arg, "table": table_arg, "resolution": resolution,
            "properties": tuple(properties), "axes": tuple(axes), "ndim": ndim,
            "compute_surface": bool(compute_surface),
        }
        runner = get_runner(job_type, job_config)
        results = runner.map(functools.partial(_object_features, ctx=ctx), n,
                             num_workers=num_workers, has_return_val=True, name="regionprops",
                             pre_cleanup=pre_cleanup)
    finally:
        if tmp_table is not None and os.path.exists(tmp_table):
            os.remove(tmp_table)

    out = pd.DataFrame(results)
    out = out[_order_columns(out, axes, compute_surface, ndim)].sort_values("label")
    out = out.reset_index(drop=True)
    if output_path is not None:
        _write_table(out, output_path)
    return out
