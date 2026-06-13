"""Generic, composable wrapper sources."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ..sources.dispatch import SourceLike
from .base import WrapperSource, register_wrapper


@register_wrapper
class ThresholdSource(WrapperSource):
    """Threshold the wrapped source on read.

    ``source[roi] > threshold`` is returned as a boolean array.

    Args:
        source: The wrapped source-like object.
        threshold: The threshold value.
    """

    def __init__(self, source: SourceLike, threshold: float) -> None:
        super().__init__(source)
        self._threshold = float(threshold)

    @property
    def threshold(self) -> float:
        """The threshold value."""
        return self._threshold

    @property
    def dtype(self) -> np.dtype:
        """The (boolean) dtype of the thresholded output."""
        return np.dtype(bool)

    def __getitem__(self, roi: Tuple[slice, ...]) -> np.ndarray:
        """Return the wrapped data thresholded at ``roi``."""
        return self._source[roi] > self._threshold

    def _params(self) -> dict:
        """Return the keyword arguments needed to reconstruct this wrapper."""
        return {"threshold": self._threshold}
