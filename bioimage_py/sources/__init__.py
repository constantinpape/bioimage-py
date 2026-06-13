"""Data sources: serializable, array-like handles for the runner."""
from .array_source import ArraySource
from .base import Source, SourceSpec
from .dispatch import SourceLike, as_source, from_spec, register_source

__all__ = [
    "ArraySource",
    "Source",
    "SourceSpec",
    "SourceLike",
    "as_source",
    "from_spec",
    "register_source",
]
