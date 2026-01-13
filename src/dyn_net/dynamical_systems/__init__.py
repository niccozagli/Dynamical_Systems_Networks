# dynamical_systems/__init__.py
from .registry import get_drift
from .double_well_single import DoubleWellSingleParams
from .kuramoto import KuramotoParams

__all__ = [
    "get_drift",
    "DoubleWellSingleParams",
    "KuramotoParams",
]
