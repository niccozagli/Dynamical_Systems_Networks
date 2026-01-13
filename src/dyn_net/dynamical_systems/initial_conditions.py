from typing import Callable

import numpy as np

from .kuramoto import build_initial_condition as kuramoto_ic

_IC_REGISTRY: dict[str, Callable[[dict, int, np.random.Generator], np.ndarray]] = {
    "kuramoto": kuramoto_ic,
}


def get_initial_condition_builder(name: str):
    if name not in _IC_REGISTRY:
        raise ValueError(
            f"Unknown initial condition builder for system '{name}'. Available: {list(_IC_REGISTRY)}"
        )
    return _IC_REGISTRY[name]
