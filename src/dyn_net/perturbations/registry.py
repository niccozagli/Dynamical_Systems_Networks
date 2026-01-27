from typing import Callable
import numpy as np

from .double_well import alpha_rot_perturbation


_PERTURBATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "alpha_rot": alpha_rot_perturbation,
}


def get_perturbation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name not in _PERTURBATIONS:
        raise ValueError(
            f"Unknown perturbation '{name}'. Available: {list(_PERTURBATIONS)}"
        )
    return _PERTURBATIONS[name]
