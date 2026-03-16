from typing import Any, Callable
import numpy as np

from .degree import build_degree_weights, make_degree_weighted_perturbation
from .double_well import alpha_rot_perturbation
from .generic import constant_perturbation


_PHASE_SPACE_PERTURBATIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "alpha_rot": alpha_rot_perturbation,
    "constant": constant_perturbation,
}


def _get_phase_space_perturbation(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name not in _PHASE_SPACE_PERTURBATIONS:
        raise ValueError(
            f"Unknown perturbation '{name}'. Available: {list(_PHASE_SPACE_PERTURBATIONS)}"
        )
    return _PHASE_SPACE_PERTURBATIONS[name]


def get_perturbation(
    spec: str | dict[str, Any],
    *,
    deg: np.ndarray | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    if isinstance(spec, str):
        return _get_phase_space_perturbation(spec)

    perturbation_type = str(spec.get("type", ""))
    if not perturbation_type:
        raise ValueError("perturbation.type must be provided")

    if perturbation_type != "degree_weighted":
        return _get_phase_space_perturbation(perturbation_type)

    if deg is None:
        raise ValueError("degree_weighted perturbation requires the network degrees")

    phase_space_spec = spec.get("phase_space")
    if not isinstance(phase_space_spec, dict):
        raise ValueError("degree_weighted perturbation requires a phase_space config")

    phase_space_type = str(phase_space_spec.get("type", ""))
    if not phase_space_type:
        raise ValueError("phase_space.type must be provided for degree_weighted")

    base_perturbation = _get_phase_space_perturbation(phase_space_type)
    weight_spec = spec.get("degree_weight", {})
    if not isinstance(weight_spec, dict):
        raise ValueError("degree_weight must be an object")
    weights = build_degree_weights(weight_spec, deg)
    return make_degree_weighted_perturbation(base_perturbation, weights)
