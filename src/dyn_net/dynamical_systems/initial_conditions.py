from typing import Callable

import numpy as np

from .kuramoto import build_initial_condition as kuramoto_ic
from .kuramoto_all_to_all import build_initial_condition as kuramoto_ata_ic
from .double_well_all_to_all import build_initial_condition as dwaa_ic
from .double_well_network import build_initial_condition as dwn_ic
from .double_well_network_annealed import build_initial_condition as dwn_annealed_ic

_IC_REGISTRY: dict[str, Callable[[dict, int, np.random.Generator], np.ndarray]] = {
    "kuramoto": kuramoto_ic,
    "kuramoto_all_to_all": kuramoto_ata_ic,
    "double_well_all_to_all": dwaa_ic,
    "double_well_network": dwn_ic,
    "double_well_network_annealed": dwn_annealed_ic,
}


def get_initial_condition_builder(name: str):
    if name not in _IC_REGISTRY:
        raise ValueError(
            f"Unknown initial condition builder for system '{name}'. Available: {list(_IC_REGISTRY)}"
        )
    return _IC_REGISTRY[name]
