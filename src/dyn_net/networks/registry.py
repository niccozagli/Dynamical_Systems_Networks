from typing import Any

from .bistable_graphon import BistableGraphonParams, build as bg_build
from .configuration_model import ConfigurationModelParams, build as cm_build
from .erdos_renyi import ErdosRenyiParams, build as er_build
from .power_law import PowerLawParams, build as pl_build
from .watts_strogatz import WattsStrogatzParams, build as ws_build

_REGISTRY = {
    "bistable_graphon": (bg_build, BistableGraphonParams),
    "configuration_model": (cm_build, ConfigurationModelParams),
    "erdos_renyi": (er_build, ErdosRenyiParams),
    "power_law": (pl_build, PowerLawParams),
    "watts_strogatz": (ws_build, WattsStrogatzParams),
}


def get_network(name: str, params: dict[str, Any]):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown network '{name}'. Available: {list(_REGISTRY)}")
    build, Params = _REGISTRY[name]
    p = Params.model_validate(params)
    return build, p
