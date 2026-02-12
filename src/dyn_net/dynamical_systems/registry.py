from typing import Any
from .double_well_single import F as dw_F, DoubleWellSingleParams
from .double_well_all_to_all import F as dwaa_F, DoubleWellAllToAllParams
from .double_well_network import F as dwn_F, DoubleWellNetworkParams
from .kuramoto import F as kuramoto_F, KuramotoParams

_REGISTRY = {
    "double_well_single": (dw_F, DoubleWellSingleParams),
    "double_well_all_to_all": (dwaa_F, DoubleWellAllToAllParams),
    "double_well_network": (dwn_F, DoubleWellNetworkParams),
    "kuramoto": (kuramoto_F, KuramotoParams),
}

def get_drift(name: str, params : dict[str,Any]):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dynamical system '{name}. Available: {list(_REGISTRY)}")
    
    F, Params = _REGISTRY[name]
    p = Params.model_validate(params)
    return F , p


def get_drift_params_model(name: str):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dynamical system '{name}. Available: {list(_REGISTRY)}")
    return _REGISTRY[name][1]
