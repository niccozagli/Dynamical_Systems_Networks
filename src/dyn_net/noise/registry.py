from typing import Any
from .gaussian_isotropic import G as additive_G, AdditiveGaussianParams

_REGISTRY = {
    "additive_gaussian": (additive_G, AdditiveGaussianParams),
}

def get_noise(name: str, params: dict[str, Any]):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown noise '{name}'. Available: {list(_REGISTRY)}")
    G, Params = _REGISTRY[name]
    p = Params.model_validate(params)
    return G, p