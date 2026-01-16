from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PoissonDegreeParams(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)
    lambda_: float = Field(..., alias="lambda", gt=0.0)


class ScaleFreeCutoffParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(..., gt=0.0)
    k_min: int = Field(..., ge=0)
    k_max: int = Field(..., ge=1)

    @field_validator("k_max")
    @classmethod
    def _ensure_min_max(cls, v: int, info):
        k_min = info.data.get("k_min")
        if k_min is not None and v < k_min:
            raise ValueError("k_max must be >= k_min")
        return v


class ScaleFreeExpCutoffParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    alpha: float = Field(..., gt=0.0)
    xi: float = Field(..., gt=0.0)
    k_min: int = Field(..., ge=0)
    k_max: int = Field(..., ge=1)

    @field_validator("k_max")
    @classmethod
    def _ensure_min_max(cls, v: int, info):
        k_min = info.data.get("k_min")
        if k_min is not None and v < k_min:
            raise ValueError("k_max must be >= k_min")
        return v


def _sample_poisson(p: PoissonDegreeParams, n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.poisson(lam=p.lambda_, size=n).astype(int)


def _sample_scale_free_cutoff(
    p: ScaleFreeCutoffParams, n: int, rng: np.random.Generator
) -> np.ndarray:
    if p.k_min > p.k_max:
        raise ValueError("k_min must be <= k_max")
    ks = np.arange(p.k_min, p.k_max + 1, dtype=float)
    weights = ks ** (-p.alpha)
    probs = weights / weights.sum()
    samples = rng.choice(ks.astype(int), size=n, p=probs)
    return samples.astype(int)


def _sample_scale_free_exp_cutoff(
    p: ScaleFreeExpCutoffParams, n: int, rng: np.random.Generator
) -> np.ndarray:
    if p.k_min > p.k_max:
        raise ValueError("k_min must be <= k_max")
    ks = np.arange(p.k_min, p.k_max + 1, dtype=float)
    weights = (ks ** (-p.alpha)) * np.exp(-ks / p.xi)
    probs = weights / weights.sum()
    samples = rng.choice(ks.astype(int), size=n, p=probs)
    return samples.astype(int)


_REGISTRY: dict[
    str, tuple[Callable[[Any, int, np.random.Generator], np.ndarray], type[BaseModel]]
] = {
    "poisson": (_sample_poisson, PoissonDegreeParams),
    "scale_free_cutoff": (_sample_scale_free_cutoff, ScaleFreeCutoffParams),
    "scale_free_exp_cutoff": (_sample_scale_free_exp_cutoff, ScaleFreeExpCutoffParams),
}


def get_degree_distribution(name: str, params: dict[str, Any]):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown degree distribution '{name}'. Available: {list(_REGISTRY)}")
    sample_fn, Params = _REGISTRY[name]
    p = Params.model_validate(params)
    return sample_fn, p
