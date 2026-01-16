from typing import Any

import numpy as np
import networkx as nx
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse

from dyn_net.networks.degree_distributions import get_degree_distribution


class DegreeDistributionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class ConfigurationModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(..., ge=1)
    degree_distribution: DegreeDistributionSpec
    seed: int | None = None
    max_resamples: int = Field(1000, ge=1)


def _ensure_even_degree_sum(
    degrees: np.ndarray,
    sample_fn,
    dist_params,
    rng: np.random.Generator,
    max_resamples: int,
) -> np.ndarray:
    if degrees.sum() % 2 == 0:
        return degrees
    for _ in range(max_resamples):
        idx = int(rng.integers(0, degrees.size))
        degrees[idx] = int(sample_fn(dist_params, 1, rng)[0])
        if degrees.sum() % 2 == 0:
            return degrees
    raise ValueError(
        "Configuration model requires an even sum of degrees; "
        "failed to sample a valid degree sequence."
    )


def build(p: ConfigurationModelParams) -> sparse.spmatrix:
    rng = np.random.default_rng(p.seed)
    sample_fn, dist_params = get_degree_distribution(
        p.degree_distribution.name, p.degree_distribution.params
    )
    degrees = sample_fn(dist_params, p.n, rng).astype(int)
    degrees = _ensure_even_degree_sum(degrees, sample_fn, dist_params, rng, p.max_resamples)
    G = nx.configuration_model(degrees, seed=p.seed)
    return nx.to_scipy_sparse_array(G, format="csr")
