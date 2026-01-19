from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import numpy as np
from scipy import sparse


class DoubleWellNetworkParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    alpha_rot: float = Field(1.0)
    theta: float = Field(1.0)
    A: np.ndarray | sparse.spmatrix | sparse.sparray
    deg: np.ndarray | None = None

    @field_validator("A")
    @classmethod
    def _ensure_csr(cls, v):
        if sparse.issparse(v):
            return v.tocsr()
        return v

    @model_validator(mode="after")
    def _set_degree(self):
        if self.deg is None:
            self.deg = np.asarray(self.A.sum(axis=1)).reshape(-1) # type:ignore
        return self


def F(x, t, p: DoubleWellNetworkParams) -> np.ndarray:
    n = x.size // 2
    X = x.reshape(n, 2)
    x1 = X[:, 0]
    x2 = X[:, 1]

    grad1 = x1 * (x1 * x1 - 1.0)
    grad2 = x2

    drift1 = -grad1 - p.alpha_rot * grad2
    drift2 = -grad2 + p.alpha_rot * grad1

    Ax = p.A @ X
    coupling1 = -p.theta * (p.deg * x1 - Ax[:, 0])
    coupling2 = -p.theta * (p.deg * x2 - Ax[:, 1])

    drift = np.column_stack((drift1 + coupling1, drift2 + coupling2))
    return drift.reshape(-1)


STATS_FIELDS = ["step", "t", "mean_x1", "mean_x2", "mean_x1_sq", "mean_x2_sq"]


def compute_stats(x, t, step, p):
    n = x.size // 2
    X = x.reshape(n, 2)
    mean_x1 = float(np.mean(X[:, 0]))
    mean_x2 = float(np.mean(X[:, 1]))
    mean_x1_sq = float(np.mean(X[:, 0] * X[:, 0]))
    mean_x2_sq = float(np.mean(X[:, 1] * X[:, 1]))
    return {
        "step": int(step),
        "t": float(t),
        "mean_x1": mean_x1,
        "mean_x2": mean_x2,
        "mean_x1_sq": mean_x1_sq,
        "mean_x2_sq": mean_x2_sq,
    }


def build_initial_condition(cfg, n, rng):
    if "values" in cfg:
        arr = np.asarray(cfg["values"], dtype=float)
        if arr.size == 2 * n:
            return arr.reshape(-1)
        if arr.shape == (n, 2):
            return arr.reshape(-1)
        raise ValueError("initial_condition.values has wrong dimension")
    kind = cfg.get("type", "normal")
    if kind == "zeros":
        return np.zeros(2 * n, dtype=float)
    if kind == "normal":
        mean = float(cfg.get("mean", 0.0))
        std = float(cfg.get("std", 1.0))
        return rng.normal(mean, std, size=2 * n)
    if kind == "uniform":
        low = float(cfg.get("low", -1.0))
        high = float(cfg.get("high", 1.0))
        return rng.uniform(low, high, size=2 * n)
    if kind == "ordered_well":
        x0 = np.zeros((n, 2), dtype=float)
        x0[:, 0] = 1.0
        return x0.reshape(-1)
    raise ValueError(f"Unknown initial_condition.type '{kind}'")
