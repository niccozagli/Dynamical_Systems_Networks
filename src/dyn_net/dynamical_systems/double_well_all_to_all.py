from pydantic import BaseModel, ConfigDict, Field
import numpy as np
from scipy import sparse


class DoubleWellAllToAllParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    theta: float = Field(1.0)
    A: np.ndarray | sparse.spmatrix | sparse.sparray | None = None


def F(x, t, p: DoubleWellAllToAllParams) -> np.ndarray:
    # Expect 1D float array; no validation for speed in time-stepping loops.
    mean_x = float(np.mean(x))
    drift = x - x * x * x - p.theta * (x - mean_x)
    return drift


STATS_FIELDS = ["step", "t", "mean_x"]


def compute_stats(x, t, step, p):
    return {
        "step": int(step),
        "t": float(t),
        "mean_x": float(np.mean(x)),
    }


def build_initial_condition(cfg, n, rng):
    if "values" in cfg:
        arr = np.asarray(cfg["values"], dtype=float).reshape(-1)
        if arr.size != n:
            raise ValueError("initial_condition.values has wrong dimension")
        return arr
    kind = cfg.get("type", "normal")
    if kind == "zeros":
        return np.zeros(n, dtype=float)
    if kind == "normal":
        mean = float(cfg.get("mean", 0.0))
        std = float(cfg.get("std", 1.0))
        return rng.normal(mean, std, size=n)
    if kind == "uniform":
        low = float(cfg.get("low", -1.0))
        high = float(cfg.get("high", 1.0))
        return rng.uniform(low, high, size=n)
    if kind == "ordered_well":
        return np.ones(n, dtype=float)
    raise ValueError(f"Unknown initial_condition.type '{kind}'")
