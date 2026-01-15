from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np
from scipy import sparse


class KuramotoParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    theta: float = Field(1.0)
    scale: float = Field(1.0, gt=0.0)
    A: np.ndarray | sparse.spmatrix | sparse.sparray

    @field_validator("A")
    @classmethod
    def _ensure_csr(cls, v):
        if sparse.issparse(v):
            return v.tocsr()
        return v


def F(x, t, p: KuramotoParams) -> np.ndarray:
    # Expect 1D float array; no validation for speed in time-stepping loops.
    n = x.size
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    sum_cos = p.A @ cos_x
    sum_sin = p.A @ sin_x
    coupling = sin_x * sum_cos - cos_x * sum_sin
    return -(p.theta / (float(n) * float(p.scale))) * coupling


STATS_FIELDS = ["step", "t", "order_param", "phase_var"]


def compute_stats(x, t, step, p):
    # Wrap for stats only; dynamics are invariant under 2π shifts.
    x_wrapped = np.mod(x, 2 * np.pi)
    r = np.mean(np.exp(1j * x_wrapped))
    return {
        "step": int(step),
        "t": float(t),
        "order_param": float(np.abs(r)),
        "phase_var": float(np.var(x_wrapped)),
    }


def build_initial_condition(cfg, n, rng):
    if "values" in cfg:
        arr = np.asarray(cfg["values"], dtype=float).reshape(-1)
        if arr.size != n:
            raise ValueError("initial_condition.values has wrong dimension")
        return arr
    kind = cfg.get("type", "uniform")
    if kind == "zeros":
        return np.zeros(n, dtype=float)
    if kind == "normal":
        mean = float(cfg.get("mean", 0.0))
        std = float(cfg.get("std", 1.0))
        return rng.normal(mean, std, size=n)
    if kind == "uniform":
        low = float(cfg.get("low", 0.0))
        high = float(cfg.get("high", 2 * np.pi))
        return rng.uniform(low, high, size=n)
    raise ValueError(f"Unknown initial_condition.type '{kind}'")
