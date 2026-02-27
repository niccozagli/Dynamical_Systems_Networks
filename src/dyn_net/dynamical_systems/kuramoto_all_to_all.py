from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import numpy as np
from scipy import sparse


class KuramotoAllToAllParams(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    theta: float = Field(1.0)
    a: np.ndarray = Field(...)
    n_harmonics: int | None = Field(None, gt=0)
    A: np.ndarray | sparse.spmatrix | sparse.sparray | None = None

    @field_validator("a", mode="before")
    @classmethod
    def _ensure_array(cls, v):
        arr = np.asarray(v, dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("a must contain at least one coefficient")
        return arr

    @model_validator(mode="after")
    def _normalize_harmonics(self):
        if self.n_harmonics is None:
            self.n_harmonics = int(self.a.size)
            return self

        m = int(self.n_harmonics)
        if self.a.size < m:
            padded = np.zeros(m, dtype=float)
            padded[: self.a.size] = self.a
            self.a = padded
        elif self.a.size > m:
            self.a = self.a[:m]
        return self


def F(x, t, p: KuramotoAllToAllParams) -> np.ndarray:
    # Expect 1D float array; no validation for speed in time-stepping loops.
    n = x.size
    m = int(p.a.size)
    harmonics = np.arange(1, m + 1, dtype=float)
    kx = np.outer(harmonics, x)
    sin_kx = np.sin(kx)
    cos_kx = np.cos(kx)
    sum_sin = np.sum(sin_kx, axis=1)
    sum_cos = np.sum(cos_kx, axis=1)
    weights = p.a * harmonics
    coupling = (sin_kx.T * sum_cos - cos_kx.T * sum_sin) @ weights
    return -(p.theta / float(n)) * coupling


STATS_FIELDS = ["step", "t", "order_param", "phase_var", "energy"]


def compute_stats(x, t, step, p):
    # Wrap for stats only; dynamics are invariant under 2π shifts.
    x_wrapped = np.mod(x, 2 * np.pi)
    r = np.mean(np.exp(1j * x_wrapped))
    m = int(p.a.size)
    harmonics = np.arange(1, m + 1, dtype=float)
    kx = np.outer(harmonics, x_wrapped)
    r_n = np.mean(np.exp(1j * kx), axis=1)
    energy = -0.5 * float(np.sum(np.abs(r_n) ** 2))
    return {
        "step": int(step),
        "t": float(t),
        "order_param": float(np.abs(r)),
        "phase_var": float(np.var(x_wrapped)),
        "energy": energy,
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
