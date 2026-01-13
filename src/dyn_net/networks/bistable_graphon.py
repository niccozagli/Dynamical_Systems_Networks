import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse


class BistableGraphonParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(..., ge=1)
    rho_n: float = Field(1.0, gt=0.0)
    amplitude_1: float = 0.2
    mean_1: float = 0.2
    var_1: float = 0.02
    amplitude_2: float = 0.1
    mean_2: float = 0.5
    var_2: float = 0.02
    amplitude_3: float = 0.2
    mean_3: float = 0.8
    var_3: float = 0.0005
    seed: int | None = None


def _w(x: np.ndarray, y: np.ndarray, p: BistableGraphonParams) -> np.ndarray:
    # Symmetric graphon with three peaks.
    return (
        p.amplitude_1
        * np.exp(-((x - p.mean_1) ** 2 + (y - p.mean_1) ** 2) / p.var_1)
        + p.amplitude_2
        * np.exp(-((x - p.mean_2) ** 2 + (y - p.mean_2) ** 2) / p.var_2)
        + p.amplitude_3
        * np.exp(-((x - p.mean_3) ** 4 + (y - p.mean_3) ** 4) / p.var_3)
    )


def build(p: BistableGraphonParams) -> sparse.spmatrix:
    rng = np.random.default_rng(p.seed)
    u = rng.random(p.n)
    uu = u[:, None]
    prob = p.rho_n * _w(uu, uu.T, p)
    prob = np.clip(prob, 0.0, 1.0)

    r = rng.random((p.n, p.n))
    mask = np.triu(r < prob, k=1)
    rows, cols = np.where(mask)
    data = np.ones(len(rows), dtype=np.int8)
    A = sparse.csr_matrix((data, (rows, cols)), shape=(p.n, p.n))
    return A + A.T
