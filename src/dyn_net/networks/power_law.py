import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy import sparse


class PowerLawParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(..., ge=1)
    alpha: float = Field(1.0)
    beta: float = Field(0.0)
    seed: int | None = None


def build(p: PowerLawParams) -> sparse.spmatrix:
    rng = np.random.default_rng(p.seed)
    rho_n = p.n ** (-p.beta)
    rows: list[int] = []
    cols: list[int] = []
    data: list[int] = []

    for i in range(p.n):
        xi = (i + 1) / p.n
        for j in range(i):
            xj = (j + 1) / p.n
            w = min(1.0 / rho_n, (xi * xj) ** (-p.alpha))
            prob = rho_n * w
            if rng.random() < prob:
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([1, 1])

    return sparse.csr_matrix((data, (rows, cols)), shape=(p.n, p.n))
