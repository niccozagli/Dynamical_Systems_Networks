from scipy import sparse
from pydantic import BaseModel, ConfigDict, Field


class AllToAllParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(..., ge=1)
    seed: int | None = None


def build(p: AllToAllParams) -> sparse.spmatrix:
    n = int(p.n)
    # Placeholder: keep only the shape (N) without constructing any edges.
    return sparse.csr_matrix((n, n), dtype=float)
