import networkx as nx
from scipy import sparse
from pydantic import BaseModel, ConfigDict, Field


class WattsStrogatzParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(..., ge=1)
    k: int = Field(2, ge=0)
    p: float = Field(0.1, ge=0.0, le=1.0)
    seed: int | None = None


def build(p: WattsStrogatzParams) -> sparse.spmatrix:
    if p.k >= p.n:
        raise ValueError("watts_strogatz requires k < n")
    if p.k % 2 != 0:
        raise ValueError("watts_strogatz requires even k")
    G = nx.watts_strogatz_graph(n=p.n, k=p.k, p=p.p, seed=p.seed)
    return nx.to_scipy_sparse_array(G, format="csr")
