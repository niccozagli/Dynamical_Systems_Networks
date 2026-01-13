import networkx as nx
from scipy import sparse
from pydantic import BaseModel, ConfigDict, Field


class ErdosRenyiParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n: int = Field(..., ge=1)
    p: float = Field(0.1, ge=0.0, le=1.0)
    seed: int | None = None
    directed: bool = False


def build(p: ErdosRenyiParams) -> sparse.spmatrix:
    G = nx.gnp_random_graph(n=p.n, p=p.p, seed=p.seed, directed=p.directed)
    return nx.to_scipy_sparse_array(G, format="csr")
