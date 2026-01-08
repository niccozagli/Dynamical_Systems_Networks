import numpy as np
from pydantic import BaseModel, ConfigDict, Field

class AdditiveGaussianParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sigma: float = Field(0.01, ge=0.0)

def G(x: np.ndarray, t: float, p: AdditiveGaussianParams) -> float:
    # scalar diffusion amplitude
    return float(p.sigma)