from pydantic import BaseModel, ConfigDict, Field, Extra
import numpy as np

class DoubleWellSingleParams(BaseModel):
    model_config = ConfigDict(extra=Extra.forbid)
    alpha_rot : float =  1.0

def F(x,t,p : DoubleWellSingleParams) -> np.ndarray:
    x1, x2 = x 
    y1 = -x1*(x1**2-1) - p.alpha_rot*x2
    y2 = -x2 + p.alpha_rot * x1* (x1**2-1)
    return np.array([y1,y2],dtype=float)
