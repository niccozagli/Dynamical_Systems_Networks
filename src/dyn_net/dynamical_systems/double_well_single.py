from pydantic import BaseModel, ConfigDict, Field, Extra
import numpy as np


# PARAMETERS OF THE DOUBLE WELL
class DoubleWellSingleParams(BaseModel):
    model_config = ConfigDict(extra=Extra.forbid)
    alpha_rot : float =  1.0

# DYNAMICS OF THE DOUBLE WELL
def F(x,t,p : DoubleWellSingleParams) -> np.ndarray:
    x1, x2 = x 
    y1 = -x1*(x1**2-1) - p.alpha_rot*x2
    y2 = -x2 + p.alpha_rot * x1* (x1**2-1)
    return np.array([y1,y2],dtype=float)

# STATISTICS FOR THE DOUBLE WELL
STATS_FIELDS = ["step", "t", "x1", "x2","energy"]

def compute_stats(x, t, step, p):
    return {
        "step": int(step),
        "t": float(t),
        "x1": x[0],
        "x2": x[1],
        "energy": float(np.dot(x, x))
    }