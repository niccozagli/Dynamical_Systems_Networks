import numpy as np


def constant_perturbation(x: np.ndarray) -> np.ndarray:
    """Constant perturbation: add 1 to every component of the state.

    Returns a vector of ones with the same shape as x.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    return np.ones_like(x_arr)
