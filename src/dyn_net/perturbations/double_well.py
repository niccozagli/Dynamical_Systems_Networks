import numpy as np


def alpha_rot_perturbation(x: np.ndarray) -> np.ndarray:
    """Perturbation field corresponding to an increase in alpha_rot.

    For each node with state (x1, x2), the local field is:
        (-x2, x1 * (x1**2 - 1))

    This matches (0, -1; 1, 0) ∇V with V(x1, x2) = (x1^2 - 1)^2 / 4 + x2^2 / 2.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    if x_arr.size % 2 != 0:
        raise ValueError("alpha_rot_perturbation expects an even-sized state vector")
    X = x_arr.reshape(-1, 2)
    x1 = X[:, 0]
    x2 = X[:, 1]
    pert1 = -x2
    pert2 = x1 * (x1 * x1 - 1.0)
    return np.column_stack((pert1, pert2)).reshape(-1)
