import numpy as np
from numba import njit


@njit
def double_well_all_to_all_chunk(
    x,
    t,
    dt,
    steps,
    theta,
    sigma,
):
    n = x.size
    sqrt_dt = np.sqrt(dt)
    drift = np.empty(n)
    noise = np.empty(n)

    for _ in range(steps):
        s = 0.0
        for i in range(n):
            s += x[i]
        mean_x = s / n

        for i in range(n):
            xi = x[i]
            drift[i] = xi - xi * xi * xi - theta * (xi - mean_x)

        for i in range(n):
            noise[i] = np.random.standard_normal()

        for i in range(n):
            x[i] += dt * drift[i] + sqrt_dt * sigma * noise[i]

        t += dt

    return x, t


def build_double_well_all_to_all_kernel_params(pF, pG):
    return (float(pF.theta), float(pG.sigma))
