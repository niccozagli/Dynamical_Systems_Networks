import numpy as np
from numba import njit


@njit
def double_well_network_annealed_chunk(
    x,
    t,
    dt,
    steps,
    alpha_rot,
    theta,
    deg,
    deg_sum,
    sigma,
):
    n = x.size // 2
    sqrt_dt = np.sqrt(dt)
    drift1 = np.empty(n)
    drift2 = np.empty(n)
    noise1 = np.empty(n)
    noise2 = np.empty(n)

    for _ in range(steps):
        s1 = 0.0
        s2 = 0.0
        for i in range(n):
            di = deg[i]
            s1 += di * x[2 * i]
            s2 += di * x[2 * i + 1]
        mean_x1 = s1 / deg_sum
        mean_x2 = s2 / deg_sum

        for i in range(n):
            x1 = x[2 * i]
            x2 = x[2 * i + 1]
            grad1 = x1 * (x1 * x1 - 1.0)
            grad2 = x2

            drift1[i] = -grad1 - alpha_rot * grad2 - theta * deg[i] * (x1 - mean_x1)
            drift2[i] = -grad2 + alpha_rot * grad1 - theta * deg[i] * (x2 - mean_x2)

        for i in range(n):
            noise1[i] = np.random.standard_normal()
            noise2[i] = np.random.standard_normal()

        for i in range(n):
            x[2 * i] += dt * drift1[i] + sqrt_dt * sigma * noise1[i]
            x[2 * i + 1] += dt * drift2[i] + sqrt_dt * sigma * noise2[i]

        t += dt

    return x, t


def build_double_well_network_annealed_kernel_params(pF, pG):
    deg = pF.deg
    if deg is None:
        deg = np.asarray(pF.A.sum(axis=1)).reshape(-1)
    deg = np.asarray(deg, dtype=float)
    deg_sum = float(np.sum(deg))
    if deg_sum <= 0:
        raise ValueError("Degree sum must be positive for annealed coupling.")
    return (float(pF.alpha_rot), float(pF.theta), deg, deg_sum, float(pG.sigma))
