import numpy as np
from numba import njit


@njit
def kuramoto_all_to_all_chunk(x, t, dt, steps, theta, weights, sigma):
    n = x.size
    m = weights.size
    sqrt_dt = np.sqrt(dt)
    coef = -(theta / n)

    sin_kx = np.empty((m, n))
    cos_kx = np.empty((m, n))
    sum_sin = np.empty(m)
    sum_cos = np.empty(m)
    drift = np.empty(n)
    noise = np.empty(n)

    for _ in range(steps):
        for k in range(m):
            s_s = 0.0
            s_c = 0.0
            k1 = k + 1
            for i in range(n):
                angle = k1 * x[i]
                s = np.sin(angle)
                c = np.cos(angle)
                sin_kx[k, i] = s
                cos_kx[k, i] = c
                s_s += s
                s_c += c
            sum_sin[k] = s_s
            sum_cos[k] = s_c

        for i in range(n):
            acc = 0.0
            for k in range(m):
                acc += weights[k] * (
                    sin_kx[k, i] * sum_cos[k] - cos_kx[k, i] * sum_sin[k]
                )
            drift[i] = coef * acc

        for i in range(n):
            noise[i] = np.random.standard_normal()

        for i in range(n):
            x[i] += dt * drift[i] + sqrt_dt * sigma * noise[i]

        t += dt

    return x, t


def build_kuramoto_all_to_all_kernel_params(pF, pG):
    a = np.asarray(pF.a, dtype=float).reshape(-1)
    harmonics = np.arange(1, a.size + 1, dtype=float)
    weights = np.ascontiguousarray(a * harmonics, dtype=float)
    return (float(pF.theta), weights, float(pG.sigma))
