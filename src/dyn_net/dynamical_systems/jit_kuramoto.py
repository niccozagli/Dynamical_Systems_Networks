import numpy as np
from numba import njit


@njit
def kuramoto_chunk(x, t, dt, steps, theta, scale, indptr, indices, data, sigma):
    n = x.size
    sqrt_dt = np.sqrt(dt)
    coef = -(theta / (n * scale))

    sin_x = np.empty(n)
    cos_x = np.empty(n)
    sum_cos = np.empty(n)
    sum_sin = np.empty(n)
    noise = np.empty(n)

    for _ in range(steps):
        for i in range(n):
            sin_x[i] = np.sin(x[i])
            cos_x[i] = np.cos(x[i])

        for i in range(n):
            s_c = 0.0
            s_s = 0.0
            start = indptr[i]
            end = indptr[i + 1]
            for idx in range(start, end):
                j = indices[idx]
                a = data[idx]
                s_c += a * cos_x[j]
                s_s += a * sin_x[j]
            sum_cos[i] = s_c
            sum_sin[i] = s_s

        for i in range(n):
            noise[i] = np.random.standard_normal()

        for i in range(n):
            coupling = sin_x[i] * sum_cos[i] - cos_x[i] * sum_sin[i]
            x[i] += dt * (coef * coupling) + sqrt_dt * sigma * noise[i]

        t += dt

    return x, t


def build_kuramoto_kernel_params(pF, pG):
    A = pF.A
    indptr = np.asarray(A.indptr, dtype=np.int64)
    indices = np.asarray(A.indices, dtype=np.int64)
    data = np.asarray(A.data, dtype=float)
    return (float(pF.theta), float(pF.scale), indptr, indices, data, float(pG.sigma))
