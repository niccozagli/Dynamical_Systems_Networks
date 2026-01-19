import numpy as np
from numba import njit


@njit
def double_well_network_chunk(
    x,
    t,
    dt,
    steps,
    alpha_rot,
    theta,
    indptr,
    indices,
    data,
    sigma,
):
    n = x.size // 2
    sqrt_dt = np.sqrt(dt)
    drift1 = np.empty(n)
    drift2 = np.empty(n)
    noise1 = np.empty(n)
    noise2 = np.empty(n)

    for _ in range(steps):
        for i in range(n):
            sum1 = 0.0
            sum2 = 0.0
            deg = 0.0
            start = indptr[i]
            end = indptr[i + 1]
            for idx in range(start, end):
                j = indices[idx]
                a = data[idx]
                deg += a
                sum1 += a * x[2 * j]
                sum2 += a * x[2 * j + 1]

            x1 = x[2 * i]
            x2 = x[2 * i + 1]
            grad1 = x1 * (x1 * x1 - 1.0)
            grad2 = x2

            drift1[i] = -grad1 - alpha_rot * grad2 - theta * (deg * x1 - sum1)
            drift2[i] = -grad2 + alpha_rot * grad1 - theta * (deg * x2 - sum2)

        for i in range(n):
            noise1[i] = np.random.standard_normal()
            noise2[i] = np.random.standard_normal()

        for i in range(n):
            x[2 * i] += dt * drift1[i] + sqrt_dt * sigma * noise1[i]
            x[2 * i + 1] += dt * drift2[i] + sqrt_dt * sigma * noise2[i]

        t += dt

    return x, t


def build_double_well_network_kernel_params(pF, pG):
    A = pF.A
    indptr = np.asarray(A.indptr, dtype=np.int64)
    indices = np.asarray(A.indices, dtype=np.int64)
    data = np.asarray(A.data, dtype=float)
    return (float(pF.alpha_rot), float(pF.theta), indptr, indices, data, float(pG.sigma))
