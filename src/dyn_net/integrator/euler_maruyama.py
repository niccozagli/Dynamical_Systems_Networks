import numpy as np

def euler_maruyama_isotropic(F, G, x0, tmin, tmax, dt, params_F, params_G, rng=None):
    n = int((tmax - tmin) / dt)
    t = tmin + dt * np.arange(n + 1)

    x0 = np.asarray(x0, dtype=float)
    d = x0.size

    x = np.empty((n + 1, d), dtype=float)
    x[0] = x0

    rng = rng or np.random.default_rng()
    sqrt_dt = np.sqrt(dt)

    for k in range(n):
        tk = t[k]
        xk = x[k]

        drift = np.asarray(F(xk, tk, params_F), dtype=float).reshape(d)
        sigma = float(G(xk, tk, params_G))  

        x[k + 1] = xk + dt * drift + sqrt_dt * sigma * rng.normal(size=d)

    return t, x