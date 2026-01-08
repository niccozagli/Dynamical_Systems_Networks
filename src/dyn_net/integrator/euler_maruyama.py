import numpy as np
from .params import EulerMaruyamaParams
from dyn_net.utils.stats import write_stats

def euler_maruyama_isotropic(
    F, G,
    x0,
    params_int: EulerMaruyamaParams,
    params_F,
    params_G,
    rng=None,
    *,
    stats_fn,
    stats_writer,   # (csv_writer, file_handle)
):
    writer, fh = stats_writer  # required, bound once

    x = np.asarray(x0, dtype=float).reshape(-1)
    d = x.size

    tmin = params_int.tmin
    tmax = params_int.tmax
    dt = params_int.dt
    stats_every = params_int.stats_every

    n = int((tmax - tmin) / dt)
    sqrt_dt = np.sqrt(dt)
    rng = rng or np.random.default_rng()

    t = float(tmin)

    if params_int.write_stats_at_start:
        write_stats(writer,fh,stats_fn(x, t, 0, params_F))
        

    for step in range(1, n + 1):
        drift = np.asarray(F(x, t, params_F), dtype=float).reshape(d)
        sigma = float(G(x, t, params_G))

        x = x + dt * drift + sqrt_dt * sigma * rng.normal(size=d)
        t = tmin + step * dt

        if step % stats_every == 0:
            write_stats(writer, fh, stats_fn(x, t, step, params_F))

    return x