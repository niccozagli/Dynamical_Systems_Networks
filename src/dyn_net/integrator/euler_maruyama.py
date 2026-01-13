import time
import numpy as np
from .params import EulerMaruyamaParams
from dyn_net.utils.stats import write_stats
from dyn_net.utils.state import write_state

def euler_maruyama_isotropic(
    F, G,
    x0,
    params_int: EulerMaruyamaParams,
    params_F,
    params_G,
    rng=None,
    *,
    stats_fn,
    stats_writer,   # HDF5 stats writer handle
    state_writer=None,   # HDF5 state writer handle
    state_transform=None,
):
    x = np.asarray(x0, dtype=float).reshape(-1)
    d = x.size

    tmin = params_int.tmin
    tmax = params_int.tmax
    dt = params_int.dt
    stats_every = params_int.stats_every
    state_every = params_int.state_every

    n = int((tmax - tmin) / dt)
    sqrt_dt = np.sqrt(dt)
    rng = rng or np.random.default_rng()
    noise = np.empty(d, dtype=float)

    t = float(tmin)

    if params_int.write_stats_at_start:
        write_stats(stats_writer, stats_fn(x, t, 0, params_F))
    if state_writer is not None and params_int.write_state_at_start:
        x_state = state_transform(x) if state_transform is not None else x
        write_state(state_writer, x_state, t, 0)

    # One-time validation; keep the time loop fast.
    drift0 = F(x, t, params_F)
    if drift0.shape != x.shape:
        raise ValueError("Drift shape mismatch")
    if not np.issubdtype(drift0.dtype, np.floating):
        raise TypeError("Drift must return float array")
    sigma0 = G(x, t, params_G)
    if np.ndim(sigma0) != 0:
        raise ValueError("Isotropic noise expects scalar sigma")

    for step in range(1, n + 1):
        drift = F(x, t, params_F)
        sigma = G(x, t, params_G)

        rng.standard_normal(size=d, out=noise)
        x += dt * drift + sqrt_dt * sigma * noise
        t = tmin + step * dt

        if step % stats_every == 0:
            write_stats(stats_writer, stats_fn(x, t, step, params_F))
        if state_writer is not None and step % state_every == 0:
            x_state = state_transform(x) if state_transform is not None else x
            write_state(state_writer, x_state, t, step)

    return x


def euler_maruyama_isotropic_timed(
    F, G,
    x0,
    params_int: EulerMaruyamaParams,
    params_F,
    params_G,
    rng=None,
    *,
    stats_fn,
    stats_writer,   # HDF5 stats writer handle
    state_writer=None,   # HDF5 state writer handle
    state_transform=None,
):
    x = np.asarray(x0, dtype=float).reshape(-1)
    d = x.size

    tmin = params_int.tmin
    tmax = params_int.tmax
    dt = params_int.dt
    stats_every = params_int.stats_every
    state_every = params_int.state_every

    n = int((tmax - tmin) / dt)
    sqrt_dt = np.sqrt(dt)
    rng = rng or np.random.default_rng()
    noise = np.empty(d, dtype=float)

    t = float(tmin)
    write_stats_time = 0.0
    write_state_time = 0.0

    if params_int.write_stats_at_start:
        write_stats(stats_writer, stats_fn(x, t, 0, params_F))
    if state_writer is not None and params_int.write_state_at_start:
        x_state = state_transform(x) if state_transform is not None else x
        write_state(state_writer, x_state, t, 0)

    # One-time validation; keep the time loop fast.
    drift0 = F(x, t, params_F)
    if drift0.shape != x.shape:
        raise ValueError("Drift shape mismatch")
    if not np.issubdtype(drift0.dtype, np.floating):
        raise TypeError("Drift must return float array")
    sigma0 = G(x, t, params_G)
    if np.ndim(sigma0) != 0:
        raise ValueError("Isotropic noise expects scalar sigma")

    loop_start = time.perf_counter()
    for step in range(1, n + 1):
        drift = F(x, t, params_F)
        sigma = G(x, t, params_G)

        rng.standard_normal(size=d, out=noise)
        x += dt * drift + sqrt_dt * sigma * noise
        t = tmin + step * dt

        if step % stats_every == 0:
            t0 = time.perf_counter()
            write_stats(stats_writer, stats_fn(x, t, step, params_F))
            write_stats_time += time.perf_counter() - t0
        if state_writer is not None and step % state_every == 0:
            t0 = time.perf_counter()
            x_state = state_transform(x) if state_transform is not None else x
            write_state(state_writer, x_state, t, step)
            write_state_time += time.perf_counter() - t0

    loop_time = time.perf_counter() - loop_start
    timings = {
        "loop_s": loop_time,
        "write_stats_s": write_stats_time,
        "write_state_s": write_state_time,
        "compute_s": loop_time - write_stats_time - write_state_time,
        "steps": int(n),
    }
    return x, timings
