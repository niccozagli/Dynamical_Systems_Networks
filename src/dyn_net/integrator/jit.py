import time
import numpy as np
from .params import EulerMaruyamaParams
from dyn_net.utils.stats import write_stats
from dyn_net.utils.state import write_state


def integrate_chunked_jit(
    kernel,
    x0,
    params_int: EulerMaruyamaParams,
    kernel_params: tuple,
    *,
    stats_fn,
    stats_writer,
    stats_params=None,
    state_writer=None,
    state_transform=None,
):
    x = np.asarray(x0, dtype=float).reshape(-1)

    tmin = params_int.tmin
    tmax = params_int.tmax
    dt = params_int.dt
    stats_every = params_int.stats_every
    state_every = params_int.state_every

    n = int((tmax - tmin) / dt)
    t = float(tmin)

    if params_int.write_stats_at_start:
        write_stats(stats_writer, stats_fn(x, t, 0, stats_params))
    if state_writer is not None and params_int.write_state_at_start:
        x_state = state_transform(x) if state_transform is not None else x
        write_state(state_writer, x_state, t, 0)

    step = 0
    next_stats = stats_every
    next_state = state_every if state_writer is not None else n + 1

    while step < n:
        next_event = min(next_stats, next_state, n)
        steps_to_run = next_event - step
        x, t = kernel(x, t, dt, steps_to_run, *kernel_params)
        step = next_event

        if step == next_stats:
            write_stats(stats_writer, stats_fn(x, t, step, stats_params))
            next_stats += stats_every
        if step == next_state:
            x_state = state_transform(x) if state_transform is not None else x
            write_state(state_writer, x_state, t, step)
            next_state += state_every

    return x


def integrate_chunked_jit_timed(
    kernel,
    x0,
    params_int: EulerMaruyamaParams,
    kernel_params: tuple,
    *,
    stats_fn,
    stats_writer,
    stats_params=None,
    state_writer=None,
    state_transform=None,
):
    x = np.asarray(x0, dtype=float).reshape(-1)

    tmin = params_int.tmin
    tmax = params_int.tmax
    dt = params_int.dt
    stats_every = params_int.stats_every
    state_every = params_int.state_every

    n = int((tmax - tmin) / dt)
    t = float(tmin)
    write_stats_time = 0.0
    write_state_time = 0.0

    if params_int.write_stats_at_start:
        write_stats(stats_writer, stats_fn(x, t, 0, stats_params))
    if state_writer is not None and params_int.write_state_at_start:
        x_state = state_transform(x) if state_transform is not None else x
        write_state(state_writer, x_state, t, 0)

    step = 0
    next_stats = stats_every
    next_state = state_every if state_writer is not None else n + 1

    loop_start = time.perf_counter()
    while step < n:
        next_event = min(next_stats, next_state, n)
        steps_to_run = next_event - step
        x, t = kernel(x, t, dt, steps_to_run, *kernel_params)
        step = next_event

        if step == next_stats:
            t0 = time.perf_counter()
            write_stats(stats_writer, stats_fn(x, t, step, stats_params))
            write_stats_time += time.perf_counter() - t0
            next_stats += stats_every
        if step == next_state:
            t0 = time.perf_counter()
            x_state = state_transform(x) if state_transform is not None else x
            write_state(state_writer, x_state, t, step)
            write_state_time += time.perf_counter() - t0
            next_state += state_every

    loop_time = time.perf_counter() - loop_start
    timings = {
        "loop_s": loop_time,
        "write_stats_s": write_stats_time,
        "write_state_s": write_state_time,
        "compute_s": loop_time - write_stats_time - write_state_time,
        "steps": int(n),
    }
    return x, timings
