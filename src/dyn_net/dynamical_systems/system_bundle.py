from .registry import get_drift
from .stats_registry import get_stats
from .jit_registry import get_jit_kernel


def get_system_bundle(name: str, params: dict):
    F, pF = get_drift(name, params)
    stats_fn, stats_fields = get_stats(name)
    kernel, kernel_params_builder = get_jit_kernel(name)
    return F, pF, stats_fn, stats_fields, kernel, kernel_params_builder
