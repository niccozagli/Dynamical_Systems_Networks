from .jit_kuramoto import kuramoto_chunk, build_kuramoto_kernel_params
from .jit_double_well_all_to_all import (
    double_well_all_to_all_chunk,
    build_double_well_all_to_all_kernel_params,
)
from .jit_double_well_network import (
    double_well_network_chunk,
    build_double_well_network_kernel_params,
)

_JIT_REGISTRY = {
    "kuramoto": (kuramoto_chunk, build_kuramoto_kernel_params),
    "double_well_all_to_all": (
        double_well_all_to_all_chunk,
        build_double_well_all_to_all_kernel_params,
    ),
    "double_well_network": (
        double_well_network_chunk,
        build_double_well_network_kernel_params,
    ),
}


def get_jit_kernel(name: str):
    if name not in _JIT_REGISTRY:
        raise ValueError(f"Unknown JIT kernel for system '{name}'. Available: {list(_JIT_REGISTRY)}")
    return _JIT_REGISTRY[name]
