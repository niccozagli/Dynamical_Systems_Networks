from .jit_kuramoto import kuramoto_chunk, build_kuramoto_kernel_params

_JIT_REGISTRY = {
    "kuramoto": (kuramoto_chunk, build_kuramoto_kernel_params),
}


def get_jit_kernel(name: str):
    if name not in _JIT_REGISTRY:
        raise ValueError(f"Unknown JIT kernel for system '{name}'. Available: {list(_JIT_REGISTRY)}")
    return _JIT_REGISTRY[name]
