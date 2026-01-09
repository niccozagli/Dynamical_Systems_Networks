from .double_well_single import compute_stats as dw_stats, STATS_FIELDS as dw_fields

_STATS_REGISTRY = {
    "double_well_single": (dw_stats, dw_fields),
}

def get_stats(name: str):
    if name not in _STATS_REGISTRY:
        raise ValueError(f"Unknown system '{name}'. Available: {list(_STATS_REGISTRY)}")
    return _STATS_REGISTRY[name]