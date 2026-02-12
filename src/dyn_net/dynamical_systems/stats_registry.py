from .double_well_single import compute_stats as dw_stats, STATS_FIELDS as dw_fields
from .double_well_all_to_all import (
    compute_stats as dwaa_stats,
    STATS_FIELDS as dwaa_fields,
)
from .double_well_network import compute_stats as dwn_stats, STATS_FIELDS as dwn_fields
from .kuramoto import compute_stats as kuramoto_stats, STATS_FIELDS as kuramoto_fields

_STATS_REGISTRY = {
    "double_well_single": (dw_stats, dw_fields),
    "double_well_all_to_all": (dwaa_stats, dwaa_fields),
    "double_well_network": (dwn_stats, dwn_fields),
    "kuramoto": (kuramoto_stats, kuramoto_fields),
}

def get_stats(name: str):
    if name not in _STATS_REGISTRY:
        raise ValueError(f"Unknown system '{name}'. Available: {list(_STATS_REGISTRY)}")
    return _STATS_REGISTRY[name]
