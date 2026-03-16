import numpy as np


def build_degree_weights(spec: dict, deg: np.ndarray) -> np.ndarray:
    """Build node-wise scalar weights S(k_i) from a degree-weight config."""
    weight_type = str(spec.get("type", "power"))
    deg_arr = np.asarray(deg, dtype=float).reshape(-1)
    if deg_arr.size == 0:
        raise ValueError("Degree vector must be non-empty")

    if weight_type != "power":
        raise ValueError(
            f"Unknown degree weight '{weight_type}'. Available: ['power']"
        )

    exponent = float(spec.get("exponent", 1.0))
    weights = np.power(deg_arr, exponent)
    if not np.all(np.isfinite(weights)):
        raise ValueError(
            "Degree weights must be finite; check exponent against zero-degree nodes."
        )

    normalize = str(spec.get("normalize", "mean"))
    if normalize == "mean":
        scale = float(np.mean(weights))
    elif normalize == "none":
        scale = 1.0
    else:
        raise ValueError(
            f"Unknown degree weight normalization '{normalize}'. "
            "Available: ['mean', 'none']"
        )

    if scale <= 0.0:
        raise ValueError("Degree weight normalization must be positive")
    return weights / scale


def make_degree_weighted_perturbation(base_perturbation, weights: np.ndarray):
    """Scale each node's perturbation block by a scalar node weight."""
    weight_arr = np.asarray(weights, dtype=float).reshape(-1)
    if weight_arr.size == 0:
        raise ValueError("Degree weights must be non-empty")

    def perturb(x: np.ndarray) -> np.ndarray:
        base = np.asarray(base_perturbation(x), dtype=float).reshape(-1)
        n = weight_arr.size
        if base.size % n != 0:
            raise ValueError(
                "Perturbation size is not compatible with the network degree vector"
            )
        block_size = base.size // n
        return (base.reshape(n, block_size) * weight_arr[:, None]).reshape(-1)

    return perturb
