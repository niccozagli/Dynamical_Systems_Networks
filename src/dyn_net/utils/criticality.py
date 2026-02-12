from typing import cast

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import pbdv
from scipy.stats import poisson

from dyn_net.networks.degree_distributions import (
    PoissonDegreeParams,
    ScaleFreeCutoffParams,
    ScaleFreeExpCutoffParams,
)


def _potential_1d(x1: float, theta: float, k: int) -> float:
    """Effective 1D potential for the mean-field single-site measure."""
    x1_sq = x1 ** 2
    return (x1_sq - 1.0) ** 2 / 4 + theta * float(k) * x1_sq / 2


def compute_x1_sq_k0(
    k: int,
    theta: float,
    sigma: float,
    *,
    quad_opts: dict | None = None,
) -> float:
    """Compute <x1^2>_{k,0} by 1D quadrature with optional quad settings."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")

    coef = 2.0 / (sigma * sigma)
    quad_opts = quad_opts or {}

    def weight(x1: float) -> float:
        return np.exp(-coef * _potential_1d(x1, theta, k))

    def num_integrand(x1: float) -> float:
        return (x1 ** 2) * weight(x1)

    num, _ = quad(num_integrand, 0.0, np.inf, limit=200, **quad_opts)
    denom, _ = quad(weight, 0.0, np.inf, limit=200, **quad_opts)
    if denom == 0.0:
        return 0.0
    return float(num / denom)


def _poisson_pmf(mu: float, *, tol: float = 1e-12, k_max: int | None = None):
    """Return truncated Poisson PMF on {0,...,k_max} with tail cutoff."""
    if mu <= 0.0:
        raise ValueError("Poisson mean must be positive")
    if k_max is None:
        # Truncate infinite Poisson support at a tail probability cutoff.
        k_max = int(poisson.ppf(1.0 - tol, mu))
        if not np.isfinite(k_max) or k_max < 0:
            k_max = int(np.ceil(mu + 10.0 * np.sqrt(mu)))
    ks = np.arange(0, k_max + 1, dtype=int)
    pmf = poisson.pmf(ks, mu)
    pmf = pmf / pmf.sum()
    return ks, pmf


def _scale_free_cutoff_pmf(alpha: float, k_min: int, k_max: int):
    """Return normalized scale-free PMF with hard cutoff."""
    ks = np.arange(k_min, k_max + 1, dtype=int)
    weights = ks.astype(float) ** (-alpha)
    pmf = weights / weights.sum()
    return ks, pmf


def _scale_free_exp_cutoff_pmf(alpha: float, xi: float, k_min: int, k_max: int):
    """Return normalized scale-free PMF with exponential cutoff."""
    ks = np.arange(k_min, k_max + 1, dtype=int)
    weights = (ks.astype(float) ** (-alpha)) * np.exp(-ks / xi)
    pmf = weights / weights.sum()
    return ks, pmf


def degree_distribution_to_pi_k(
    degree_distribution: dict,
    *,
    poisson_tol: float = 1e-12,
    poisson_k_max: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a degree distribution spec to (ks, P(k), Pi(k))."""
    name = degree_distribution["name"]
    params = degree_distribution.get("params", {})

    if name == "poisson":
        p = PoissonDegreeParams.model_validate(params)
        # Poisson has infinite support; use tail cutoff or explicit k_max.
        ks, P_k = _poisson_pmf(
            float(p.lambda_), tol=poisson_tol, k_max=poisson_k_max
        )
    elif name == "scale_free_cutoff":
        p = ScaleFreeCutoffParams.model_validate(params)
        # Scale-free cutoffs have finite support via k_min/k_max.
        ks, P_k = _scale_free_cutoff_pmf(
            float(p.alpha), int(p.k_min), int(p.k_max)
        )
    elif name == "scale_free_exp_cutoff":
        p = ScaleFreeExpCutoffParams.model_validate(params)
        # Scale-free exponential cutoffs have finite support via k_min/k_max.
        ks, P_k = _scale_free_exp_cutoff_pmf(
            float(p.alpha), float(p.xi), int(p.k_min), int(p.k_max)
        )
    else:
        raise ValueError(f"Unsupported degree distribution '{name}'")

    k_mean = float(np.sum(ks * P_k))
    if k_mean <= 0.0:
        raise ValueError("Mean degree must be positive")
    pi_k = (ks * P_k) / k_mean
    return ks, P_k, pi_k


def criticality_function(
    theta: float,
    ks: np.ndarray,
    pi_k: np.ndarray,
    sigma: float,
    *,
    quad_opts: dict | None = None,
) -> float:
    """Evaluate the criticality condition function F(theta)."""
    if ks.shape != pi_k.shape:
        raise ValueError("ks and pi_k must have the same shape")
    coef = 2.0 * theta / (sigma * sigma)
    total = 0.0
    for k, pi in zip(ks, pi_k):
        total += float(k) * float(pi) * compute_x1_sq_k0(
            int(k), theta, sigma, quad_opts=quad_opts
        )
    return coef * total - 1.0


def find_theta_c(
    ks: np.ndarray,
    pi_k: np.ndarray,
    sigma: float,
    *,
    theta_bracket: tuple[float, float],
    quad_opts: dict | None = None,
    rtol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Solve F(theta)=0 for a given (ks, Pi(k)) and sigma."""
    def f(theta: float) -> float:
        return criticality_function(
            theta, ks, pi_k, sigma, quad_opts=quad_opts
        )

    a, b = theta_bracket
    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("theta_bracket does not bracket a root")

    root = brentq(f, a, b, rtol=np.float64(rtol), maxiter=max_iter)
    return cast(float, root)


def find_theta_c_from_degree_distribution(
    degree_distribution: dict,
    sigma: float,
    *,
    theta_bracket: tuple[float, float],
    quad_opts: dict | None = None,
    rtol: float = 1e-6,
    max_iter: int = 100,
    poisson_tol: float = 1e-12,
    poisson_k_max: int | None = None,
) -> float:
    """Solve for theta_c given a degree distribution spec and sigma."""
    ks, _, pi_k = degree_distribution_to_pi_k(
        degree_distribution,
        poisson_tol=poisson_tol,
        poisson_k_max=poisson_k_max,
    )
    return find_theta_c(
        ks,
        pi_k,
        sigma,
        theta_bracket=theta_bracket,
        quad_opts=quad_opts,
        rtol=rtol,
        max_iter=max_iter,
    )


def _parabolic_cylinder_D(v: float, z: float) -> float:
    # scipy.special.pbdv returns (D_v(z), D'_v(z))
    val, _ = pbdv(v, z)
    return float(val)


def critical_noise_function_all_to_all(theta: float, sigma: float) -> float:
    """Equation for critical noise strength in all-to-all double well."""
    if theta <= 0.0:
        raise ValueError("theta must be positive")
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    z = (theta - 1.0) / sigma
    numerator = _parabolic_cylinder_D(-1.5, z)
    denominator = _parabolic_cylinder_D(-0.5, z)
    if denominator == 0.0:
        return np.inf
    return (numerator / denominator) - (sigma / theta)


def find_sigma_c_all_to_all(
    theta: float,
    *,
    sigma_bracket: tuple[float, float],
    rtol: float = 1e-6,
    max_iter: int = 100,
) -> float:
    """Solve for sigma_c in the all-to-all double well criticality equation."""
    if theta <= 0.0:
        raise ValueError("theta must be positive")

    a, b = sigma_bracket
    if a <= 0.0 or b <= 0.0:
        raise ValueError("sigma_bracket entries must be positive")

    def f(sigma: float) -> float:
        return critical_noise_function_all_to_all(theta, sigma)

    fa = f(a)
    fb = f(b)
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    if fa * fb > 0.0:
        raise ValueError("sigma_bracket does not bracket a root")

    root = brentq(f, a, b, rtol=np.float64(rtol), maxiter=max_iter)
    return cast(float, root)
