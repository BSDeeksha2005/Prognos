"""Small transparent PSI and two-sample KS drift checks."""

import numpy as np


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compare distributions using reference quantile buckets and PSI."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if (
        reference.size == 0
        or current.size == 0
        or not np.isfinite(reference).all()
        or not np.isfinite(current).all()
    ):
        raise ValueError("PSI inputs must be non-empty and finite")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 2:
        return 0.0 if np.allclose(reference, current) else float("inf")
    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=edges)[0].astype(float)
    ref_share = np.clip(ref_counts / reference.size, 1e-6, None)
    cur_share = np.clip(cur_counts / current.size, 1e-6, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def kolmogorov_smirnov(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Calculate two-sample KS distance and its asymptotic p-value directly."""
    first = np.asarray(reference, dtype=float)
    second = np.asarray(current, dtype=float)
    if (
        first.size == 0
        or second.size == 0
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
    ):
        raise ValueError("KS inputs must be non-empty and finite")
    combined = np.sort(np.concatenate([first, second]))
    first_cdf = np.searchsorted(np.sort(first), combined, side="right") / first.size
    second_cdf = np.searchsorted(np.sort(second), combined, side="right") / second.size
    statistic = float(np.max(np.abs(first_cdf - second_cdf)))
    if statistic == 0.0:
        return 0.0, 1.0
    effective_size = first.size * second.size / (first.size + second.size)
    root_size = np.sqrt(effective_size)
    scaled_statistic = (root_size + 0.12 + 0.11 / root_size) * statistic
    terms = [
        (-1) ** (index - 1) * np.exp(-2 * index**2 * scaled_statistic**2) for index in range(1, 101)
    ]
    p_value = float(np.clip(2 * np.sum(terms), 0.0, 1.0))
    return statistic, p_value


def flag_drift(
    reference: dict[str, np.ndarray],
    current: dict[str, np.ndarray],
    psi_threshold: float = 0.2,
    p_threshold: float = 0.05,
) -> dict[str, dict[str, float | bool]]:
    """Report sensor drift when PSI or KS significance crosses its threshold."""
    results: dict[str, dict[str, float | bool]] = {}
    for sensor, values in reference.items():
        psi = population_stability_index(values, current[sensor])
        statistic, p_value = kolmogorov_smirnov(values, current[sensor])
        results[sensor] = {
            "psi": psi,
            "ks_statistic": statistic,
            "ks_p_value": p_value,
            "drifted": psi >= psi_threshold or p_value < p_threshold,
        }
    return results
