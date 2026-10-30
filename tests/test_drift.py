"""Tests for PSI and KS drift detection."""

import numpy as np

from rul.drift import flag_drift, kolmogorov_smirnov, population_stability_index


def test_identical_distribution_has_zero_psi() -> None:
    """An unchanged series has no population stability shift."""
    values = np.linspace(0, 1, 100)
    assert population_stability_index(values, values) == 0.0
    assert kolmogorov_smirnov(values, values) == (0.0, 1.0)


def test_shift_is_detected_by_psi_and_ks() -> None:
    """A deterministic location shift is reported as drifted."""
    base = np.linspace(-1, 1, 500)
    shifted = base + 2
    statistic, p_value = kolmogorov_smirnov(base, shifted)
    assert statistic > 0.5
    assert p_value < 0.05
    result = flag_drift({"sensor_1": base}, {"sensor_1": shifted})
    assert result["sensor_1"]["drifted"] is True
