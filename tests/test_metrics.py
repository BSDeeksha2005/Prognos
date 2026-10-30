"""Hand-computed checks for the RUL metrics."""

import numpy as np

from rul.metrics import nasa_score, rmse


def test_rmse_known_values() -> None:
    """Errors of zero and three have RMSE 3/sqrt(2)."""
    assert np.isclose(rmse(np.array([10.0, 10.0]), np.array([10.0, 13.0])), np.sqrt(4.5))


def test_nasa_score_zero_and_asymmetric_penalties() -> None:
    """Zero error scores zero and a two-cycle late error uses divisor ten."""
    assert nasa_score(np.array([10.0]), np.array([10.0])) == 0.0
    assert np.isclose(nasa_score(np.array([10.0]), np.array([12.0])), np.exp(0.2) - 1)
    assert nasa_score(np.array([10.0]), np.array([12.0])) > nasa_score(
        np.array([10.0]), np.array([8.0])
    )
