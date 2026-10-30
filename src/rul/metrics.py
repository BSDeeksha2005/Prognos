"""RUL evaluation metrics."""

import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(mean_squared_error(actual, predicted)))


def nasa_score(actual: np.ndarray, predicted: np.ndarray) -> float:
    """NASA asymmetric score; positive errors mean late failure predictions."""
    differences = np.asarray(predicted) - np.asarray(actual)
    penalties = np.where(
        differences < 0, np.exp(-differences / 13.0) - 1, np.exp(differences / 10.0) - 1
    )
    return float(np.sum(penalties))
