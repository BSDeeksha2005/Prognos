"""Train and evaluate tabular regression baselines."""

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from rul.metrics import nasa_score, rmse


def train_baselines(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    columns: list[str],
    truth: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Fit linear and LightGBM models; report validation and terminal test scores."""
    results: dict[str, dict[str, float]] = {}
    for name, model in [
        ("linear", LinearRegression()),
        (
            "lightgbm",
            LGBMRegressor(
                n_estimators=400,
                learning_rate=0.04,
                num_leaves=31,
                random_state=42,
                verbosity=-1,
                n_jobs=1,
            ),
        ),
    ]:
        with mlflow.start_run(run_name=name):
            model.fit(train[columns], train["rul"])
            validation_prediction = model.predict(validation[columns])
            terminal = test.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=True).tail(1)
            test_prediction = model.predict(terminal[columns])
            scores = {
                "validation_rmse": rmse(validation.rul.to_numpy(), validation_prediction),
                "validation_nasa": nasa_score(validation.rul.to_numpy(), validation_prediction),
                "test_rmse": rmse(truth, test_prediction),
                "test_nasa": nasa_score(truth, test_prediction),
            }
            mlflow.log_params({"model": name, "features": len(columns)})
            mlflow.log_metrics(scores)
            results[name] = scores
    return results
