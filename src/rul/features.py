"""Causal per-cycle features and per-engine sequence windows."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from rul.config import FEATURE_COLUMNS, SEED


def engineer_features(cycles: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Add trailing mean, standard deviation, and least-squares slope features."""
    ordered = cycles.sort_values(["unit_id", "cycle"]).copy()
    grouped = ordered.groupby("unit_id", sort=False)
    for column in FEATURE_COLUMNS:
        rolling = grouped[column].rolling(window, min_periods=1)
        ordered[f"{column}_mean"] = rolling.mean().reset_index(level=0, drop=True)
        ordered[f"{column}_std"] = rolling.std().fillna(0).reset_index(level=0, drop=True)
        ordered[f"{column}_slope"] = (
            grouped[column]
            .transform(lambda values: values.rolling(window, min_periods=2).apply(_slope, raw=True))
            .fillna(0)
        )
    return ordered.reset_index(drop=True)


def _slope(values: np.ndarray) -> float:
    """Return a simple linear slope over this trailing segment."""
    x = np.arange(values.size, dtype=np.float64)
    return float(np.polyfit(x, values, 1)[0]) if values.size > 1 else 0.0


def split_by_unit(
    frame: pd.DataFrame, validation_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split complete engine trajectories, with a fixed seeded assignment."""
    units = np.array(sorted(frame["unit_id"].unique()))
    if units.size < 2 or not 0 < validation_fraction < 1:
        raise ValueError("Need at least two engines and a validation fraction between zero and one")
    rng = np.random.default_rng(SEED)
    validation_units = set(
        rng.permutation(units)[: max(1, round(units.size * validation_fraction))]
    )
    train = frame.loc[~frame["unit_id"].isin(validation_units)].copy()
    validation = frame.loc[frame["unit_id"].isin(validation_units)].copy()
    if set(train.unit_id) & set(validation.unit_id):
        raise AssertionError("Engine leakage between train and validation")
    return train, validation


def select_sensor_columns(
    train: pd.DataFrame, sensor_columns: list[str], variance_threshold: float = 1e-8
) -> list[str]:
    """Keep channels whose raw training-engine variance exceeds the fixed threshold."""
    variances = train[sensor_columns].var()
    selected = [column for column in sensor_columns if variances[column] > variance_threshold]
    if not selected:
        raise ValueError("Training data has no sensors above the near-constant threshold")
    return selected


def fit_scaler(
    train: pd.DataFrame, validation: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit scaling only on training rows and transform both splits."""
    scaler = StandardScaler().fit(train[columns])
    train_scaled, validation_scaled = train.copy(), validation.copy()
    train_scaled.loc[:, columns] = scaler.transform(train[columns])
    validation_scaled.loc[:, columns] = scaler.transform(validation[columns])
    return train_scaled, validation_scaled, scaler


def make_windows(
    cycles: pd.DataFrame, window_size: int, feature_columns: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build trailing fixed-size windows, left-padding early cycles with their first row."""
    xs: list[np.ndarray] = []
    ys: list[float] = []
    units: list[int] = []
    for unit_id, engine in cycles.sort_values(["unit_id", "cycle"]).groupby("unit_id"):
        values = engine[feature_columns].to_numpy(dtype=np.float32)
        labels = engine["rul"].to_numpy(dtype=np.float32)
        for index in range(values.shape[0]):
            start = max(0, index - window_size + 1)
            segment = values[start : index + 1]
            if segment.shape[0] < window_size:
                segment = np.concatenate(
                    [np.repeat(segment[:1], window_size - len(segment), axis=0), segment]
                )
            xs.append(segment)
            ys.append(float(labels[index]))
            units.append(int(unit_id))
    return np.stack(xs), np.asarray(ys), np.asarray(units)
