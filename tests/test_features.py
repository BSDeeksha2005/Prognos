"""Tests for leakage controls and causal feature generation."""

import pandas as pd

from rul.config import FEATURE_COLUMNS
from rul.features import engineer_features, make_windows, select_sensor_columns, split_by_unit


def test_unit_split_has_no_overlap() -> None:
    """An engine belongs to only one split."""
    frame = pd.DataFrame({"unit_id": [1, 1, 2, 2, 3, 3, 4, 4], "cycle": range(1, 9)})
    train, validation = split_by_unit(frame)
    assert set(train.unit_id).isdisjoint(set(validation.unit_id))


def test_rolling_features_use_current_and_past_only() -> None:
    """Changing future values cannot alter features already computed."""
    frame = pd.DataFrame(0.0, index=range(3), columns=FEATURE_COLUMNS)
    frame.insert(0, "cycle", [1, 2, 3])
    frame.insert(0, "unit_id", [1, 1, 1])
    frame["sensor_1"] = [2.0, 4.0, 8.0]
    baseline = engineer_features(frame, window=2)
    changed = frame.copy()
    changed.loc[2, "sensor_1"] = 800.0
    updated = engineer_features(changed, window=2)
    assert baseline.loc[:1, "sensor_1_mean"].tolist() == updated.loc[:1, "sensor_1_mean"].tolist()


def test_windows_are_engine_local_and_padded() -> None:
    """Each sequence has a fixed length and the first engine row is left-padded."""
    frame = pd.DataFrame(
        {"unit_id": [1, 1, 2], "cycle": [1, 2, 1], "x": [3.0, 4.0, 9.0], "rul": [1, 0, 0]}
    )
    values, labels, units = make_windows(frame, 3, ["x"])
    assert values.shape == (3, 3, 1)
    assert values[0, :, 0].tolist() == [3.0, 3.0, 3.0]
    assert labels.tolist() == [1.0, 0.0, 0.0]
    assert units.tolist() == [1, 1, 2]


def test_sensor_filter_uses_training_variance() -> None:
    """Constant and near-constant channels are removed at the stated threshold."""
    frame = pd.DataFrame(
        {"s1": [1.0, 1.0, 1.0], "s2": [0.0, 1.0, 2.0], "s3": [1.0, 1.0 + 1e-6, 1.0]}
    )
    assert select_sensor_columns(frame, ["s1", "s2", "s3"]) == ["s2"]
