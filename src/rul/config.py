"""Shared, explicit project configuration."""

from pathlib import Path

SEED = 42
RUL_CAP = 125
SENSOR_COUNT = 21
SETTING_COUNT = 3
RAW_DIR = Path("data/raw")
MODEL_DIR = Path("models")
WINDOW_SIZE = 30
FEATURE_COLUMNS = [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
RAW_COLUMNS = ["unit_id", "cycle", *FEATURE_COLUMNS]
