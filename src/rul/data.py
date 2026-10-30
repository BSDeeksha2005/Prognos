"""Read and label NASA C-MAPSS plain text files."""

from pathlib import Path

import pandas as pd

from rul.config import RAW_COLUMNS, RUL_CAP


def load_cycles(path: Path) -> pd.DataFrame:
    """Load one headerless, whitespace-separated cycle file."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Put the requested NASA FD001 file there.")
    frame = pd.read_csv(path, sep=r"\s+", header=None, names=RAW_COLUMNS, engine="python")
    if frame.shape[1] != 26 or frame.empty:
        raise ValueError(f"Expected a non-empty 26-column C-MAPSS file: {path}")
    if frame.isna().any().any():
        raise ValueError(f"Unexpected missing values in {path}")
    return frame.sort_values(["unit_id", "cycle"]).reset_index(drop=True)


def add_training_rul(cycles: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """Label each observed training cycle using its engine's last cycle."""
    result = cycles.copy()
    last_cycle = result.groupby("unit_id")["cycle"].transform("max")
    result["rul"] = (last_cycle - result["cycle"]).clip(upper=cap)
    return result


def load_test_rul(path: Path) -> pd.Series:
    """Load the terminal-cycle RUL truth values in file order."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Put RUL_FD001.txt there.")
    values = pd.read_csv(path, sep=r"\s+", header=None, usecols=[0], engine="python").iloc[:, 0]
    if values.empty or values.isna().any():
        raise ValueError(f"Expected non-empty RUL values in {path}")
    return values.clip(upper=RUL_CAP).reset_index(drop=True)
