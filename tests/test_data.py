"""Tests for C-MAPSS parsing and target construction."""

import pandas as pd
import pytest

from rul.config import RAW_COLUMNS
from rul.data import add_training_rul, load_cycles


def test_add_training_rul_clips_at_cap() -> None:
    """Early-life RUL is capped while later labels preserve cycle distance."""
    data = pd.DataFrame({"unit_id": [1, 1, 1, 2, 2], "cycle": [1, 2, 130, 3, 8]})
    result = add_training_rul(data)
    assert result.rul.tolist() == [125, 125, 0, 5, 0]


def test_load_cycles_reads_headerless_whitespace(tmp_path) -> None:
    """The loader assigns all 26 documented column names."""
    path = tmp_path / "cycles.txt"
    path.write_text(" ".join(["1", "1", *["0.5"] * 24]) + "\n", encoding="utf-8")
    result = load_cycles(path)
    assert list(result.columns) == RAW_COLUMNS
    assert len(result) == 1


def test_missing_cycle_file_has_actionable_error(tmp_path) -> None:
    """A missing input names the precise missing path."""
    with pytest.raises(FileNotFoundError, match="cycles.txt"):
        load_cycles(tmp_path / "cycles.txt")
