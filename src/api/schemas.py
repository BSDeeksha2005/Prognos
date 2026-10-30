"""Validated API payloads."""

import math
import os

from pydantic import BaseModel, field_validator


class PredictionRequest(BaseModel):
    """Trailing sensor history, oldest observation first."""

    window: list[list[float]]

    @field_validator("window")
    @classmethod
    def validate_rows(cls, rows: list[list[float]]) -> list[list[float]]:
        """Require exactly 21 finite sensor readings at every cycle."""
        expected_length = int(os.environ.get("PROGNOS_WINDOW_SIZE", "30"))
        if len(rows) != expected_length:
            raise ValueError(f"Window must contain exactly {expected_length} cycles")
        if any(len(row) != 21 for row in rows):
            raise ValueError("Each cycle must have exactly 21 sensor values")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise ValueError("Sensor values must be finite")
        return rows


class PredictionResponse(BaseModel):
    """Single-engine inference result."""

    rul: float
    model: str


class HealthResponse(BaseModel):
    """Service readiness."""

    status: str
