"""Compact one-dimensional convolutional network for trailing sensor windows."""

import torch
from torch import nn


class RULConvNet(nn.Module):
    """Predict one non-negative RUL value from channels-first sensor history."""

    def __init__(self, sensor_count: int, hidden_size: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(sensor_count, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_size, 1),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        """Return one RUL estimate per input window."""
        return self.network(values).squeeze(-1)
