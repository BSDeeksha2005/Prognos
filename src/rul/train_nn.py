"""Run a small seeded hyperparameter comparison and retain the best validation model."""

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from rul.config import SEED
from rul.nn_model import RULConvNet


def seed_everything(seed: int = SEED) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_experiment(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    parameters: dict[str, float | int],
    output_path: Path,
    epochs: int = 40,
) -> tuple[RULConvNet, float]:
    """Train with validation early stopping and save the best weights."""
    seed_everything()
    model = RULConvNet(
        train_x.shape[2], int(parameters["hidden_size"]), float(parameters["dropout"])
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(parameters["learning_rate"]))
    loss_fn = nn.MSELoss()
    train_loader = DataLoader(
        TensorDataset(torch.tensor(train_x).transpose(1, 2), torch.tensor(train_y)),
        batch_size=128,
        shuffle=True,
    )
    val_tensor = torch.tensor(val_x).transpose(1, 2)
    best_loss, best_weights, stale = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            loss_fn(model(batch_x), batch_y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss_fn(model(val_tensor), torch.tensor(val_y)))
        if validation_loss < best_loss:
            best_loss, best_weights, stale = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= 5:
                break
    model.load_state_dict(best_weights)
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return model, float(np.sqrt(best_loss))


def experiment_grid() -> list[dict[str, float | int]]:
    """Return five deliberately small, reproducible experiments."""
    return [
        {"window_size": 30, "hidden_size": 16, "dropout": 0.1, "learning_rate": 0.001},
        {"window_size": 30, "hidden_size": 32, "dropout": 0.2, "learning_rate": 0.001},
        {"window_size": 30, "hidden_size": 64, "dropout": 0.3, "learning_rate": 0.0005},
        {"window_size": 20, "hidden_size": 32, "dropout": 0.1, "learning_rate": 0.0005},
        {"window_size": 40, "hidden_size": 32, "dropout": 0.3, "learning_rate": 0.002},
    ]
