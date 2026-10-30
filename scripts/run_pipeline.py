# ruff: noqa: E402  # Check for the required data before importing heavy ML dependencies.
"""Train, validate, compare and export models on FD001."""

import json
from pathlib import Path

DATA_DIR = Path("data/raw")
REQUIRED_DATA = [
    DATA_DIR / "train_FD001.txt",
    DATA_DIR / "test_FD001.txt",
    DATA_DIR / "RUL_FD001.txt",
]
missing_data = [path for path in REQUIRED_DATA if not path.is_file()]
if missing_data:
    missing_names = ", ".join(path.name for path in missing_data)
    raise SystemExit(
        f"Missing {missing_names}. Download NASA's C-MAPSS Turbofan Engine Degradation "
        "Simulation Data Set archive from the NASA PCoE repository, then extract the FD001 "
        "files from CMaps/ into data/raw/."
    )

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch

from rul.baseline import train_baselines
from rul.config import MODEL_DIR, RAW_DIR, SENSOR_COLUMNS, WINDOW_SIZE
from rul.data import add_training_rul, load_cycles, load_test_rul
from rul.drift import flag_drift
from rul.export_onnx import export_and_check
from rul.features import (
    engineer_features,
    fit_scaler,
    make_windows,
    select_sensor_columns,
    split_by_unit,
)
from rul.metrics import nasa_score, rmse
from rul.train_nn import experiment_grid, fit_experiment


def main() -> None:
    """Execute the real-data workflow without substituting synthetic results."""
    mlflow.set_tracking_uri("file:./mlruns")
    MODEL_DIR.mkdir(exist_ok=True)
    train_path, test_path, truth_path = (
        RAW_DIR / f"{name}_FD001.txt" for name in ("train", "test", "RUL")
    )
    train_raw, test_raw, truth = (
        load_cycles(train_path),
        load_cycles(test_path),
        load_test_rul(truth_path),
    )
    labeled = engineer_features(add_training_rul(train_raw))
    train, validation = split_by_unit(labeled)
    engineered_columns = [
        column for column in train.columns if column not in {"unit_id", "cycle", "rul"}
    ]
    raw_train = labeled.loc[labeled.unit_id.isin(train.unit_id)].copy()
    raw_validation = labeled.loc[labeled.unit_id.isin(validation.unit_id)].copy()
    active_sensors = select_sensor_columns(raw_train, SENSOR_COLUMNS)
    sensor_drift = flag_drift(
        {sensor: raw_train[sensor].to_numpy() for sensor in active_sensors},
        {sensor: test_raw[sensor].to_numpy() for sensor in active_sensors},
    )
    inactive_sensors = set(SENSOR_COLUMNS) - set(active_sensors)
    inactive_features = {
        feature
        for feature in engineered_columns
        if any(feature == sensor or feature.startswith(f"{sensor}_") for sensor in inactive_sensors)
    }
    feature_variances = train[engineered_columns].var()
    columns = [
        feature
        for feature in engineered_columns
        if feature not in inactive_features and feature_variances[feature] > 1e-8
    ]
    if not columns:
        raise ValueError("No baseline features remain after train-only variance filtering")
    train, validation, scaler = fit_scaler(train, validation, columns)
    test = engineer_features(test_raw)
    test.loc[:, columns] = scaler.transform(test[columns])
    terminal_count = test.groupby("unit_id").ngroups
    if terminal_count != len(truth):
        raise ValueError(f"Test engines ({terminal_count}) do not match RUL labels ({len(truth)})")
    scores = train_baselines(train, validation, test, columns, truth.to_numpy())
    raw_test = test_raw.copy()
    raw_train, raw_validation, raw_scaler = fit_scaler(raw_train, raw_validation, active_sensors)
    raw_test.loc[:, active_sensors] = raw_scaler.transform(raw_test[active_sensors])
    terminal_test = (
        raw_test.sort_values(["unit_id", "cycle"]).groupby("unit_id", sort=True).tail(1).copy()
    )
    terminal_test["rul"] = 0.0
    best_validation = float("inf")
    best_model: torch.nn.Module | None = None
    best_window = WINDOW_SIZE
    best_parameters: dict[str, float | int] | None = None
    experiment_results: list[dict[str, float | int]] = []
    for experiment_index, parameters in enumerate(experiment_grid(), start=1):
        size = int(parameters["window_size"])
        experiment_train_x, experiment_train_y, _ = make_windows(raw_train, size, active_sensors)
        experiment_validation_x, experiment_validation_y, _ = make_windows(
            raw_validation, size, active_sensors
        )
        with mlflow.start_run(run_name=f"cnn-{experiment_index}"):
            model, validation_rmse = fit_experiment(
                experiment_train_x,
                experiment_train_y,
                experiment_validation_x,
                experiment_validation_y,
                parameters,
                MODEL_DIR / f"experiment-{experiment_index}.pt",
            )
            mlflow.log_params(parameters)
            mlflow.log_metric("validation_rmse", validation_rmse)
            experiment_results.append(
                {"run": experiment_index, **parameters, "validation_rmse": validation_rmse}
            )
            if validation_rmse < best_validation:
                best_validation, best_model, best_window, best_parameters = (
                    validation_rmse,
                    model,
                    size,
                    parameters,
                )
    if best_model is None or best_parameters is None:
        raise RuntimeError("No neural experiment produced a model")
    torch.save(best_model.state_dict(), MODEL_DIR / "best.pt")
    test_x, _, _ = make_windows(terminal_test, best_window, active_sensors)
    with torch.no_grad():
        nn_prediction = best_model(torch.tensor(test_x).transpose(1, 2)).numpy()
    scores["cnn"] = {
        "test_rmse": rmse(truth.to_numpy(), nn_prediction),
        "test_nasa": nasa_score(truth.to_numpy(), nn_prediction),
        "validation_rmse": best_validation,
    }
    with mlflow.start_run(run_name="cnn-final-test"):
        mlflow.log_params(best_parameters)
        mlflow.log_metrics(
            {
                "test_rmse": scores["cnn"]["test_rmse"],
                "test_nasa": scores["cnn"]["test_nasa"],
                "validation_rmse": best_validation,
            }
        )
    pd.DataFrame(scores).T.to_csv(MODEL_DIR / "results.csv")
    errors = nn_prediction - truth.to_numpy()
    plt.figure(figsize=(8, 5))
    plt.scatter(truth, errors, alpha=0.7)
    plt.axhline(0, color="black", linewidth=1)
    plt.xlabel("True remaining useful life (cycles)")
    plt.ylabel("Prediction error (cycles)")
    plt.title("FD001 test error vs. remaining life")
    plt.tight_layout()
    plt.savefig(MODEL_DIR / "error_vs_rul.png", dpi=160)
    parity = export_and_check(
        MODEL_DIR / "best.pt",
        MODEL_DIR / "best.onnx",
        len(active_sensors),
        best_window,
        int(best_parameters["hidden_size"]),
        float(best_parameters["dropout"]),
    )
    (MODEL_DIR / "model_info.json").write_text(
        json.dumps(
            {
                "window_length": best_window,
                "sensor_count": len(SENSOR_COLUMNS),
                "active_sensor_count": len(active_sensors),
                "active_sensors": active_sensors,
                "model": "Prognos 1D-CNN",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    sensor_indices = [SENSOR_COLUMNS.index(sensor) for sensor in active_sensors]
    np.savez(
        MODEL_DIR / "nn_preprocessing.npz",
        mean=raw_scaler.mean_,
        scale=raw_scaler.scale_,
        indices=sensor_indices,
    )
    engine_predictions = [
        {
            "unit_id": int(row.unit_id),
            "last_cycle": int(row.cycle),
            "predicted_rul": float(np.clip(prediction, 0, 125)),
            "actual_rul": float(actual),
        }
        for row, prediction, actual in zip(
            terminal_test.itertuples(), nn_prediction, truth.to_numpy(), strict=True
        )
    ]
    dashboard_report = {
        "dataset": "NASA C-MAPSS FD001",
        "metrics": scores,
        "engines": engine_predictions,
        "sensors_flagged": sum(bool(result["drifted"]) for result in sensor_drift.values()),
        "sensor_drift": sensor_drift,
        "run_metadata": {
            "validation_engine_count": int(validation.unit_id.nunique()),
            "test_engine_count": int(len(engine_predictions)),
            "window_size": best_window,
            "onnx_max_abs_error": parity,
            "experiments": experiment_results,
        },
    }
    (MODEL_DIR / "dashboard.json").write_text(
        json.dumps(dashboard_report, indent=2), encoding="utf-8"
    )
    print(pd.DataFrame(scores).T.to_string())
    print(
        f"Best CNN window={best_window}, validation RMSE={best_validation:.3f}, "
        f"ONNX parity max abs error={parity:.2e}"
    )
    rmse_comparison = (
        "beats" if scores["cnn"]["test_rmse"] < scores["lightgbm"]["test_rmse"] else "does not beat"
    )
    nasa_comparison = (
        "beats" if scores["cnn"]["test_nasa"] < scores["lightgbm"]["test_nasa"] else "does not beat"
    )
    print(
        f"On FD001 test metrics, the CNN {rmse_comparison} LightGBM on RMSE and "
        f"{nasa_comparison} LightGBM on NASA score."
    )
    print(
        f"Retained {len(columns)} baseline features and "
        f"{len(active_sensors)}/{len(SENSOR_COLUMNS)} sensors after train-only variance filtering"
    )
    np.savez(
        MODEL_DIR / "preprocessing.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        features=np.asarray(columns),
    )


if __name__ == "__main__":
    main()
