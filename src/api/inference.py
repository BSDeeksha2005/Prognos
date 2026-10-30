"""ONNX Runtime wrapper loaded once when the service starts."""

from pathlib import Path

import numpy as np
import onnxruntime as ort


class Predictor:
    """Run ONNX inference for one fixed-size sensor window."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        preprocessing_path = model_path.with_name("nn_preprocessing.npz")
        with np.load(preprocessing_path) as preprocessing:
            self.mean = preprocessing["mean"]
            self.scale = preprocessing["scale"]
            self.sensor_indices = preprocessing["indices"].astype(int)

    def predict(self, window: list[list[float]]) -> float:
        """Predict RUL for a chronological configured-length by 21 sensor window."""
        raw_values = np.asarray(window, dtype=np.float32)[:, self.sensor_indices]
        standardized = (raw_values - self.mean) / self.scale
        values = standardized.T[None, :, :]
        result = self.session.run(["rul"], {"sensor_window": values})[0]
        return max(0.0, float(result.reshape(-1)[0]))
