"""Export a trained CNN to ONNX and check runtime parity."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from rul.nn_model import RULConvNet


def export_and_check(
    weights_path: Path,
    output_path: Path,
    sensor_count: int,
    window_size: int,
    hidden_size: int = 32,
    dropout: float = 0.2,
) -> float:
    """Export trained weights and return maximum absolute parity error."""
    model = RULConvNet(sensor_count, hidden_size, dropout)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    example = torch.zeros(1, sensor_count, window_size)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["sensor_window"],
        output_names=["rul"],
        dynamo=False,
        opset_version=17,
    )
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    torch_output = model(example).detach().numpy()
    onnx_output = session.run(["rul"], {"sensor_window": example.numpy()})[0]
    error = float(np.max(np.abs(torch_output - onnx_output)))
    if error > 1e-5:
        raise AssertionError(f"ONNX parity error too large: {error}")
    return error
