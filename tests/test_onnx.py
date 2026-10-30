"""ONNX export parity for a known initialized model."""

import torch

from rul.export_onnx import export_and_check
from rul.nn_model import RULConvNet


def test_onnx_runtime_matches_pytorch(tmp_path) -> None:
    """The exported graph preserves predictions for the same window."""
    weights_path = tmp_path / "model.pt"
    output_path = tmp_path / "model.onnx"
    torch.manual_seed(7)
    model = RULConvNet(sensor_count=21, hidden_size=16, dropout=0.1)
    torch.save(model.state_dict(), weights_path)
    error = export_and_check(
        weights_path, output_path, sensor_count=21, window_size=30, hidden_size=16, dropout=0.1
    )
    assert output_path.is_file()
    assert error <= 1e-5
