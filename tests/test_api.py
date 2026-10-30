"""API payload and model-unavailable behavior."""

import json

from fastapi.testclient import TestClient

from api.main import app


def test_health_and_model_info_without_artifact() -> None:
    """The service reports unavailable weights without pretending to be ready."""
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/model-info").status_code == 200
        dashboard = client.get("/api/dashboard")
        assert dashboard.status_code == 200
        payload = dashboard.json()
        assert payload["state"] in {"awaiting_data", "files_ready", "model_ready", "evaluated"}
        assert [file["name"] for file in payload["required_files"]] == [
            "train_FD001.txt",
            "test_FD001.txt",
            "RUL_FD001.txt",
        ]
        assert isinstance(payload["sensor_drift"], dict)


def test_product_console_is_served_by_api() -> None:
    """Operators can open the dashboard from the same-origin service."""
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Remaining life," in response.text
    assert 'data-view="engines"' in response.text
    assert 'data-view="comparison"' in response.text
    assert 'data-view="models"' in response.text
    assert 'data-view="quality"' in response.text


def test_api_rejects_wrong_sensor_count() -> None:
    """Every input cycle must contain exactly 21 sensors."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"window": [[0.0] * 20 for _ in range(30)]})
    assert response.status_code == 422


def test_api_rejects_wrong_window_length() -> None:
    """Inference only accepts the documented 30-cycle window."""
    with TestClient(app) as client:
        response = client.post("/predict", json={"window": [[0.0] * 21 for _ in range(29)]})
    assert response.status_code == 422


def test_api_rejects_non_finite_sensor_value() -> None:
    """NaN and infinity cannot enter model inference."""
    payload = {"window": [[0.0] * 21 for _ in range(30)]}
    payload["window"][0][0] = float("nan")
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=json.dumps(payload, allow_nan=True),
            headers={"Content-Type": "application/json"},
        )
    assert response.status_code == 422
