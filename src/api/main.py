"""FastAPI application with a model session created at startup."""

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.inference import Predictor
from api.schemas import HealthResponse, PredictionRequest, PredictionResponse

MODEL_PATH = Path(os.environ.get("PROGNOS_MODEL", "models/best.onnx"))
INFO_PATH = MODEL_PATH.with_name("model_info.json")
PROJECT_DIR = Path(os.environ.get("PROGNOS_PROJECT_DIR", Path(__file__).resolve().parents[2]))
FRONTEND_DIR = PROJECT_DIR / "frontend"
DATA_DIR = PROJECT_DIR / "data" / "raw"
RESULTS_PATH = MODEL_PATH.with_name("dashboard.json")
MODEL_INFO: dict[str, object] = {
    "window_length": int(os.environ.get("PROGNOS_WINDOW_SIZE", "30")),
    "sensor_count": 21,
}
if INFO_PATH.is_file():
    MODEL_INFO.update(json.loads(INFO_PATH.read_text(encoding="utf-8")))
os.environ["PROGNOS_WINDOW_SIZE"] = str(MODEL_INFO["window_length"])
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model once, and provide a useful startup error when absent."""
    global predictor
    if MODEL_PATH.is_file():
        predictor = Predictor(MODEL_PATH)
    yield
    predictor = None


app = FastAPI(title="Prognos RUL API", version="0.1.0", lifespan=lifespan)
app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.exception_handler(RequestValidationError)
async def validation_error_response(
    _: Request, error: RequestValidationError
) -> JSONResponse:
    """Return safe 422 details without echoing invalid sensor data."""
    details = [
        {key: value for key, value in item.items() if key not in {"input", "ctx"}}
        for item in error.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": details})


@app.get("/", include_in_schema=False)
def dashboard_page() -> FileResponse:
    """Serve the same-origin operations dashboard."""
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Expose readiness without loading or modifying model state."""
    return HealthResponse(status="ready" if predictor else "model_missing")


@app.get("/model-info")
def model_info() -> dict[str, object]:
    """Expose model artifact metadata."""
    return {
        "name": "Prognos 1D-CNN",
        "format": "ONNX",
        **MODEL_INFO,
        "artifact": str(MODEL_PATH),
        "available": MODEL_PATH.is_file(),
    }


@app.get("/api/dashboard")
def dashboard_data() -> dict[str, object]:
    """Return final benchmark metrics and terminal test predictions for the console."""
    required_files = [
        DATA_DIR / "train_FD001.txt",
        DATA_DIR / "test_FD001.txt",
        DATA_DIR / "RUL_FD001.txt",
    ]
    files_present = all(path.is_file() for path in required_files)
    required_file_status = [
        {"name": path.name, "available": path.is_file()} for path in required_files
    ]
    report: dict[str, object] = {}
    if RESULTS_PATH.is_file():
        report = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    state = (
        "evaluated"
        if report.get("engines")
        else "model_ready"
        if predictor
        else "files_ready"
        if files_present
        else "awaiting_data"
    )
    return {
        "state": state,
        "files_present": files_present,
        "required_files": required_file_status,
        "model_available": predictor is not None,
        "active_sensor_count": MODEL_INFO.get("active_sensor_count"),
        "metrics": report.get("metrics", {}),
        "engines": report.get("engines", []),
        "sensors_flagged": report.get("sensors_flagged"),
        "sensor_drift": report.get("sensor_drift", {}),
        "run_metadata": report.get("run_metadata", {}),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict remaining cycles for the final cycle in the supplied window."""
    if predictor is None:
        raise HTTPException(
            status_code=503, detail="Model artifact is unavailable; train and export it first."
        )
    return PredictionResponse(rul=predictor.predict(payload.window), model="Prognos 1D-CNN")
