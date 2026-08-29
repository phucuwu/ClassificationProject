"""FastAPI backend server for the local art taste classifier.

Provides REST endpoints for sample recording, prediction inference, desktop screen capture,
classifier retraining, review queue management, and dataset metrics.
"""

from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any

import mss
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from backend.database import (
    DATA_DIR,
    DEFAULT_DB_PATH,
    IMAGES_DIR,
    get_dataset_statistics,
    get_samples,
    init_db,
    insert_sample,
    load_training_matrix,
    update_sample_reviews,
)
from backend.model import (
    DEFAULT_DECISION_THRESHOLD,
    extract_vision_embedding,
    load_classifier,
    predict_taste,
    train_taste_classifier,
)

from contextlib import asynccontextmanager

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize database and directories on server startup."""
    init_db()
    yield


from datetime import datetime

# In-memory circular activity log buffer (holds last 500 events)
MAX_LOGS = 500
ACTIVITY_LOGS: list[dict[str, Any]] = []


def add_activity_log(
    level: str,
    event: str,
    message: str,
    mode: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Record a system or interaction event in the activity log."""
    log_entry = {
        "id": len(ACTIVITY_LOGS) + 1,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level.upper(),  # 'INFO', 'SUCCESS', 'WARNING', 'ERROR'
        "event": event,
        "mode": mode,
        "message": message,
        "details": details or {},
    }
    ACTIVITY_LOGS.append(log_entry)
    if len(ACTIVITY_LOGS) > MAX_LOGS:
        ACTIVITY_LOGS.pop(0)


# Initialize FastAPI application
app = FastAPI(
    title="Local Art Taste Classifier API",
    description="Backend API for local art taste classification, model training, and dataset review.",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS to allow local browser userscripts and localhost web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/logs")
def get_activity_logs(
    limit: int = Query(100, ge=1, le=500, description="Max logs to return."),
    level: str | None = Query(None, description="Filter by level: INFO, SUCCESS, WARNING, ERROR."),
    mode: str | None = Query(None, description="Filter by mode: manual, supervised, auto."),
) -> list[dict[str, Any]]:
    """Retrieve recent console activity logs for the developer dashboard."""
    filtered = ACTIVITY_LOGS
    if level:
        filtered = [l for l in filtered if l["level"] == level.upper()]
    if mode:
        filtered = [l for l in filtered if l.get("mode") == mode.lower()]

    # Return in reverse chronological order (newest first)
    return list(reversed(filtered[-limit:]))


@app.post("/api/logs/clear")
def clear_activity_logs() -> dict[str, str]:
    """Clear all console activity logs."""
    ACTIVITY_LOGS.clear()
    add_activity_log("INFO", "system", "Activity log cleared by user.")
    return {"status": "cleared"}


# -----------------------------------------------------------------------------
# Request and Response Pydantic Schemas
# -----------------------------------------------------------------------------

class LogRequest(BaseModel):
    level: str = Field("INFO", description="Log level: INFO, SUCCESS, WARNING, ERROR.")
    event: str = Field("userscript", description="Event name identifier.")
    message: str = Field(..., description="Message text to display in console.")
    mode: str | None = Field(None, description="Operating mode ('manual', 'supervised', 'auto').")
    details: dict[str, Any] | None = Field(None, description="Optional extra details dictionary.")


class RecordRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string or data URI.")
    label: int | None = Field(None, description="1 for Like (positive class), 0 for Dislike, or null for unlabeled.")
    mode: str = Field("manual", description="Operating mode: 'manual', 'supervised', or 'auto'.")
    prediction_score: float | None = Field(None, description="Model prediction score P(Like) between 0.0 and 1.0.")
    reviewed: int = Field(0, description="1 if confirmed by human, 0 if pending review.")
    image_set_count: int | None = Field(None, description="Total photos detected in the active card's image set.")


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string or data URI.")
    threshold: float | None = Field(None, description="Optional override for decision threshold.")


class CaptureRequest(BaseModel):
    x: int = Field(..., description="Top-left X coordinate on desktop screen.")
    y: int = Field(..., description="Top-left Y coordinate on desktop screen.")
    width: int = Field(..., description="Width of screen capture region in pixels.")
    height: int = Field(..., description="Height of screen capture region in pixels.")


class TrainRequest(BaseModel):
    target_recall: float = Field(0.90, description="Target recall rate for decision threshold calibration.")


class ReviewUpdateItem(BaseModel):
    id: int = Field(..., description="Sample database identifier.")
    label: int = Field(..., description="Updated label (1 for Like, 0 for Dislike).")
    reviewed: int = Field(1, description="Review confirmation status flag.")


class ReviewRequest(BaseModel):
    updates: list[ReviewUpdateItem] = Field(..., description="List of sample review updates.")


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.post("/api/log")
def create_activity_log(payload: LogRequest) -> dict[str, str]:
    """Ingest an activity log message from the userscript for the dashboard console."""
    add_activity_log(
        level=payload.level,
        event=payload.event,
        message=payload.message,
        mode=payload.mode,
        details=payload.details,
    )
    return {"status": "logged"}


def _decode_and_save_image(image_base64: str) -> tuple[bytes, str, Path]:
    """Decode base64 string, calculate SHA-256 hash, and save JPEG to disk."""
    raw_str = image_base64
    if "," in raw_str:
        raw_str = raw_str.split(",", 1)[1]

    image_bytes = base64.b64decode(raw_str)
    image_hash = hashlib.sha256(image_bytes).hexdigest()

    import backend.database as _db
    images_dir = Path(_db.IMAGES_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)
    file_path = images_dir / f"{image_hash}.jpg"

    if not file_path.exists():
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_image.save(file_path, format="JPEG", quality=95)

    return image_bytes, image_hash, file_path


@app.post("/api/record", status_code=status.HTTP_201_CREATED)
def record_sample(payload: RecordRequest) -> dict[str, Any]:
    """Ingest a sample with an image, label, and mode. Extracts and saves the vision embedding."""
    try:
        image_bytes, image_hash, file_path = _decode_and_save_image(payload.image_base64)
        embedding = extract_vision_embedding(image_bytes)

        rel_path = f"data/images/{image_hash}.jpg"
        sample_id = insert_sample(
            image_hash=image_hash,
            file_path=rel_path,
            embedding=embedding,
            label=payload.label,
            mode=payload.mode,
            prediction_score=payload.prediction_score,
            reviewed=payload.reviewed,
        )

        label_str = "LIKE (1)" if payload.label == 1 else ("DISLIKE (0)" if payload.label == 0 else "UNLABELED")
        set_str = f" [Set: {payload.image_set_count} photos]" if payload.image_set_count and payload.image_set_count > 1 else ""

        if payload.mode == "manual":
            add_activity_log(
                "SUCCESS",
                "sample_recorded",
                f"Manual rating recorded: Sample #{sample_id} ({image_hash[:8]}...) as {label_str}{set_str}",
                mode="manual",
                details={"id": sample_id, "label": payload.label, "hash": image_hash[:8], "image_set_count": payload.image_set_count},
            )
        elif payload.mode == "auto":
            score_str = f"{payload.prediction_score:.2f}" if payload.prediction_score is not None else "N/A"
            add_activity_log(
                "INFO",
                "auto_decision",
                f"Auto decision: Sample #{sample_id} ({image_hash[:8]}...) rated as {label_str} (Score: {score_str}){set_str}",
                mode="auto",
                details={"id": sample_id, "label": payload.label, "score": payload.prediction_score, "image_set_count": payload.image_set_count},
            )
        else:
            add_activity_log(
                "SUCCESS",
                "supervised_confirmed",
                f"Supervised rating: Sample #{sample_id} ({image_hash[:8]}...) saved as {label_str}{set_str}",
                mode="supervised",
                details={"id": sample_id, "label": payload.label, "image_set_count": payload.image_set_count},
            )

        return {
            "status": "success",
            "id": sample_id,
            "image_hash": image_hash,
            "label": payload.label,
        }
    except Exception as exc:
        add_activity_log("ERROR", "record_failed", f"Failed to record sample: {str(exc)}", mode=payload.mode)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to record sample: {str(exc)}",
        ) from exc


@app.post("/api/predict")
def predict_sample(payload: PredictRequest) -> dict[str, Any]:
    """Extract vision embedding for an image and return prediction score and binary decision."""
    try:
        embedding = extract_vision_embedding(payload.image_base64)
        result = predict_taste(embedding, threshold=payload.threshold)

        if result.get("model_loaded"):
            score = result.get("prediction_score")
            decision_str = "LIKE (1)" if result.get("decision") == 1 else "DISLIKE (0)"
            thresh = result.get("threshold")
            add_activity_log(
                "INFO",
                "prediction",
                f"Prediction inference: P(Like) = {score} (θ={thresh}) -> {decision_str}",
                details=result,
            )
        else:
            add_activity_log("WARNING", "prediction_cold_start", "Inference requested but model is not trained yet.")

        return result
    except Exception as exc:
        add_activity_log("ERROR", "prediction_failed", f"Inference failed: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(exc)}",
        ) from exc


@app.post("/api/capture")
def capture_screen_region(payload: CaptureRequest) -> dict[str, Any]:
    """Capture an OS desktop screen region using mss and return a base64 image."""
    try:
        with mss.mss() as sct:
            monitor = {
                "left": int(payload.x),
                "top": int(payload.y),
                "width": int(payload.width),
                "height": int(payload.height),
            }
            sct_img = sct.grab(monitor)
            pil_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

            buffer = io.BytesIO()
            pil_img.save(buffer, format="JPEG", quality=95)
            b64_str = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")

            add_activity_log("INFO", "screen_capture", f"Screen capture fallback: {payload.width}x{payload.height} px at ({payload.x}, {payload.y})")
            return {
                "status": "captured",
                "image_base64": b64_str,
            }
    except Exception as exc:
        add_activity_log("ERROR", "screen_capture_failed", f"Screen capture failed: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Screen capture failed: {str(exc)}",
        ) from exc


@app.post("/api/train")
def train_model(payload: TrainRequest = TrainRequest()) -> dict[str, Any]:
    """Train the balanced Logistic Regression model on all labeled samples in the database."""
    try:
        X, y = load_training_matrix()
        result = train_taste_classifier(X, y, target_recall=payload.target_recall)

        if result.get("status") == "trained":
            m = result.get("metrics", {})
            add_activity_log(
                "SUCCESS",
                "model_trained",
                f"Model retrained successfully on {result.get('sample_count')} samples (PR-AUC: {m.get('pr_auc')}, Recall: {m.get('recall')}, θ: {m.get('decision_threshold')})",
                details=m,
            )
        else:
            add_activity_log("WARNING", "train_insufficient_data", result.get("message", "Insufficient samples to train."))

        return result
    except Exception as exc:
        add_activity_log("ERROR", "train_failed", f"Training failed: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model training failed: {str(exc)}",
        ) from exc


@app.get("/api/samples")
def query_samples(
    mode: str | None = Query(None, description="Filter by mode: 'manual', 'supervised', or 'auto'."),
    reviewed: int | None = Query(None, description="Filter by review status: 1 or 0."),
    label: int | None = Query(None, description="Filter by label: 1 or 0."),
    limit: int = Query(50, ge=1, le=200, description="Max number of samples to return."),
    offset: int = Query(0, ge=0, description="Query offset."),
) -> list[dict[str, Any]]:
    """Query samples for the dashboard review queue with base64 encoded images included."""
    samples = get_samples(mode=mode, reviewed=reviewed, label=label, limit=limit, offset=offset)

    import backend.database as _db
    images_dir = Path(_db.IMAGES_DIR)

    results: list[dict[str, Any]] = []
    for sample in samples:
        raw_path = Path(sample["file_path"])
        if raw_path.is_absolute() and raw_path.exists():
            resolved_path = raw_path
        elif (PROJECT_ROOT / raw_path).exists():
            resolved_path = PROJECT_ROOT / raw_path
        elif (images_dir / raw_path.name).exists():
            resolved_path = images_dir / raw_path.name
        else:
            resolved_path = None

        b64_image = None
        if resolved_path and resolved_path.exists():
            try:
                with open(resolved_path, "rb") as img_file:
                    b64_image = "data:image/jpeg;base64," + base64.b64encode(img_file.read()).decode("utf-8")
            except Exception:
                b64_image = None

        sample_data = dict(sample)
        sample_data["image_base64"] = b64_image
        results.append(sample_data)

    return results


@app.post("/api/review")
def review_samples(payload: ReviewRequest) -> dict[str, Any]:
    """Bulk update labels and mark samples as reviewed."""
    try:
        updates = [item.dict() for item in payload.updates]
        updated_count = update_sample_reviews(updates)
        return {
            "status": "success",
            "updated_count": updated_count,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Review update failed: {str(exc)}",
        ) from exc


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    """Return dataset statistics, class balance, and active model metrics."""
    try:
        statistics = get_dataset_statistics()
        model_data = load_classifier()

        if model_data is not None:
            model_info = {
                "model_loaded": True,
                "metrics": model_data.get("metrics", {}),
                "decision_threshold": round(float(model_data.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)), 2),
                "positive_count": model_data.get("positive_count", 0),
                "negative_count": model_data.get("negative_count", 0),
            }
        else:
            model_info = {
                "model_loaded": False,
                "metrics": None,
                "decision_threshold": round(float(DEFAULT_DECISION_THRESHOLD), 2),
                "message": "Model not trained yet.",
            }

        return {
            "statistics": statistics,
            "model_status": model_info,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch metrics: {str(exc)}",
        ) from exc


# Mount static directory for developer dashboard if it exists
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
