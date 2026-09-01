"""FastAPI backend server for the local art taste classifier.

Provides REST endpoints for sample recording, prediction inference, desktop screen capture,
classifier retraining, review queue management, and dataset metrics.
"""

from __future__ import annotations

import base64
import hashlib
import io
import os
import warnings
from pathlib import Path
from typing import Any

# Filter third-party library deprecation notices
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastapi.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

import mss
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import random

import numpy as np

import backend.database as db
from backend.database import (
    get_dataset_statistics,
    get_recent_samples,
    get_samples,
    init_db,
    insert_sample,
    find_near_duplicate,
    update_sample_record,
    load_training_matrix,
    load_embedding_scatter_data,
    update_sample_reviews,
    delete_samples,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from backend.model import (
    DEFAULT_DECISION_THRESHOLD,
    detect_positive_outliers,
    detect_negative_outliers,
    extract_vision_embedding,
    load_classifier,
    predict_taste,
    train_taste_classifier,
    update_decision_threshold,
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


@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if any(request.url.path.endswith(ext) for ext in (".html", ".css", ".js", "/", "/api/metrics")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


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
    negative_sample_rate: float = Field(0.05, ge=0.0, le=1.0, description="Audit rate for sampling automated negative decisions into review queue.")


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string or data URI.")
    threshold: float | None = Field(None, description="Optional override for decision threshold.")


class CaptureRequest(BaseModel):
    x: int = Field(..., description="Top-left X coordinate on desktop screen.")
    y: int = Field(..., description="Top-left Y coordinate on desktop screen.")
    width: int = Field(..., description="Width of screen capture region in pixels.")
    height: int = Field(..., description="Height of screen capture region in pixels.")


class TrainRequest(BaseModel):
    target_recall: float | None = Field(None, description="Optional target recall rate for decision threshold calibration.")
    threshold: float | None = Field(None, description="Optional explicit threshold override.")
    min_recall_floor: float = Field(0.70, ge=0.05, le=0.95, description="Minimum recall floor for hybrid F2 threshold calibration.")
    holdout_ratio: float = Field(0.15, ge=0.0, le=0.50, description="Fraction of most recent samples reserved for generalization holdout.")
    baseline_prompt_text: str | None = Field(None, description="Optional custom prompt text for the zero-shot reference baseline.")
    baseline_image_base64: str | None = Field(None, description="Optional base64 exemplar image for the zero-shot reference baseline.")
    reset_baseline_to_default: bool = Field(False, description="Flag to reset the reference baseline back to the default text prompt.")


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.01, le=0.99, description="Active decision threshold.")


class ReviewUpdateItem(BaseModel):
    id: int = Field(..., description="Sample database identifier.")
    label: int = Field(..., description="Updated label (1 for Like, 0 for Dislike).")
    reviewed: int = Field(1, description="Review confirmation status flag.")


class ReviewRequest(BaseModel):
    updates: list[ReviewUpdateItem] = Field(..., description="List of sample review updates.")


class BatchDeleteRequest(BaseModel):
    ids: list[int] = Field(..., description="List of sample database IDs to delete.")


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


def check_session_drift(db_path: Path | str | None = None, window_size: int = 100, min_window: int = 20) -> dict[str, Any] | None:
    """Analyze rolling window of recent samples to detect taste preference session drift.

    Emits a WARNING activity log if the positive class ratio deviates outside [0.05, 0.10].
    """
    effective_db = db_path if db_path is not None else db.DEFAULT_DB_PATH
    recent = db.get_recent_samples(limit=window_size, db_path=effective_db)
    labeled = [s for s in recent if s["label"] is not None]
    if len(labeled) < min_window:
        return None

    pos_count = sum(1 for s in labeled if s["label"] == 1)
    sample_count = len(labeled)
    pos_ratio = pos_count / sample_count

    scores = [s["prediction_score"] for s in labeled if s.get("prediction_score") is not None]
    avg_score = float(sum(scores) / len(scores)) if scores else None

    drift_detected = pos_ratio < 0.05 or pos_ratio > 0.10
    if drift_detected:
        direction = "low" if pos_ratio < 0.05 else "high"
        msg = (
            f"Taste session drift warning: rolling positive ratio is {pos_ratio:.1%} "
            f"({pos_count}/{sample_count} likes, {direction} vs expected 5%-10% range)."
        )
        if avg_score is not None:
            msg += f" Mean score: {avg_score:.2f}."
        add_activity_log(
            level="WARNING",
            event="session_drift",
            message=msg,
            details={
                "window_size": sample_count,
                "positive_count": pos_count,
                "positive_ratio": round(pos_ratio, 4),
                "mean_prediction_score": round(avg_score, 4) if avg_score is not None else None,
                "expected_range": [0.05, 0.10],
            },
        )

    return {
        "window_size": sample_count,
        "positive_count": pos_count,
        "positive_ratio": pos_ratio,
        "mean_prediction_score": avg_score,
        "drift_detected": drift_detected,
    }


@app.post("/api/record", status_code=status.HTTP_201_CREATED)
def record_sample(payload: RecordRequest) -> dict[str, Any]:
    """Ingest a sample with an image, label, and mode. Extracts and saves the vision embedding."""
    try:
        raw_str = payload.image_base64
        if "," in raw_str:
            raw_str = raw_str.split(",", 1)[1]
        image_bytes = base64.b64decode(raw_str)
        embedding = extract_vision_embedding(image_bytes)

        # Determine reviewed status with negative sampling for full auto mode
        reviewed_status = payload.reviewed
        sampled_for_review = False
        if payload.mode == "auto":
            if payload.label == 1:
                # All automated positive decisions enter review queue
                reviewed_status = 0
            elif payload.label == 0:
                # Dislikes are sampled at negative_sample_rate (default 5%)
                if random.random() < payload.negative_sample_rate:
                    reviewed_status = 0
                    sampled_for_review = True
                else:
                    reviewed_status = 1

        # Check for near-duplicate artwork before saving new image file or inserting row
        near_dup = find_near_duplicate(embedding, threshold=0.98, db_path=db.DEFAULT_DB_PATH)
        label_str = "LIKE (1)" if payload.label == 1 else ("DISLIKE (0)" if payload.label == 0 else "UNLABELED")
        set_str = f" [Set: {payload.image_set_count} photos]" if payload.image_set_count and payload.image_set_count > 1 else ""

        if near_dup is not None:
            canonical_id = int(near_dup["id"])
            target_label = payload.label
            target_mode = payload.mode
            target_reviewed = reviewed_status

            # Protect verified human decisions from being overwritten by unreviewed automated guesses
            if near_dup.get("reviewed") == 1 and payload.mode == "auto":
                target_label = near_dup["label"]
                target_mode = near_dup.get("mode", payload.mode)
                target_reviewed = 1
            elif target_label is None and near_dup.get("label") is not None:
                target_label = near_dup["label"]

            update_sample_record(
                canonical_id,
                label=target_label,
                mode=target_mode,
                prediction_score=payload.prediction_score,
                reviewed=target_reviewed,
                db_path=db.DEFAULT_DB_PATH,
            )
            sim_val = float(near_dup["similarity"])
            final_label_str = "LIKE (1)" if target_label == 1 else ("DISLIKE (0)" if target_label == 0 else "UNLABELED")
            add_activity_log(
                "INFO",
                "duplicate_consolidated",
                f"Near-duplicate artwork detected (sim={sim_val:.4f} with #{canonical_id}): consolidated into sample #{canonical_id} as {final_label_str}{set_str}",
                mode=payload.mode,
                details={
                    "id": canonical_id,
                    "similarity": sim_val,
                    "image_hash": near_dup["image_hash"],
                    "label": target_label,
                    "reviewed": target_reviewed,
                    "original_label": near_dup["label"],
                    "image_set_count": payload.image_set_count,
                },
            )
            check_session_drift(db_path=db.DEFAULT_DB_PATH)
            return {
                "status": "consolidated",
                "id": canonical_id,
                "duplicate_of": canonical_id,
                "similarity": sim_val,
                "image_hash": near_dup["image_hash"],
                "label": target_label,
                "reviewed": target_reviewed,
            }

        # If not a near-duplicate, save image to disk and insert new sample
        _, image_hash, file_path = _decode_and_save_image(payload.image_base64)
        rel_path = f"data/images/{image_hash}.jpg"
        sample_id = insert_sample(
            image_hash=image_hash,
            file_path=rel_path,
            embedding=embedding,
            label=payload.label,
            mode=payload.mode,
            prediction_score=payload.prediction_score,
            reviewed=reviewed_status,
            db_path=db.DEFAULT_DB_PATH,
        )

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
            if sampled_for_review:
                add_activity_log(
                    "INFO",
                    "negative_sampled_for_review",
                    f"Auto decision: Sample #{sample_id} ({image_hash[:8]}...) rated DISLIKE (Score: {score_str}) -> Sampled into Review Queue (5% audit rate){set_str}",
                    mode="auto",
                    details={"id": sample_id, "label": payload.label, "score": payload.prediction_score, "sampled_for_review": True},
                )
            else:
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

        check_session_drift(db_path=db.DEFAULT_DB_PATH)

        return {
            "status": "success",
            "id": sample_id,
            "image_hash": image_hash,
            "label": payload.label,
            "reviewed": reviewed_status,
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
        X, y, sample_ids = load_training_matrix(return_ids=True)
        result = train_taste_classifier(
            X,
            y,
            sample_ids=sample_ids,
            target_recall=payload.target_recall,
            threshold=payload.threshold,
            min_recall_floor=payload.min_recall_floor,
            holdout_ratio=payload.holdout_ratio,
            baseline_prompt_text=payload.baseline_prompt_text,
            baseline_image_base64=payload.baseline_image_base64,
            reset_baseline_to_default=payload.reset_baseline_to_default,
        )

        if result.get("status") == "trained":
            m = result.get("metrics", {})
            eval_info = f" ({m.get('evaluation_type', 'cv')}, {m.get('folds', 5)} folds)" if m.get("folds") else ""
            bp = m.get("best_params", {})
            params_info = f", C={bp.get('C')}, weight={bp.get('class_weight')}" if bp else ""
            add_activity_log(
                "SUCCESS",
                "model_trained",
                f"Model retrained on {result.get('sample_count')} samples: OOF PR-AUC {m.get('pr_auc')}, Recall {m.get('recall')}, θ={m.get('decision_threshold')}{eval_info}{params_info}",
                details=m,
            )
            if m.get("generalization_warning"):
                h = m.get("holdout", {})
                add_activity_log(
                    "WARNING",
                    "generalization_warning",
                    f"Holdout generalization drop: Holdout PR-AUC {h.get('pr_auc')} is significantly below OOF PR-AUC {m.get('pr_auc')}.",
                    details=h,
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


@app.post("/api/threshold")
def set_decision_threshold(payload: ThresholdRequest) -> dict[str, Any]:
    """Update active decision threshold in the model and recalculate metrics."""
    try:
        result = update_decision_threshold(payload.threshold)
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("message", "Failed to update threshold."),
            )
        add_activity_log(
            "INFO",
            "threshold_updated",
            f"Active decision threshold updated to θ={result.get('decision_threshold'):.2f}",
            details=result.get("metrics"),
        )
        return result
    except HTTPException:
        raise
    except Exception as exc:
        add_activity_log("ERROR", "threshold_update_failed", f"Failed to update decision threshold: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update decision threshold: {str(exc)}",
        ) from exc


@app.get("/api/samples")
def query_samples(
    mode: str | None = Query(None, description="Filter by mode: 'manual', 'supervised', or 'auto'."),
    reviewed: int | None = Query(None, description="Filter by review status: 1 or 0."),
    label: int | None = Query(None, description="Filter by label: 1 or 0."),
    quality: str | None = Query(None, description="Quality filter: 'inconsistent_likes' or 'inconsistent_dislikes'."),
    outliers_only: bool = Query(False, description="Filter only positive class outlier samples."),
    outlier: bool | None = Query(None, description="Alias for outliers_only filter."),
    limit: int = Query(50, ge=1, le=200, description="Max number of samples to return."),
    offset: int = Query(0, ge=0, description="Query offset."),
) -> list[dict[str, Any]]:
    """Query samples for the dashboard review queue with base64 encoded images included."""
    filter_likes = (quality == "inconsistent_likes") or outliers_only or bool(outlier)
    filter_dislikes = (quality == "inconsistent_dislikes")

    outlier_map: dict[int, dict[str, Any]] = {}
    target_ids: list[int] | None = None

    try:
        import numpy as np
        X_all, y_all, ids_all = load_training_matrix(return_ids=True, db_path=db.DEFAULT_DB_PATH)
        if len(y_all) > 0:
            if filter_likes:
                if np.any(y_all == 1):
                    outlier_map = detect_positive_outliers(X_all, y_all, sample_ids=ids_all)
                    target_ids = [s_id for s_id, data in outlier_map.items() if data.get("is_outlier")]
                else:
                    target_ids = []
            elif filter_dislikes:
                if np.any(y_all == 0):
                    outlier_map = detect_negative_outliers(X_all, y_all, sample_ids=ids_all)
                    target_ids = [s_id for s_id, data in outlier_map.items() if data.get("is_outlier")]
                else:
                    target_ids = []
            else:
                if np.any(y_all == 1):
                    pos_map = detect_positive_outliers(X_all, y_all, sample_ids=ids_all)
                    outlier_map.update(pos_map)
                if np.any(y_all == 0):
                    neg_map = detect_negative_outliers(X_all, y_all, sample_ids=ids_all)
                    outlier_map.update(neg_map)
    except Exception:
        outlier_map = {}
        if filter_likes or filter_dislikes:
            target_ids = []

    if filter_likes or filter_dislikes:
        if not target_ids:
            return []
        target_label = 1 if filter_likes else 0
        samples = get_samples(
            sample_ids=target_ids,
            mode=mode,
            reviewed=reviewed,
            label=target_label,
            limit=limit,
            offset=offset,
            db_path=db.DEFAULT_DB_PATH,
        )
    else:
        samples = get_samples(
            mode=mode,
            reviewed=reviewed,
            label=label,
            limit=limit,
            offset=offset,
            db_path=db.DEFAULT_DB_PATH,
        )

    images_dir = Path(db.IMAGES_DIR)

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

        s_id = sample_data["id"]
        if s_id in outlier_map:
            out_info = outlier_map[s_id]
            sample_data["is_outlier"] = bool(out_info.get("is_outlier", False))
            sample_data["outlier_type"] = out_info.get("outlier_type")
            sample_data["outlier_reason"] = out_info.get("outlier_reason")
            sample_data["centroid_distance"] = out_info.get("centroid_distance")
        else:
            sample_data["is_outlier"] = False
            sample_data["outlier_type"] = None
            sample_data["outlier_reason"] = None
            sample_data["centroid_distance"] = None

        results.append(sample_data)

    return results


@app.post("/api/review")
def review_samples(payload: ReviewRequest) -> dict[str, Any]:
    """Bulk update labels and mark samples as reviewed."""
    try:
        updates = [item.dict() for item in payload.updates]
        updated_count = update_sample_reviews(updates, db_path=db.DEFAULT_DB_PATH)
        return {
            "status": "success",
            "updated_count": updated_count,
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Review update failed: {str(exc)}",
        ) from exc


@app.delete("/api/samples/{sample_id}")
def delete_single_sample(sample_id: int) -> dict[str, Any]:
    """Delete a single sample from the database and remove its image file."""
    try:
        deleted_count, _ = delete_samples([sample_id], db_path=db.DEFAULT_DB_PATH)
        if deleted_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Sample #{sample_id} not found.",
            )
        add_activity_log("INFO", "sample_deleted", f"Sample #{sample_id} deleted by user.")
        return {"status": "success", "deleted_id": sample_id}
    except HTTPException:
        raise
    except Exception as exc:
        add_activity_log("ERROR", "sample_delete_failed", f"Failed to delete sample #{sample_id}: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete sample: {str(exc)}",
        ) from exc


@app.post("/api/samples/batch-delete")
def batch_delete_samples(payload: BatchDeleteRequest) -> dict[str, Any]:
    """Delete multiple samples from the database in a single transaction."""
    try:
        if not payload.ids:
            return {"status": "success", "deleted_count": 0, "deleted_ids": []}

        deleted_count, _ = delete_samples(payload.ids, db_path=db.DEFAULT_DB_PATH)
        preview_ids = str(payload.ids[:8]) + ("..." if len(payload.ids) > 8 else "")
        add_activity_log(
            "INFO",
            "samples_batch_deleted",
            f"Batch deleted {deleted_count} sample(s): IDs {preview_ids}",
            details={"deleted_count": deleted_count, "ids": payload.ids},
        )
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "deleted_ids": payload.ids,
        }
    except Exception as exc:
        add_activity_log("ERROR", "samples_batch_delete_failed", f"Failed to batch delete samples: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to batch delete samples: {str(exc)}",
        ) from exc


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    """Return dataset statistics, class balance, and active model metrics."""
    try:
        statistics = get_dataset_statistics(db_path=db.DEFAULT_DB_PATH)
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
            cold_baselines = None
            if statistics.get("positive_count", 0) >= 1 and statistics.get("negative_count", 0) >= 1:
                try:
                    X, y = load_training_matrix(db_path=db.DEFAULT_DB_PATH)
                    if len(y) > 0 and np.sum(y == 1) > 0:
                        from backend.model import DEFAULT_ZERO_SHOT_PROMPT, extract_text_embedding
                        from sklearn.metrics import precision_recall_curve, auc

                        pos_rate = float(np.sum(y == 1) / len(y))
                        c_vec = np.mean(X[y == 1], axis=0)
                        c_norm = np.linalg.norm(c_vec)
                        c_vec = (c_vec / c_norm) if c_norm > 0 else c_vec
                        c_sim = X @ c_vec
                        cp, cr, _ = precision_recall_curve(y, c_sim)
                        c_auc = float(auc(cr, cp)) if len(cr) > 1 else 0.0

                        zs_vec = extract_text_embedding(DEFAULT_ZERO_SHOT_PROMPT)
                        zs_sim = X @ zs_vec
                        zp, zr, _ = precision_recall_curve(y, zs_sim)
                        z_auc = float(auc(zr, zp)) if len(zr) > 1 else 0.0

                        cold_baselines = {
                            "random_guess": round(pos_rate, 4),
                            "positive_centroid": round(c_auc, 4),
                            "zero_shot": round(z_auc, 4),
                            "reference_type": "text",
                            "reference_source": DEFAULT_ZERO_SHOT_PROMPT,
                            "prompt_text": DEFAULT_ZERO_SHOT_PROMPT,
                        }
                except Exception:
                    cold_baselines = None

            model_info = {
                "model_loaded": False,
                "metrics": {
                    "baselines": cold_baselines,
                } if cold_baselines else None,
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


@app.get("/api/embeddings/scatter")
def get_embeddings_scatter(
    method: str = Query("pca", pattern="^(pca|tsne)$", description="Reduction method: 'pca' or 'tsne'."),
) -> dict[str, Any]:
    """Extract 2D coordinates for dataset sample embeddings using PCA or t-SNE."""
    try:
        metadata, X = load_embedding_scatter_data(db_path=db.DEFAULT_DB_PATH)
        total_points = len(metadata)

        if total_points == 0:
            return {
                "status": "empty",
                "total_points": 0,
                "method": method,
                "variance_ratio": None,
                "points": [],
            }

        if total_points == 1:
            meta = metadata[0]
            return {
                "status": "success",
                "total_points": 1,
                "method": method,
                "variance_ratio": [1.0, 0.0] if method == "pca" else None,
                "points": [
                    {
                        "id": meta["id"],
                        "image_hash": meta["image_hash"],
                        "image_url": f"/images/{meta['image_hash']}.jpg",
                        "label": meta["label"],
                        "prediction_score": meta["prediction_score"],
                        "mode": meta["mode"],
                        "reviewed": meta["reviewed"],
                        "created_at": meta["created_at"],
                        "x": 0.0,
                        "y": 0.0,
                    }
                ],
            }

        variance_ratio = None
        if method == "pca":
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)
            variance_ratio = [round(float(v), 4) for v in pca.explained_variance_ratio_]
        else:
            # t-SNE requires perplexity < n_samples
            max_perp = max(1, min(30, (total_points - 1) // 3))
            tsne = TSNE(
                n_components=2,
                perplexity=max_perp,
                random_state=42,
                init="pca" if total_points >= 4 else "random",
                learning_rate="auto",
            )
            coords = tsne.fit_transform(X)

        points: list[dict[str, Any]] = []
        for i, meta in enumerate(metadata):
            points.append(
                {
                    "id": meta["id"],
                    "image_hash": meta["image_hash"],
                    "image_url": f"/images/{meta['image_hash']}.jpg",
                    "label": meta["label"],
                    "prediction_score": meta["prediction_score"],
                    "mode": meta["mode"],
                    "reviewed": meta["reviewed"],
                    "created_at": meta["created_at"],
                    "x": round(float(coords[i, 0]), 4),
                    "y": round(float(coords[i, 1]), 4),
                }
            )

        return {
            "status": "success",
            "total_points": total_points,
            "method": method,
            "variance_ratio": variance_ratio,
            "points": points,
        }
    except Exception as exc:
        add_activity_log("ERROR", "scatter_projection_failed", f"Failed to project embeddings: {str(exc)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding projection: {str(exc)}",
        ) from exc


# Mount image store if directory exists
if db.IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(db.IMAGES_DIR)), name="images")

# Mount static directory for developer dashboard if it exists
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

