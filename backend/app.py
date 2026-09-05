"""FastAPI backend server for the local art taste classifier.

Provides REST endpoints for sample recording, prediction inference,
classifier retraining, review queue management, and dataset metrics.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import threading
import warnings
from pathlib import Path
from typing import Any, Literal

# Filter third-party library deprecation notices
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="fastapi.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field, field_validator

import numpy as np

import backend.database as db
from backend.database import (
    get_dataset_statistics,
    get_recent_samples,
    get_samples,
    init_db,
    insert_sample,
    find_near_duplicate,
    load_training_matrix,
    load_embedding_scatter_data,
    update_sample_reviews,
    delete_samples,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from backend.model import (
    DEFAULT_DECISION_THRESHOLD,
    EFFECTIVENESS_PRECISION_TARGET,
    EFFECTIVENESS_RECALL_TARGET,
    TEMPORAL_MIN_HOLDOUT_LIKES,
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

# Configure CORS to allow only the local Developer dashboard origin(s).
# The Userscript communicates via GM_xmlhttpRequest (no CORS preflight); the
# dashboard is served by this Backend server, so only its local origins are listed.
LOCAL_DASHBOARD_ORIGINS = [
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=LOCAL_DASHBOARD_ORIGINS,
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
    label: Literal[0, 1] | None = Field(None, description="1 for Like (positive class), 0 for Dislike, or null for unlabeled.")
    mode: Literal["manual", "supervised", "auto"] = Field("manual", description="Operating mode: 'manual', 'supervised', or 'auto'.")
    prediction_score: float | None = Field(None, ge=0.0, le=1.0, description="Model prediction score P(Like) between 0.0 and 1.0.")
    reviewed: Literal[0, 1] = Field(0, description="Caller review hint; the server derives review state from mode and ignores conflicting values.")
    image_set_count: int | None = Field(None, ge=1, description="Total photos detected in the active card's image set.")
    negative_sample_rate: float = Field(0.05, ge=0.0, le=1.0, description="Legacy audit-rate hint; ignored because every Full auto Sample stays unreviewed.")

    @field_validator("prediction_score")
    @classmethod
    def _check_finite_score(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("prediction_score must be a finite float in [0.0, 1.0].")
        return value


class PredictRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded image string or data URI.")
    threshold: float | None = Field(None, description="Optional override for decision threshold.")


class TrainRequest(BaseModel):
    target_recall: float | None = Field(None, description="Optional target recall rate for decision threshold calibration.")
    threshold: float | None = Field(None, description="Optional explicit threshold override.")
    min_recall_floor: float = Field(0.70, ge=0.05, le=0.95, description="Minimum recall floor for hybrid F2 threshold calibration.")
    holdout_ratio: float = Field(0.15, ge=0.0, le=0.50, description="Fraction of newest training-eligible samples reserved for the temporal holdout.")
    min_holdout_positives: int = Field(30, ge=1, description="Minimum Positive-class (Like) count required for a valid temporal holdout.")
    baseline_prompt_text: str | None = Field(None, description="Optional custom prompt text for the zero-shot reference baseline.")
    baseline_image_base64: str | None = Field(None, description="Optional base64 exemplar image for the zero-shot reference baseline.")
    reset_baseline_to_default: bool = Field(False, description="Flag to reset the reference baseline back to the default text prompt.")


class ThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.01, le=0.99, description="Active decision threshold.")


class ReviewUpdateItem(BaseModel):
    id: int = Field(..., description="Sample database identifier.")
    label: Literal[0, 1] = Field(..., description="Updated label (1 for Like, 0 for Dislike).")
    reviewed: Literal[0, 1] = Field(1, description="Caller review hint; the server always persists reviewed=1 with review_confirmation provenance.")


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


def _derive_record_provenance(mode: str) -> tuple[int, str]:
    """Derive server-side review state and label provenance from the operating mode.

    Every Full auto Sample persists as unreviewed ``auto_decision``. Manual and
    confirmed Supervised-mode Samples persist as reviewed human-confirmed
    Samples (``manual_rating`` / ``supervised_confirmation``).
    """
    if mode == "auto":
        return 0, "auto_decision"
    if mode == "supervised":
        return 1, "supervised_confirmation"
    return 1, "manual_rating"


def _samples_has_provenance_column(db_path: Path | str | None = None) -> bool:
    """Return True when the samples table carries the Phase 1 label_provenance column."""
    conn = db.get_db_connection(db_path)
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(samples);").fetchall()}
        return "label_provenance" in columns
    finally:
        conn.close()


def _fetch_sample_by_hash(image_hash: str, db_path: Path | str | None = None) -> dict[str, Any] | None:
    """Fetch a Sample row by exact Primary-image hash.

    Reads label_provenance when the Phase 1 column is present; otherwise derives
    provenance locally from mode/reviewed without touching the schema.
    """
    conn = db.get_db_connection(db_path)
    try:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(samples);").fetchall()}
        has_provenance = "label_provenance" in columns
        selected = "id, image_hash, file_path, label, prediction_score, mode, reviewed"
        if has_provenance:
            selected += ", label_provenance"
        cursor = conn.execute(
            f"SELECT {selected} FROM samples WHERE image_hash = ?;",
            (image_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        sample = dict(row)
        if not has_provenance:
            reviewed_status, provenance = _derive_record_provenance(str(sample.get("mode", "auto")))
            if sample.get("reviewed") != reviewed_status:
                provenance = "auto_decision"
            sample["label_provenance"] = provenance
        return sample
    finally:
        conn.close()


def _insert_recorded_sample(
    *,
    image_hash: str,
    file_path: str,
    embedding: Any,
    label: int | None,
    mode: str,
    prediction_score: float | None,
    reviewed: int,
    provenance: str,
    db_path: Path | str | None = None,
) -> int:
    """Insert a Sample, persisting provenance when the column exists."""
    if _samples_has_provenance_column(db_path):
        return insert_sample(
            image_hash=image_hash,
            file_path=file_path,
            embedding=embedding,
            label=label,
            mode=mode,
            prediction_score=prediction_score,
            reviewed=reviewed,
            label_provenance=provenance,
            db_path=db_path,
        )
    return insert_sample(
        image_hash=image_hash,
        file_path=file_path,
        embedding=embedding,
        label=label,
        mode=mode,
        prediction_score=prediction_score,
        reviewed=reviewed,
        db_path=db_path,
    )


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
    """Ingest a Sample with an image, label, and operating mode.

    The server derives review state and label provenance from the operating
    mode: every Full auto Sample persists as unreviewed ``auto_decision``,
    while Manual and confirmed Supervised-mode Samples persist as reviewed
    human-confirmed Samples. Deduplication is limited to the exact
    Primary-image hash; visual similarity is advisory only and never mutates
    an existing Sample's label, mode, review state, or provenance.
    """
    try:
        raw_str = payload.image_base64
        if "," in raw_str:
            raw_str = raw_str.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(raw_str)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Invalid image_base64 payload: {str(exc)}",
            ) from exc
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Invalid image_base64 payload: empty image bytes.",
            )
        image_hash = hashlib.sha256(image_bytes).hexdigest()

        # Server-side review state and label provenance; caller hints are ignored.
        reviewed_status, provenance = _derive_record_provenance(payload.mode)

        label_str = "LIKE (1)" if payload.label == 1 else ("DISLIKE (0)" if payload.label == 0 else "UNLABELED")
        set_str = f" [Set: {payload.image_set_count} photos]" if payload.image_set_count and payload.image_set_count > 1 else ""

        # Exact Primary-image hash deduplication: no second row and no overwrite
        # of a confirmed label, mode, review state, or provenance.
        existing = _fetch_sample_by_hash(image_hash, db.DEFAULT_DB_PATH)
        if existing is not None:
            existing_id = int(existing["id"])
            add_activity_log(
                "INFO",
                "exact_duplicate_ignored",
                f"Exact Primary-image hash already recorded as Sample #{existing_id}; no second row created and confirmed label preserved{set_str}",
                mode=payload.mode,
                details={
                    "id": existing_id,
                    "similarity": 1.0,
                    "image_hash": image_hash,
                    "label": existing["label"],
                    "reviewed": existing["reviewed"],
                    "label_provenance": existing.get("label_provenance"),
                    "incoming_label": payload.label,
                    "incoming_mode": payload.mode,
                    "image_set_count": payload.image_set_count,
                },
            )
            check_session_drift(db_path=db.DEFAULT_DB_PATH)
            return {
                "status": "duplicate",
                "id": existing_id,
                "duplicate_of": existing_id,
                "similarity": 1.0,
                "image_hash": image_hash,
                "label": existing["label"],
                "reviewed": existing["reviewed"],
                "label_provenance": existing.get("label_provenance"),
                "provenance": existing.get("label_provenance"),
            }

        embedding = extract_vision_embedding(image_bytes)

        # Advisory visual-similarity signal: a non-identical Primary image with
        # similarity >= 0.98 still creates a separate Sample and only logs a
        # warning. It never mutates label, mode, review state, or provenance.
        advisory = None
        try:
            advisory = find_near_duplicate(embedding, threshold=0.98, db_path=db.DEFAULT_DB_PATH)
        except Exception:
            advisory = None

        # Save image to disk and insert the new separate Sample.
        _, saved_hash, file_path = _decode_and_save_image(payload.image_base64)
        rel_path = f"data/images/{saved_hash}.jpg"
        sample_id = _insert_recorded_sample(
            image_hash=saved_hash,
            file_path=rel_path,
            embedding=embedding,
            label=payload.label,
            mode=payload.mode,
            prediction_score=payload.prediction_score,
            reviewed=reviewed_status,
            provenance=provenance,
            db_path=db.DEFAULT_DB_PATH,
        )

        if advisory is not None and advisory.get("image_hash") != saved_hash:
            sim_val = float(advisory.get("similarity", 0.0))
            add_activity_log(
                "WARNING",
                "similarity_advisory",
                f"Visually similar artwork (sim={sim_val:.4f} with #{advisory.get('id')}): recorded as separate Sample #{sample_id}; existing Sample unchanged{set_str}",
                mode=payload.mode,
                details={
                    "id": sample_id,
                    "similarity": sim_val,
                    "similar_to": advisory.get("id"),
                    "similar_to_hash": advisory.get("image_hash"),
                    "similar_to_label": advisory.get("label"),
                    "image_hash": saved_hash,
                    "label": payload.label,
                    "reviewed": reviewed_status,
                    "label_provenance": provenance,
                    "image_set_count": payload.image_set_count,
                },
            )

        if payload.mode == "manual":
            add_activity_log(
                "SUCCESS",
                "sample_recorded",
                f"Manual rating recorded: Sample #{sample_id} ({saved_hash[:8]}...) as {label_str}{set_str}",
                mode="manual",
                details={"id": sample_id, "label": payload.label, "hash": saved_hash[:8], "label_provenance": provenance, "image_set_count": payload.image_set_count},
            )
        elif payload.mode == "auto":
            score_str = f"{payload.prediction_score:.2f}" if payload.prediction_score is not None else "N/A"
            add_activity_log(
                "INFO",
                "auto_decision",
                f"Auto decision: Sample #{sample_id} ({saved_hash[:8]}...) rated as {label_str} (Score: {score_str}) -> Review Queue (unreviewed auto_decision){set_str}",
                mode="auto",
                details={"id": sample_id, "label": payload.label, "score": payload.prediction_score, "label_provenance": provenance, "image_set_count": payload.image_set_count},
            )
        else:
            add_activity_log(
                "SUCCESS",
                "supervised_confirmed",
                f"Supervised rating: Sample #{sample_id} ({saved_hash[:8]}...) saved as {label_str}{set_str}",
                mode="supervised",
                details={"id": sample_id, "label": payload.label, "label_provenance": provenance, "image_set_count": payload.image_set_count},
            )

        check_session_drift(db_path=db.DEFAULT_DB_PATH)

        return {
            "status": "success",
            "id": sample_id,
            "image_hash": saved_hash,
            "label": payload.label,
            "reviewed": reviewed_status,
            "label_provenance": provenance,
            "provenance": provenance,
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
            min_holdout_positives=payload.min_holdout_positives,
            baseline_prompt_text=payload.baseline_prompt_text,
            baseline_image_base64=payload.baseline_image_base64,
            reset_baseline_to_default=payload.reset_baseline_to_default,
        )

        if result.get("status") == "trained":
            m = result.get("metrics", {})
            tuning = m.get("tuning", {})
            eff = m.get("effectiveness", {})
            eval_info = f" ({tuning.get('evaluation_type', m.get('evaluation_type', 'cv'))}, {tuning.get('folds', m.get('folds', 5))} folds)" if (tuning.get("folds") or m.get("folds")) else ""
            bp = tuning.get("best_params") or m.get("best_params", {})
            params_info = f", C={bp.get('C')}, weight={bp.get('class_weight')}" if bp else ""
            eligible = m.get("training_eligible", {})
            eligible_info = f" on {eligible.get('sample_count', result.get('sample_count'))} training-eligible Samples" if eligible else f" on {result.get('sample_count')} samples"
            temporal = m.get("temporal_holdout") or {}
            if temporal.get("status") == "available":
                temporal_info = f"; temporal holdout Recall {temporal.get('recall')}, Precision {temporal.get('precision')} ({temporal.get('positive_count')} Likes / {temporal.get('sample_count')} Samples)"
            else:
                temporal_info = "; temporal evaluation unavailable"
            add_activity_log(
                "SUCCESS",
                "model_trained",
                f"Model retrained{eligible_info}: tuning PR-AUC {tuning.get('pr_auc', m.get('pr_auc'))}, Recall {tuning.get('recall', m.get('recall'))}, θ={tuning.get('decision_threshold', m.get('decision_threshold'))}{eval_info}{params_info}{temporal_info}",
                details=m,
            )
            if m.get("warning_active"):
                reasons = m.get("warning_reasons", [])
                add_activity_log(
                    "WARNING",
                    "effectiveness_warning",
                    f"Full auto effectiveness warning active ({eff.get('status', 'unknown')}): {', '.join(reasons) if reasons else 'target unmet'}.",
                    details={"warning_reasons": reasons, "effectiveness": eff},
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
    """Bulk review Samples: atomically writes the selected label, reviewed=1, and review_confirmation provenance."""
    try:
        if _samples_has_provenance_column(db.DEFAULT_DB_PATH):
            updates = [
                {
                    "id": item.id,
                    "label": item.label,
                    "reviewed": 1,
                    "label_provenance": "review_confirmation",
                }
                for item in payload.updates
            ]
        else:
            updates = [
                {"id": item.id, "label": item.label, "reviewed": 1}
                for item in payload.updates
            ]
        updated_count = update_sample_reviews(updates, db_path=db.DEFAULT_DB_PATH)
        return {
            "status": "success",
            "updated_count": updated_count,
            "label_provenance": "review_confirmation",
            "provenance": "review_confirmation",
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

        current_eligible = {
            "sample_count": statistics.get("training_eligible_count", 0),
            "positive_count": statistics.get("training_eligible_positive_count", 0),
            "negative_count": statistics.get("training_eligible_negative_count", 0),
        }

        if model_data is not None:
            stored_metrics = model_data.get("metrics", {})
            stored_eligible = stored_metrics.get("training_eligible") or {
                "sample_count": model_data.get("sample_count", 0),
                "positive_count": model_data.get("positive_count", 0),
                "negative_count": model_data.get("negative_count", 0),
            }
            effectiveness = stored_metrics.get("effectiveness")
            if not isinstance(effectiveness, dict):
                # Legacy artifact predating provenance-aware temporal evaluation
                # metadata: never present it as a current effectiveness report.
                effectiveness = {
                    "status": "temporal_evaluation_unavailable",
                    "warning_active": True,
                    "warning_reasons": [
                        "temporal_evaluation_unavailable",
                        "legacy_artifact_missing_temporal_evaluation",
                    ],
                    "recall_target": EFFECTIVENESS_RECALL_TARGET,
                    "precision_target": EFFECTIVENESS_PRECISION_TARGET,
                    "min_holdout_positives": TEMPORAL_MIN_HOLDOUT_LIKES,
                    "threshold_source": stored_metrics.get("threshold_source", "unknown"),
                }
            stale_reason: str | None = None
            if "temporal_holdout" not in stored_metrics or "eval_boundary" not in stored_metrics:
                stale_reason = "legacy artifact without temporal evaluation metadata"
            elif stored_eligible.get("sample_count") != current_eligible["sample_count"]:
                stale_reason = (
                    f"Dataset database has {current_eligible['sample_count']} training-eligible Samples "
                    f"but the model was trained on {stored_eligible.get('sample_count')}"
                )
            model_info = {
                "model_loaded": True,
                "metrics": stored_metrics,
                "decision_threshold": round(float(model_data.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)), 2),
                "threshold_source": stored_metrics.get("threshold_source", model_data.get("threshold_source", "unknown")),
                "positive_count": model_data.get("positive_count", 0),
                "negative_count": model_data.get("negative_count", 0),
                "training_eligible": stored_eligible,
                "effectiveness": effectiveness,
                "warning_active": bool(effectiveness.get("warning_active", stored_metrics.get("warning_active", True))),
                "warning_reasons": list(effectiveness.get("warning_reasons", stored_metrics.get("warning_reasons", []))),
                "stale_model": stale_reason is not None,
                "stale_reason": stale_reason,
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
                "threshold_source": "none",
                "training_eligible": current_eligible,
                "effectiveness": {
                    "status": "temporal_evaluation_unavailable",
                    "warning_active": True,
                    "warning_reasons": ["temporal_evaluation_unavailable", "model_not_trained"],
                    "recall_target": EFFECTIVENESS_RECALL_TARGET,
                    "precision_target": EFFECTIVENESS_PRECISION_TARGET,
                    "min_holdout_positives": TEMPORAL_MIN_HOLDOUT_LIKES,
                    "threshold_source": "none",
                },
                "warning_active": True,
                "warning_reasons": ["temporal_evaluation_unavailable", "model_not_trained"],
                "stale_model": False,
                "stale_reason": None,
                "message": "Model not trained yet.",
            }

        return {
            "statistics": statistics,
            "model_status": model_info,
            "training_eligible": current_eligible,
            "provenance_counts": statistics.get("provenance_counts", {}),
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
                        "label_provenance": meta.get("label_provenance"),
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
                    "label_provenance": meta.get("label_provenance"),
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


class BenchmarkRequest(BaseModel):
    models: list[str] | None = None
    limit: int | None = None
    batch_size: int = 32
    force_extract: bool = False


_BENCHMARK_LOCK = threading.Lock()
_BENCHMARK_STATE: dict[str, Any] = {
    "status": "idle",
    "percent": 0.0,
    "current_model": None,
    "processed_samples": 0,
    "total_samples": 0,
    "message": "No benchmark has been run yet.",
    "results": None,
    "updated_at": None,
    "error": None,
}


def _run_benchmark_worker(
    models: list[str] | None,
    limit: int | None,
    batch_size: int,
    force_extract: bool,
) -> None:
    global _BENCHMARK_STATE
    from tasks.benchmark_backbones import run_backbone_benchmark

    def progress_cb(info: dict[str, Any]) -> None:
        with _BENCHMARK_LOCK:
            _BENCHMARK_STATE["status"] = info.get("status", "running")
            _BENCHMARK_STATE["percent"] = info.get("percent", 0.0)
            _BENCHMARK_STATE["current_model"] = info.get("current_model")
            _BENCHMARK_STATE["processed_samples"] = info.get("processed_samples", 0)
            _BENCHMARK_STATE["total_samples"] = info.get("total_samples", 0)
            _BENCHMARK_STATE["message"] = info.get("message", "")
            if "results" in info:
                _BENCHMARK_STATE["results"] = info["results"]

    try:
        res = run_backbone_benchmark(
            models=models,
            limit=limit,
            batch_size=batch_size,
            force_extract=force_extract,
            progress_callback=progress_cb,
        )
        with _BENCHMARK_LOCK:
            _BENCHMARK_STATE["status"] = "completed"
            _BENCHMARK_STATE["percent"] = 100.0
            _BENCHMARK_STATE["current_model"] = None
            _BENCHMARK_STATE["message"] = "Benchmark completed successfully."
            _BENCHMARK_STATE["results"] = res.get("results", [])
            _BENCHMARK_STATE["updated_at"] = res.get("timestamp")
            _BENCHMARK_STATE["total_duration_seconds"] = res.get("total_duration_seconds")
            _BENCHMARK_STATE["sample_count"] = res.get("sample_count", 0)
    except Exception as exc:
        with _BENCHMARK_LOCK:
            _BENCHMARK_STATE["status"] = "error"
            _BENCHMARK_STATE["error"] = str(exc)
            _BENCHMARK_STATE["message"] = f"Benchmark failed: {str(exc)}"


@app.post("/api/benchmark")
def start_backbone_benchmark(payload: BenchmarkRequest | None = None) -> dict[str, Any]:
    """Start asynchronous vision backbone benchmark evaluation in the background."""
    with _BENCHMARK_LOCK:
        if _BENCHMARK_STATE.get("status") == "running":
            return {
                "status": "already_running",
                "message": "A benchmark evaluation is already in progress.",
                "state": _BENCHMARK_STATE,
            }
        _BENCHMARK_STATE["status"] = "running"
        _BENCHMARK_STATE["percent"] = 0.0
        _BENCHMARK_STATE["current_model"] = None
        _BENCHMARK_STATE["error"] = None
        _BENCHMARK_STATE["message"] = "Starting vision backbone benchmark..."

    req = payload or BenchmarkRequest()
    thread = threading.Thread(
        target=_run_benchmark_worker,
        args=(req.models, req.limit, req.batch_size, req.force_extract),
        daemon=True,
    )
    thread.start()

    return {
        "status": "started",
        "message": "Vision backbone benchmark started in background.",
    }


@app.get("/api/benchmark")
def get_backbone_benchmark_status() -> dict[str, Any]:
    """Retrieve status, live progress, and results of vision backbone benchmark."""
    with _BENCHMARK_LOCK:
        state_copy = dict(_BENCHMARK_STATE)

    # If idle in this process, check if results file exists on disk from prior run
    if state_copy["status"] == "idle" and state_copy["results"] is None:
        cache_results = Path(PROJECT_ROOT) / "data" / "cache" / "backbone_benchmark_results.json"
        if cache_results.exists():
            try:
                with open(cache_results, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                state_copy["status"] = "completed"
                state_copy["percent"] = 100.0
                state_copy["results"] = cached.get("results", [])
                state_copy["updated_at"] = cached.get("timestamp")
                state_copy["total_duration_seconds"] = cached.get("total_duration_seconds")
                state_copy["sample_count"] = cached.get("sample_count", 0)
                state_copy["message"] = "Loaded cached benchmark results."
            except Exception:
                pass

    return state_copy


# Mount image store if directory exists
if db.IMAGES_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(db.IMAGES_DIR)), name="images")

# Mount static directory for developer dashboard if it exists
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

