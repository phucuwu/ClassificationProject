"""Vision embedding extraction and machine learning classification pipeline.

Implements feature extraction via CLIP (clip-ViT-L-14) and subjective taste profile
classification using balanced Logistic Regression, threshold calibration, and
Precision-Recall evaluation metrics.
"""

from __future__ import annotations

import base64
import io
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, fbeta_score

# Default file paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = DATA_DIR / "model.pkl"

# Model architecture settings
CLIP_MODEL_NAME = "clip-ViT-L-14"
EMBEDDING_DIM = 768
DEFAULT_DECISION_THRESHOLD = 0.35

# Global lazy-loaded embedding model instance
_EMBEDDING_MODEL = None


def get_device() -> torch.device:
    """Detect and return available compute device (CUDA GPU if available, else CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_embedding_model():
    """Lazy load and cache the CLIP vision embedding model."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        from sentence_transformers import SentenceTransformer

        device = get_device()
        _EMBEDDING_MODEL = SentenceTransformer(CLIP_MODEL_NAME, device=str(device))
    return _EMBEDDING_MODEL


def extract_vision_embedding(image_input: Image.Image | bytes | str) -> np.ndarray:
    """Extract a 768-dimensional L2-normalized float32 vision embedding from an image.

    Args:
        image_input: A PIL Image instance, raw image bytes, or base64 data URI string.

    Returns:
        A 768-dimensional float32 NumPy array with unit L2 norm.
    """
    if isinstance(image_input, str):
        if "," in image_input:
            image_input = image_input.split(",", 1)[1]
        image_bytes = base64.b64decode(image_input)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif isinstance(image_input, bytes):
        pil_image = Image.open(io.BytesIO(image_input)).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_image = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    model = get_embedding_model()
    # SentenceTransformer returns an embedding array
    embedding = model.encode(pil_image, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype(np.float32)


def train_taste_classifier(
    X: np.ndarray,
    y: np.ndarray,
    target_recall: float = 0.90,
    model_path: Path | str = MODEL_PATH,
) -> dict[str, Any]:
    """Fit a balanced Logistic Regression classifier on feature matrix X and label vector y.

    Args:
        X: Feature matrix of shape (N, 768).
        y: Label vector of shape (N,) with binary integers (0 or 1).
        target_recall: Desired recall rate used to calibrate decision threshold.
        model_path: Destination path for saved model pickle file.

    Returns:
        Dictionary containing training status, sample distribution, and evaluation metrics.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    sample_count = len(y)
    unique_classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique_classes.tolist(), counts.tolist()))
    positive_count = class_counts.get(1, 0)
    negative_count = class_counts.get(0, 0)

    # Check for minimum class representation
    if positive_count < 1 or negative_count < 1:
        return {
            "status": "insufficient_data",
            "message": "Need at least 1 Like (positive class) and 1 Dislike (negative class) sample to train.",
            "sample_count": sample_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
        }

    # Initialize and fit balanced Logistic Regression
    classifier = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    classifier.fit(X, y)

    # Calculate probabilities on training dataset
    # P(Like) is the probability of class 1
    probabilities = classifier.predict_proba(X)[:, 1]

    # Calculate Precision-Recall curve and PR-AUC
    precisions, recalls, thresholds = precision_recall_curve(y, probabilities)
    pr_auc_score = float(auc(recalls, precisions)) if len(recalls) > 1 else 0.0

    # Calibrate decision threshold aiming for target_recall (default 90%)
    calibrated_threshold = DEFAULT_DECISION_THRESHOLD
    if len(thresholds) > 0:
        # Find threshold where recall is closest to target_recall without dropping too low
        valid_indices = np.where(recalls[:-1] >= target_recall)[0]
        if len(valid_indices) > 0:
            calibrated_threshold = float(thresholds[valid_indices[-1]])
        else:
            calibrated_threshold = float(thresholds[0])

    # Clamp threshold within reasonable bounds and round to 2 decimal places
    calibrated_threshold = float(round(float(np.clip(calibrated_threshold, 0.10, 0.90)), 2))

    # Evaluate metrics at calibrated threshold
    binary_preds = (probabilities >= calibrated_threshold).astype(int)
    tp = int(np.sum((binary_preds == 1) & (y == 1)))
    fp = int(np.sum((binary_preds == 1) & (y == 0)))
    tn = int(np.sum((binary_preds == 0) & (y == 0)))
    fn = int(np.sum((binary_preds == 0) & (y == 1)))

    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f2 = float(fbeta_score(y, binary_preds, beta=2, zero_division=0))

    metrics = {
        "pr_auc": round(pr_auc_score, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "f2_score": round(f2, 4),
        "decision_threshold": round(calibrated_threshold, 2),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
    }

    # Save model artifact
    model_payload = {
        "classifier": classifier,
        "decision_threshold": calibrated_threshold,
        "metrics": metrics,
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
    }
    with open(model_path, "wb") as f:
        pickle.dump(model_payload, f)

    return {
        "status": "trained",
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "metrics": metrics,
    }


def load_classifier(model_path: Path | str = MODEL_PATH) -> dict[str, Any] | None:
    """Load the trained taste classifier from disk if available."""
    model_path = Path(model_path)
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def predict_taste(
    embedding: np.ndarray,
    threshold: float | None = None,
    model_path: Path | str = MODEL_PATH,
) -> dict[str, Any]:
    """Calculate prediction score and binary decision for a single vision embedding.

    Args:
        embedding: 768-dimensional float32 vision embedding.
        threshold: Optional override for decision threshold. If None, uses model default.
        model_path: Path to serialized model file.

    Returns:
        Dictionary containing prediction_score, decision, threshold, and model_loaded status.
    """
    model_data = load_classifier(model_path)
    if model_data is None:
        return {
            "prediction_score": None,
            "decision": None,
            "threshold": None,
            "model_loaded": False,
            "message": "Model not trained yet. Gather samples in Manual Mode first.",
        }

    classifier: LogisticRegression = model_data["classifier"]
    active_threshold = threshold if threshold is not None else model_data.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)

    # Ensure embedding is 2D with shape (1, 768)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)

    # Calculate probability P(Like)
    prob_like = float(classifier.predict_proba(embedding)[0, 1])
    decision = 1 if prob_like >= active_threshold else 0

    return {
        "prediction_score": round(prob_like, 4),
        "decision": decision,
        "threshold": round(active_threshold, 2),
        "model_loaded": True,
    }
