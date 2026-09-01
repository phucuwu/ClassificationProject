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
from sklearn.metrics import average_precision_score, precision_recall_curve, auc, fbeta_score
from sklearn.model_selection import StratifiedKFold, train_test_split

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
    target_recall: float | None = None,
    threshold: float | None = None,
    min_recall_floor: float = 0.70,
    holdout_ratio: float = 0.15,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Fit a balanced Logistic Regression classifier on feature matrix X and label vector y.

    Uses Stratified 5-Fold Cross-Validation on the development partition to generate
    out-of-fold predictions, calibrates the decision threshold (using a hybrid strategy
    maximizing F2 with an enforced recall floor or target recall), evaluates generalization
    against a chronological holdout partition, and fits final model weights on 100% of labeled data.

    Args:
        X: Feature matrix of shape (N, 768).
        y: Label vector of shape (N,) with binary integers (0 or 1).
        target_recall: Optional target recall rate for decision threshold calibration.
        threshold: Optional explicit decision threshold. If specified, overrides calibration.
        min_recall_floor: Minimum recall floor used for hybrid F2 threshold calibration (default 0.70).
        holdout_ratio: Fraction of most recent samples reserved for generalization holdout (default 0.15).
        model_path: Destination path for saved model pickle file.

    Returns:
        Dictionary containing training status, sample distribution, and out-of-fold evaluation metrics.
    """
    actual_model_path = Path(model_path) if model_path is not None else Path(MODEL_PATH)
    actual_model_path.parent.mkdir(parents=True, exist_ok=True)

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

    # 1. Partition data into development set and stratified holdout set
    use_holdout = False
    holdout_metrics: dict[str, Any] | None = None
    generalization_warning = False

    if holdout_ratio > 0.0 and sample_count >= 5 and positive_count >= 2 and negative_count >= 2:
        try:
            test_size = float(np.clip(holdout_ratio, 0.05, 0.40))
            X_dev_split, X_holdout_split, y_dev_split, y_holdout_split = train_test_split(
                X,
                y,
                test_size=test_size,
                stratify=y,
                random_state=42,
            )
            if (
                np.sum(y_dev_split == 1) >= 1
                and np.sum(y_dev_split == 0) >= 1
                and np.sum(y_holdout_split == 1) >= 1
                and np.sum(y_holdout_split == 0) >= 1
            ):
                use_holdout = True
                X_dev = X_dev_split
                y_dev = y_dev_split
                X_holdout = X_holdout_split
                y_holdout = y_holdout_split
        except Exception:
            use_holdout = False

    if not use_holdout:
        X_dev = X
        y_dev = y
        X_holdout = None
        y_holdout = None

    dev_pos = int(np.sum(y_dev == 1))
    dev_neg = int(np.sum(y_dev == 0))

    # 2. Cross-validation fold adaptation
    n_splits = min(5, dev_pos, dev_neg)
    oof_probabilities = np.zeros(len(y_dev), dtype=np.float32)

    if n_splits >= 2:
        eval_type = "stratified_cv"
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in skf.split(X_dev, y_dev):
            fold_clf = LogisticRegression(
                class_weight="balanced",
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )
            fold_clf.fit(X_dev[train_idx], y_dev[train_idx])
            oof_probabilities[val_idx] = fold_clf.predict_proba(X_dev[val_idx])[:, 1]
    else:
        # Cold start fallback when positive count < 2
        eval_type = "in_sample_fallback"
        fold_clf = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        fold_clf.fit(X_dev, y_dev)
        oof_probabilities[:] = fold_clf.predict_proba(X_dev)[:, 1]

    # 3. Out-of-fold Precision-Recall curve and metrics
    precisions, recalls, pr_thresholds = precision_recall_curve(y_dev, oof_probabilities)
    pr_auc_score = float(auc(recalls, precisions)) if len(recalls) > 1 else 0.0
    ap_score = float(average_precision_score(y_dev, oof_probabilities)) if len(np.unique(y_dev)) > 1 else 0.0

    # 4. Calibrate decision threshold
    if threshold is not None:
        calibrated_threshold = float(round(float(np.clip(threshold, 0.05, 0.95)), 2))
    elif target_recall is not None:
        # Calibrate aiming for explicit target_recall
        calibrated_threshold = DEFAULT_DECISION_THRESHOLD
        if len(pr_thresholds) > 0:
            valid_indices = np.where(recalls[:-1] >= target_recall)[0]
            if len(valid_indices) > 0:
                calibrated_threshold = float(pr_thresholds[valid_indices[-1]])
            else:
                calibrated_threshold = float(pr_thresholds[0])
        calibrated_threshold = float(np.floor(np.clip(calibrated_threshold, 0.05, 0.95) * 100) / 100)
    else:
        # Hybrid calibration: maximize F2 with an enforced recall floor (default 0.70)
        calibrated_threshold = DEFAULT_DECISION_THRESHOLD
        candidate_cutoffs = np.linspace(0.05, 0.95, 91)
        valid_candidates: list[tuple[float, float, float]] = []  # (f2, recall, cutoff)
        all_candidates: list[tuple[float, float, float]] = []

        for cutoff in candidate_cutoffs:
            preds = (oof_probabilities >= cutoff).astype(int)
            c_tp = np.sum((preds == 1) & (y_dev == 1))
            c_fn = np.sum((preds == 0) & (y_dev == 1))
            c_rec = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
            c_f2 = float(fbeta_score(y_dev, preds, beta=2, zero_division=0))
            all_candidates.append((c_f2, c_rec, float(cutoff)))
            if c_rec >= min_recall_floor:
                valid_candidates.append((c_f2, c_rec, float(cutoff)))

        if valid_candidates:
            # Sort by F2 descending, then cutoff descending (higher precision for equal F2)
            valid_candidates.sort(key=lambda item: (item[0], item[2]), reverse=True)
            calibrated_threshold = valid_candidates[0][2]
        elif all_candidates:
            # Fallback: cutoff achieving highest recall, then highest F2
            all_candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
            calibrated_threshold = all_candidates[0][2]

        calibrated_threshold = float(round(np.clip(calibrated_threshold, 0.05, 0.95), 2))

    # Evaluate out-of-fold metrics at calibrated threshold
    binary_preds = (oof_probabilities >= calibrated_threshold).astype(int)
    tp = int(np.sum((binary_preds == 1) & (y_dev == 1)))
    fp = int(np.sum((binary_preds == 1) & (y_dev == 0)))
    tn = int(np.sum((binary_preds == 0) & (y_dev == 0)))
    fn = int(np.sum((binary_preds == 0) & (y_dev == 1)))

    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f2 = float(fbeta_score(y_dev, binary_preds, beta=2, zero_division=0))

    # 5. Holdout generalization verification
    if use_holdout and X_holdout is not None and y_holdout is not None:
        dev_clf = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        dev_clf.fit(X_dev, y_dev)
        holdout_probs = dev_clf.predict_proba(X_holdout)[:, 1]

        h_precisions, h_recalls, _ = precision_recall_curve(y_holdout, holdout_probs)
        h_pr_auc = float(auc(h_recalls, h_precisions)) if len(h_recalls) > 1 else 0.0
        h_ap = float(average_precision_score(y_holdout, holdout_probs)) if len(np.unique(y_holdout)) > 1 else 0.0

        h_preds = (holdout_probs >= calibrated_threshold).astype(int)
        h_tp = int(np.sum((h_preds == 1) & (y_holdout == 1)))
        h_fp = int(np.sum((h_preds == 1) & (y_holdout == 0)))
        h_tn = int(np.sum((h_preds == 0) & (y_holdout == 0)))
        h_fn = int(np.sum((h_preds == 0) & (y_holdout == 1)))

        h_rec = float(h_tp / (h_tp + h_fn)) if (h_tp + h_fn) > 0 else 0.0
        h_prec = float(h_tp / (h_tp + h_fp)) if (h_tp + h_fp) > 0 else 0.0
        h_f2 = float(fbeta_score(y_holdout, h_preds, beta=2, zero_division=0))

        # Check generalization: PR-AUC drop > 0.25 triggers warning
        generalization_warning = bool(h_pr_auc < (pr_auc_score - 0.25))

        holdout_metrics = {
            "sample_count": len(y_holdout),
            "positive_count": int(np.sum(y_holdout == 1)),
            "negative_count": int(np.sum(y_holdout == 0)),
            "pr_auc": round(h_pr_auc, 4),
            "average_precision": round(h_ap, 4),
            "recall": round(h_rec, 4),
            "precision": round(h_prec, 4),
            "f2_score": round(h_f2, 4),
            "confusion_matrix": {
                "true_positives": h_tp,
                "false_positives": h_fp,
                "true_negatives": h_tn,
                "false_negatives": h_fn,
            },
            "generalization_warning": generalization_warning,
        }

    # 6. Fit final model on 100% of labeled data
    final_classifier = LogisticRegression(
        class_weight="balanced",
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    final_classifier.fit(X, y)

    metrics = {
        "pr_auc": round(pr_auc_score, 4),
        "average_precision": round(ap_score, 4),
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
        "evaluation_type": eval_type,
        "folds": n_splits if eval_type == "stratified_cv" else None,
        "holdout": holdout_metrics,
        "generalization_warning": generalization_warning,
    }

    # Save model artifact
    model_payload = {
        "classifier": final_classifier,
        "decision_threshold": calibrated_threshold,
        "metrics": metrics,
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "oof_probabilities": oof_probabilities.tolist(),
        "y_oof": y_dev.tolist(),
    }
    with open(actual_model_path, "wb") as f:
        pickle.dump(model_payload, f)

    return {
        "status": "trained",
        "sample_count": sample_count,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "metrics": metrics,
    }


def update_decision_threshold(
    new_threshold: float,
    model_path: Path | str | None = None,
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Update active decision threshold in the trained model and recompute metrics.

    Uses cached out-of-fold validation probabilities when available to avoid
    retraining and prevent in-sample evaluation leakage.

    Args:
        new_threshold: New probability cutoff value between 0.05 and 0.95.
        model_path: Destination path for saved model pickle file.
        db_path: Optional SQLite database file path.

    Returns:
        Dictionary containing success flag, updated threshold, and recomputed metrics.
    """
    actual_model_path = Path(model_path) if model_path is not None else Path(MODEL_PATH)
    model_data = load_classifier(actual_model_path)
    if model_data is None:
        return {
            "success": False,
            "message": "Model not trained yet. Cannot update decision threshold.",
        }

    clamped_threshold = float(round(float(np.clip(new_threshold, 0.05, 0.95)), 2))

    # Prefer cached out-of-fold predictions to prevent in-sample leakage
    if "oof_probabilities" in model_data and "y_oof" in model_data:
        probabilities = np.array(model_data["oof_probabilities"], dtype=np.float32)
        y_eval = np.array(model_data["y_oof"], dtype=np.int32)
    else:
        from backend.database import load_training_matrix

        X, y_eval = load_training_matrix(db_path=db_path)
        if len(y_eval) == 0:
            return {
                "success": False,
                "message": "No labeled samples found in dataset.",
            }
        classifier: LogisticRegression = model_data["classifier"]
        probabilities = classifier.predict_proba(X)[:, 1]

    precisions, recalls, _ = precision_recall_curve(y_eval, probabilities)
    pr_auc_score = float(auc(recalls, precisions)) if len(recalls) > 1 else 0.0
    ap_score = float(average_precision_score(y_eval, probabilities)) if len(np.unique(y_eval)) > 1 else 0.0

    binary_preds = (probabilities >= clamped_threshold).astype(int)
    tp = int(np.sum((binary_preds == 1) & (y_eval == 1)))
    fp = int(np.sum((binary_preds == 1) & (y_eval == 0)))
    tn = int(np.sum((binary_preds == 0) & (y_eval == 0)))
    fn = int(np.sum((binary_preds == 0) & (y_eval == 1)))

    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f2 = float(fbeta_score(y_eval, binary_preds, beta=2, zero_division=0))

    existing_metrics = model_data.get("metrics", {})
    metrics = {
        "pr_auc": round(pr_auc_score, 4),
        "average_precision": round(ap_score, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "f2_score": round(f2, 4),
        "decision_threshold": clamped_threshold,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "evaluation_type": existing_metrics.get("evaluation_type", "out_of_fold"),
        "folds": existing_metrics.get("folds"),
        "holdout": existing_metrics.get("holdout"),
        "generalization_warning": existing_metrics.get("generalization_warning", False),
    }

    model_data["decision_threshold"] = clamped_threshold
    model_data["metrics"] = metrics

    with open(actual_model_path, "wb") as f:
        pickle.dump(model_data, f)

    return {
        "success": True,
        "decision_threshold": clamped_threshold,
        "metrics": metrics,
    }


def load_classifier(model_path: Path | str | None = None) -> dict[str, Any] | None:
    """Load the trained taste classifier from disk if available."""
    actual_model_path = Path(model_path) if model_path is not None else Path(MODEL_PATH)
    if not actual_model_path.exists():
        return None
    try:
        with open(actual_model_path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def predict_taste(
    embedding: np.ndarray,
    threshold: float | None = None,
    model_path: Path | str | None = None,
) -> dict[str, Any]:
    """Calculate prediction score and binary decision for a single vision embedding.

    Args:
        embedding: 768-dimensional float32 vision embedding.
        threshold: Optional override for decision threshold. If None, uses model default.
        model_path: Path to serialized model file.

    Returns:
        Dictionary containing prediction_score, decision, threshold, and model_loaded status.
    """
    actual_model_path = Path(model_path) if model_path is not None else Path(MODEL_PATH)
    model_data = load_classifier(actual_model_path)
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
