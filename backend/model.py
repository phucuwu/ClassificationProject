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
DEFAULT_ZERO_SHOT_PROMPT = "goth aesthetic alternative indie girl style"
HYPERPARAMETER_C_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

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


def extract_text_embedding(text: str) -> np.ndarray:
    """Extract a 768-dimensional L2-normalized float32 text embedding using CLIP.

    Args:
        text: Prompt text string to encode.

    Returns:
        A 768-dimensional float32 NumPy array with unit L2 norm.
    """
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype(np.float32)


def get_candidate_class_weights(y_train: np.ndarray) -> list[tuple[str, Any]]:
    """Generate candidate class weight configurations adapted to class distribution.

    Args:
        y_train: Binary label vector for training partition.

    Returns:
        List of (weight_key, class_weight_param) tuples.
    """
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))

    if pos_count == 0 or neg_count == 0:
        return [("balanced", "balanced"), ("unweighted", None)]

    ratio = float(neg_count / pos_count)
    w_balanced_1_5x = {0: 1.0, 1: float(1.5 * ratio)}
    w_balanced_2_0x = {0: 1.0, 1: float(2.0 * ratio)}

    return [
        ("balanced", "balanced"),
        ("unweighted", None),
        ("balanced_1.5x", w_balanced_1_5x),
        ("balanced_2.0x", w_balanced_2_0x),
    ]


def train_taste_classifier(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[int] | None = None,
    target_recall: float | None = None,
    threshold: float | None = None,
    min_recall_floor: float = 0.70,
    holdout_ratio: float = 0.15,
    baseline_prompt_text: str | None = None,
    baseline_image_base64: str | None = None,
    reset_baseline_to_default: bool = False,
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
        sample_ids: Optional parallel list of integer sample IDs.
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
    dev_ids: list[int] | None = None

    if holdout_ratio > 0.0 and sample_count >= 5 and positive_count >= 2 and negative_count >= 2:
        try:
            test_size = float(np.clip(holdout_ratio, 0.05, 0.40))
            indices = np.arange(sample_count)
            dev_idx, holdout_idx = train_test_split(
                indices,
                test_size=test_size,
                stratify=y,
                random_state=42,
            )
            if (
                np.sum(y[dev_idx] == 1) >= 1
                and np.sum(y[dev_idx] == 0) >= 1
                and np.sum(y[holdout_idx] == 1) >= 1
                and np.sum(y[holdout_idx] == 0) >= 1
            ):
                use_holdout = True
                X_dev = X[dev_idx]
                y_dev = y[dev_idx]
                dev_ids = [sample_ids[i] for i in dev_idx] if sample_ids else None
                X_holdout = X[holdout_idx]
                y_holdout = y[holdout_idx]
        except Exception:
            use_holdout = False

    if not use_holdout:
        X_dev = X
        y_dev = y
        dev_ids = sample_ids
        X_holdout = None
        y_holdout = None

    dev_pos = int(np.sum(y_dev == 1))
    dev_neg = int(np.sum(y_dev == 0))

    # 2. Hyperparameter search over C and candidate class weights
    n_splits = min(5, dev_pos, dev_neg)
    oof_probabilities = np.zeros(len(y_dev), dtype=np.float32)
    tuning_results: list[dict[str, Any]] = []

    if n_splits >= 2:
        eval_type = "stratified_cv"
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(X_dev, y_dev))
        class_weight_candidates = get_candidate_class_weights(y_dev)

        oof_probs_by_config: dict[tuple[float, str], np.ndarray] = {}

        for cand_c in HYPERPARAMETER_C_GRID:
            for weight_key, weight_param in class_weight_candidates:
                cand_oof_probs = np.zeros(len(y_dev), dtype=np.float32)
                for train_idx, val_idx in splits:
                    fold_clf = LogisticRegression(
                        C=cand_c,
                        class_weight=weight_param,
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=42,
                    )
                    fold_clf.fit(X_dev[train_idx], y_dev[train_idx])
                    cand_oof_probs[val_idx] = fold_clf.predict_proba(X_dev[val_idx])[:, 1]

                c_prec, c_rec, _ = precision_recall_curve(y_dev, cand_oof_probs)
                c_pr_auc = float(auc(c_rec, c_prec)) if len(c_rec) > 1 else 0.0
                c_ap = float(average_precision_score(y_dev, cand_oof_probs)) if len(np.unique(y_dev)) > 1 else 0.0

                config_key = (cand_c, weight_key)
                oof_probs_by_config[config_key] = cand_oof_probs
                tuning_results.append({
                    "C": cand_c,
                    "class_weight": weight_key,
                    "pr_auc": round(c_pr_auc, 4),
                    "average_precision": round(c_ap, 4),
                })

        # Rank configs: highest PR-AUC, then AP, then closest to C=1.0, then balanced
        tuning_results.sort(
            key=lambda r: (
                r["pr_auc"],
                r["average_precision"],
                -abs(r["C"] - 1.0),
                1 if r["class_weight"] == "balanced" else 0,
            ),
            reverse=True,
        )

        best_config = tuning_results[0]
        best_C = float(best_config["C"])
        best_weight_key = str(best_config["class_weight"])
        best_weight_param = next(w[1] for w in class_weight_candidates if w[0] == best_weight_key)
        oof_probabilities = oof_probs_by_config[(best_C, best_weight_key)]

        # Out-of-fold centroid cosine similarity baseline
        oof_centroid_scores = np.zeros(len(y_dev), dtype=np.float32)
        for train_idx, val_idx in splits:
            pos_mask = (y_dev[train_idx] == 1)
            if np.any(pos_mask):
                c_vec = np.mean(X_dev[train_idx][pos_mask], axis=0)
                norm = np.linalg.norm(c_vec)
                c_vec = (c_vec / norm) if norm > 0 else c_vec
                oof_centroid_scores[val_idx] = X_dev[val_idx] @ c_vec
            else:
                oof_centroid_scores[val_idx] = 0.0

        cent_prec, cent_rec, _ = precision_recall_curve(y_dev, oof_centroid_scores)
        centroid_pr_auc = float(auc(cent_rec, cent_prec)) if len(cent_rec) > 1 else 0.0

    else:
        # Cold start fallback when positive count < 2
        eval_type = "in_sample_fallback"
        best_C = 1.0
        best_weight_key = "balanced"
        best_weight_param = "balanced"

        fold_clf = LogisticRegression(
            class_weight="balanced",
            C=1.0,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        fold_clf.fit(X_dev, y_dev)
        oof_probabilities[:] = fold_clf.predict_proba(X_dev)[:, 1]

        c_prec, c_rec, _ = precision_recall_curve(y_dev, oof_probabilities)
        c_pr_auc = float(auc(c_rec, c_prec)) if len(c_rec) > 1 else 0.0
        c_ap = float(average_precision_score(y_dev, oof_probabilities)) if len(np.unique(y_dev)) > 1 else 0.0
        tuning_results.append({
            "C": 1.0,
            "class_weight": "balanced",
            "pr_auc": round(c_pr_auc, 4),
            "average_precision": round(c_ap, 4),
        })

        # In-sample centroid baseline fallback
        pos_mask = (y_dev == 1)
        if np.any(pos_mask):
            c_vec = np.mean(X_dev[pos_mask], axis=0)
            norm = np.linalg.norm(c_vec)
            c_vec = (c_vec / norm) if norm > 0 else c_vec
            in_sample_scores = X_dev @ c_vec
            cent_prec, cent_rec, _ = precision_recall_curve(y_dev, in_sample_scores)
            centroid_pr_auc = float(auc(cent_rec, cent_prec)) if len(cent_rec) > 1 else 0.0
        else:
            centroid_pr_auc = 0.0

    # 3. Reference baselines calculation
    random_guess_pr_auc = round(float(dev_pos / len(y_dev)), 4) if len(y_dev) > 0 else 0.0

    # Zero-shot text prompt or exemplar image reference baseline
    prior_model = load_classifier(actual_model_path)
    if reset_baseline_to_default:
        ref_type = "text"
        ref_source = DEFAULT_ZERO_SHOT_PROMPT
        v_ref = extract_text_embedding(DEFAULT_ZERO_SHOT_PROMPT)
    elif baseline_image_base64:
        ref_type = "image"
        ref_source = "exemplar_image"
        v_ref = extract_vision_embedding(baseline_image_base64)
    elif baseline_prompt_text and baseline_prompt_text.strip():
        ref_type = "text"
        ref_source = baseline_prompt_text.strip()
        v_ref = extract_text_embedding(baseline_prompt_text.strip())
    elif prior_model and prior_model.get("reference_embedding") is not None:
        ref_type = prior_model.get("reference_type", "text")
        ref_source = prior_model.get("reference_source", DEFAULT_ZERO_SHOT_PROMPT)
        v_ref = np.array(prior_model["reference_embedding"], dtype=np.float32)
    else:
        ref_type = "text"
        ref_source = DEFAULT_ZERO_SHOT_PROMPT
        v_ref = extract_text_embedding(DEFAULT_ZERO_SHOT_PROMPT)

    zs_scores = X_dev @ v_ref
    zs_prec, zs_rec, _ = precision_recall_curve(y_dev, zs_scores)
    zero_shot_pr_auc = float(auc(zs_rec, zs_prec)) if len(zs_rec) > 1 else 0.0

    baselines = {
        "random_guess": round(random_guess_pr_auc, 4),
        "positive_centroid": round(centroid_pr_auc, 4),
        "zero_shot": round(zero_shot_pr_auc, 4),
        "reference_type": ref_type,
        "reference_source": ref_source,
        "prompt_text": ref_source if ref_type == "text" else DEFAULT_ZERO_SHOT_PROMPT,
    }

    # 4. Out-of-fold Precision-Recall curve and metrics
    precisions, recalls, pr_thresholds = precision_recall_curve(y_dev, oof_probabilities)
    pr_auc_score = float(auc(recalls, precisions)) if len(recalls) > 1 else 0.0
    ap_score = float(average_precision_score(y_dev, oof_probabilities)) if len(np.unique(y_dev)) > 1 else 0.0

    # 5. Calibrate decision threshold
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
        valid_candidates: list[tuple[float, float, float]] = []
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
            valid_candidates.sort(key=lambda item: (item[0], item[2]), reverse=True)
            calibrated_threshold = valid_candidates[0][2]
        elif all_candidates:
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

    # 6. Holdout generalization verification
    if use_holdout and X_holdout is not None and y_holdout is not None:
        dev_clf = LogisticRegression(
            class_weight=best_weight_param,
            C=best_C,
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

    # 7. Fit final model on 100% of labeled data
    final_classifier = LogisticRegression(
        class_weight=best_weight_param,
        C=best_C,
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
        "best_params": {
            "C": best_C,
            "class_weight": best_weight_key,
        },
        "tuning_summary": tuning_results,
        "baselines": baselines,
    }

    # Map out-of-fold scores to development partition sample IDs if provided
    oof_score_map: dict[int, float] = {}
    if dev_ids is not None and len(dev_ids) == len(oof_probabilities):
        for i, s_id in enumerate(dev_ids):
            oof_score_map[s_id] = float(oof_probabilities[i])

    # Compute outlier statistics on positive samples
    outlier_analysis = detect_positive_outliers(
        X,
        y,
        sample_ids=sample_ids,
        oof_score_map=oof_score_map,
        model_data={"classifier": final_classifier},
    )
    flagged_outliers_count = sum(1 for v in outlier_analysis.values() if v.get("is_outlier"))
    mean_cent_dist = 0.0
    if outlier_analysis:
        mean_cent_dist = round(float(np.mean([v["centroid_distance"] for v in outlier_analysis.values()])), 4)

    metrics["outliers"] = {
        "flagged_count": flagged_outliers_count,
        "positive_count": positive_count,
        "mean_centroid_distance": mean_cent_dist,
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
        "oof_score_map": oof_score_map,
        "reference_embedding": v_ref.tolist(),
        "reference_type": ref_type,
        "reference_source": ref_source,
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
        "best_params": existing_metrics.get("best_params"),
        "tuning_summary": existing_metrics.get("tuning_summary"),
        "baselines": existing_metrics.get("baselines"),
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


def detect_positive_outliers(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[int] | None = None,
    oof_score_map: dict[int, float] | None = None,
    model_data: dict[str, Any] | None = None,
    model_path: Path | str | None = None,
) -> dict[int, dict[str, Any]]:
    """Detect inconsistent positive class ratings (outliers) in the dataset.

    Computes cosine distance from each positive sample vision embedding to the
    positive class centroid. Flags positive samples with distances greater than
    2 standard deviations above the mean (distance > mean + 2 * std) OR prediction
    scores below 0.20 (using out-of-fold cross-validation scores when available,
    falling back to fitted model inference score).

    Args:
        X: Feature matrix of shape (N, 768).
        y: Label vector of shape (N,) with integers 1 (Like) or 0 (Dislike).
        sample_ids: Parallel list of integer sample IDs. If None, uses integer indices 0..N-1.
        oof_score_map: Optional mapping of sample_id to out-of-fold probability score.
        model_data: Optional pre-loaded classifier dictionary.
        model_path: Optional path to serialized model file.

    Returns:
        Dictionary mapping positive sample_id to:
        {
            "is_outlier": bool,
            "outlier_reason": "distance" | "low_score" | "both" | None,
            "centroid_distance": float,
            "prediction_score": float | None,
            "distance_threshold": float,
            "score_threshold": 0.20,
        }
    """
    if len(y) == 0:
        return {}

    ids = sample_ids if sample_ids is not None else list(range(len(y)))
    pos_mask = (y == 1)
    pos_indices = np.where(pos_mask)[0]

    if len(pos_indices) == 0:
        return {}

    X_pos = X[pos_indices]

    # Calculate L2-normalized positive class centroid
    centroid = np.mean(X_pos, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    # Cosine distance = 1 - cosine similarity
    cos_sims = X_pos @ centroid
    cos_dists = np.clip(1.0 - cos_sims, 0.0, 2.0)

    mean_dist = float(np.mean(cos_dists))
    std_dist = float(np.std(cos_dists))
    dist_threshold = float(mean_dist + 2.0 * std_dist)

    # Load model if needed for fallback prediction scores
    active_model = model_data if model_data is not None else load_classifier(model_path)
    classifier = active_model.get("classifier") if active_model else None

    cached_oof_map = oof_score_map or {}
    if not cached_oof_map and active_model and "oof_score_map" in active_model:
        cached_oof_map = active_model["oof_score_map"]

    results: dict[int, dict[str, Any]] = {}

    for idx_in_pos, global_idx in enumerate(pos_indices):
        s_id = ids[global_idx]
        d_val = float(cos_dists[idx_in_pos])

        # Obtain prediction score: prefer OOF score, fallback to model score
        pred_score: float | None = None
        if s_id in cached_oof_map:
            pred_score = float(cached_oof_map[s_id])
        elif classifier is not None:
            emb_vec = X_pos[idx_in_pos].reshape(1, -1)
            try:
                pred_score = float(classifier.predict_proba(emb_vec)[0, 1])
            except Exception:
                pred_score = None

        # Outlier conditions: dist > mean + 2*std OR score < 0.20
        dist_flag = bool(len(pos_indices) >= 3 and d_val > dist_threshold)
        score_flag = bool(pred_score is not None and pred_score < 0.20)

        is_outlier = dist_flag or score_flag
        reason: str | None = None
        if dist_flag and score_flag:
            reason = "both"
        elif dist_flag:
            reason = "distance"
        elif score_flag:
            reason = "low_score"

        results[s_id] = {
            "is_outlier": is_outlier,
            "outlier_reason": reason,
            "outlier_type": "like",
            "centroid_distance": round(d_val, 4),
            "prediction_score": round(pred_score, 4) if pred_score is not None else None,
            "distance_threshold": round(dist_threshold, 4),
            "score_threshold": 0.20,
        }

    return results


def detect_negative_outliers(
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: list[int] | None = None,
    oof_score_map: dict[int, float] | None = None,
    model_data: dict[str, Any] | None = None,
    model_path: Path | str | None = None,
    score_threshold: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Detect inconsistent negative class ratings (dislike outliers) in the dataset.

    Computes cosine distance from each negative sample vision embedding to the
    negative class centroid. Flags negative samples with distances greater than
    2 standard deviations above the mean (distance > mean + 2 * std) OR prediction
    scores >= score_threshold (defaulting to the active decision threshold, minimum 0.35).

    Args:
        X: Feature matrix of shape (N, 768).
        y: Label vector of shape (N,) with integers 1 (Like) or 0 (Dislike).
        sample_ids: Parallel list of integer sample IDs. If None, uses integer indices 0..N-1.
        oof_score_map: Optional mapping of sample_id to out-of-fold probability score.
        model_data: Optional pre-loaded classifier dictionary.
        model_path: Optional path to serialized model file.
        score_threshold: Optional threshold above which a dislike is flagged as inconsistent.

    Returns:
        Dictionary mapping negative sample_id to:
        {
            "is_outlier": bool,
            "outlier_reason": "distance" | "high_score" | "both" | None,
            "outlier_type": "dislike",
            "centroid_distance": float,
            "prediction_score": float | None,
            "distance_threshold": float,
            "score_threshold": float,
        }
    """
    if len(y) == 0:
        return {}

    ids = sample_ids if sample_ids is not None else list(range(len(y)))
    neg_mask = (y == 0)
    neg_indices = np.where(neg_mask)[0]

    if len(neg_indices) == 0:
        return {}

    X_neg = X[neg_indices]

    # Calculate L2-normalized negative class centroid
    centroid = np.mean(X_neg, axis=0)
    centroid_norm = np.linalg.norm(centroid)
    if centroid_norm > 0:
        centroid = centroid / centroid_norm

    # Cosine distance = 1 - cosine similarity
    cos_sims = X_neg @ centroid
    cos_dists = np.clip(1.0 - cos_sims, 0.0, 2.0)

    mean_dist = float(np.mean(cos_dists))
    std_dist = float(np.std(cos_dists))
    dist_threshold = float(mean_dist + 2.0 * std_dist)

    # Load model if needed for fallback prediction scores
    active_model = model_data if model_data is not None else load_classifier(model_path)
    classifier = active_model.get("classifier") if active_model else None

    # Use active decision threshold (floor of 0.35 for dislike conflict detection)
    model_thresh = float(active_model.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)) if active_model else DEFAULT_DECISION_THRESHOLD
    active_score_threshold = score_threshold if score_threshold is not None else max(model_thresh, 0.35)

    cached_oof_map = oof_score_map or {}
    if not cached_oof_map and active_model and "oof_score_map" in active_model:
        cached_oof_map = active_model["oof_score_map"]

    results: dict[int, dict[str, Any]] = {}

    for idx_in_neg, global_idx in enumerate(neg_indices):
        s_id = ids[global_idx]
        d_val = float(cos_dists[idx_in_neg])

        pred_score: float | None = None
        if s_id in cached_oof_map:
            pred_score = float(cached_oof_map[s_id])
        elif classifier is not None:
            emb_vec = X_neg[idx_in_neg].reshape(1, -1)
            try:
                pred_score = float(classifier.predict_proba(emb_vec)[0, 1])
            except Exception:
                pred_score = None

        dist_flag = bool(len(neg_indices) >= 3 and d_val > dist_threshold)
        score_flag = bool(pred_score is not None and pred_score >= active_score_threshold)

        is_outlier = dist_flag or score_flag
        reason: str | None = None
        if dist_flag and score_flag:
            reason = "both"
        elif dist_flag:
            reason = "distance"
        elif score_flag:
            reason = "high_score"

        results[s_id] = {
            "is_outlier": is_outlier,
            "outlier_reason": reason,
            "outlier_type": "dislike",
            "centroid_distance": round(d_val, 4),
            "prediction_score": round(pred_score, 4) if pred_score is not None else None,
            "distance_threshold": round(dist_threshold, 4),
            "score_threshold": round(active_score_threshold, 4),
        }

    return results

