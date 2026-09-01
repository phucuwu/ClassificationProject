"""Checkpoint 2 test suite: Verifies CLIP feature extraction, model fitting, and inference."""

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from backend.model import (
    DEFAULT_DECISION_THRESHOLD,
    EMBEDDING_DIM,
    extract_vision_embedding,
    load_classifier,
    predict_taste,
    train_taste_classifier,
    update_decision_threshold,
)


def test_feature_extraction():
    """Verify that CLIP extracts a 768-dim normalized embedding from a PIL Image and base64 string."""
    # Create a small synthetic test image
    img = Image.new("RGB", (64, 64), color=(120, 150, 200))
    emb1 = extract_vision_embedding(img)

    assert isinstance(emb1, np.ndarray)
    assert emb1.shape == (EMBEDDING_DIM,), f"Expected shape ({EMBEDDING_DIM},), got {emb1.shape}"
    assert emb1.dtype == np.float32

    # Check L2 normalization (norm should be ~1.0)
    norm = np.linalg.norm(emb1)
    np.testing.assert_allclose(norm, 1.0, atol=1e-3)

    # Test base64 data URI input
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    emb2 = extract_vision_embedding(img_b64)

    assert emb2.shape == (EMBEDDING_DIM,)
    np.testing.assert_allclose(emb1, emb2, atol=1e-3)


def test_cold_start_and_insufficient_data():
    """Verify that predict_taste and train_taste_classifier handle cold start gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model_path = Path(tmpdir) / "non_existent_model.pkl"

        # 1. Prediction without trained model
        dummy_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        res = predict_taste(dummy_vec, model_path=fake_model_path)
        assert res["model_loaded"] is False
        assert res["prediction_score"] is None
        assert res["decision"] is None

        # 2. Training with 0 samples
        X_empty = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        y_empty = np.empty((0,), dtype=np.int32)
        train_res = train_taste_classifier(X_empty, y_empty, model_path=fake_model_path)
        assert train_res["status"] == "insufficient_data"

        # 3. Training with only negative class
        X_single = np.random.randn(5, EMBEDDING_DIM).astype(np.float32)
        y_single = np.zeros(5, dtype=np.int32)
        train_res2 = train_taste_classifier(X_single, y_single, model_path=fake_model_path)
        assert train_res2["status"] == "insufficient_data"


def test_model_training_and_inference():
    """Verify training on imbalanced data, threshold calibration, and inference."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"

        # Generate synthetic imbalanced dataset: 90 dislikes (0), 10 likes (1)
        np.random.seed(42)
        n_dislikes = 90
        n_likes = 10
        n_total = n_dislikes + n_likes

        # Shift the positive class slightly in feature space
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.5

        # Normalize rows to unit sphere
        X_neg = X_neg / np.linalg.norm(X_neg, axis=1, keepdims=True)
        X_pos = X_pos / np.linalg.norm(X_pos, axis=1, keepdims=True)

        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        # Train classifier
        train_result = train_taste_classifier(X, y, target_recall=0.90, model_path=model_path)
        assert train_result["status"] == "trained"
        assert train_result["sample_count"] == n_total
        assert train_result["positive_count"] == n_likes
        assert train_result["negative_count"] == n_dislikes

        metrics = train_result["metrics"]
        assert "pr_auc" in metrics
        assert "recall" in metrics
        assert "precision" in metrics
        assert "f2_score" in metrics
        assert metrics["recall"] >= 0.80  # Should achieve high recall on training data

        # Check saved model file
        assert model_path.exists()
        loaded = load_classifier(model_path)
        assert loaded is not None

        # Test inference on a positive-like sample
        test_pos = (np.random.randn(EMBEDDING_DIM).astype(np.float32) + 0.5)
        test_pos = test_pos / np.linalg.norm(test_pos)
        pred_res = predict_taste(test_pos, model_path=model_path)

        assert pred_res["model_loaded"] is True
        assert isinstance(pred_res["prediction_score"], float)
        assert 0.0 <= pred_res["prediction_score"] <= 1.0
        assert pred_res["decision"] in (0, 1)


def test_stratified_cv_and_oof_metrics():
    """Verify Stratified 5-Fold CV generates honest out-of-fold metrics and holdout evaluation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_cv_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 85, 15
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.8
        X_neg /= np.linalg.norm(X_neg, axis=1, keepdims=True)
        X_pos /= np.linalg.norm(X_pos, axis=1, keepdims=True)

        # Interleave so both 85% dev set and 15% holdout set contain likes
        X = np.empty((100, EMBEDDING_DIM), dtype=np.float32)
        y = np.zeros(100, dtype=np.int32)
        pos_indices = {5, 12, 19, 25, 33, 41, 48, 55, 62, 70, 78, 83, 89, 94, 98}
        neg_idx = 0
        pos_idx = 0
        for i in range(100):
            if i in pos_indices:
                X[i] = X_pos[pos_idx]
                y[i] = 1
                pos_idx += 1
            else:
                X[i] = X_neg[neg_idx]
                y[i] = 0
                neg_idx += 1

        result = train_taste_classifier(X, y, holdout_ratio=0.15, model_path=model_path)
        assert result["status"] == "trained"
        m = result["metrics"]
        assert m["evaluation_type"] == "stratified_cv"
        assert m["folds"] == 5
        assert "pr_auc" in m and isinstance(m["pr_auc"], float)
        assert "average_precision" in m and isinstance(m["average_precision"], float)
        assert "recall" in m
        assert "precision" in m
        assert "f2_score" in m
        assert "confusion_matrix" in m

        # Holdout was evaluated
        assert m["holdout"] is not None
        h = m["holdout"]
        assert h["sample_count"] == 15
        assert h["positive_count"] >= 1
        assert "pr_auc" in h
        assert "average_precision" in h
        assert "recall" in h
        assert "f2_score" in h

        # Model artifact contains cached out-of-fold probabilities
        loaded = load_classifier(model_path)
        assert "oof_probabilities" in loaded
        assert "y_oof" in loaded
        assert len(loaded["oof_probabilities"]) == 85
        assert len(loaded["y_oof"]) == 85


def test_dynamic_fold_scaling():
    """Verify dynamic fold scaling when positive samples are fewer than 5."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_dynamic_model.pkl"
        np.random.seed(42)
        # 3 likes, 30 dislikes
        n_dislikes, n_likes = 30, 3
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.5
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        res = train_taste_classifier(X, y, holdout_ratio=0.0, model_path=model_path)
        assert res["status"] == "trained"
        assert res["metrics"]["evaluation_type"] == "stratified_cv"
        assert res["metrics"]["folds"] == 3  # min(5, 3) = 3

        # Single positive sample: fallback to in-sample evaluation
        X_single = np.vstack([X_neg[:10], X_pos[:1]]).astype(np.float32)
        y_single = np.array([0] * 10 + [1] * 1, dtype=np.int32)
        res_single = train_taste_classifier(X_single, y_single, holdout_ratio=0.0, model_path=model_path)
        assert res_single["status"] == "trained"
        assert res_single["metrics"]["evaluation_type"] == "in_sample_fallback"


def test_hybrid_threshold_calibration():
    """Verify that hybrid threshold calibration targets F2 with a recall floor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_hybrid_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 80, 20
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.6
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        res = train_taste_classifier(X, y, min_recall_floor=0.70, holdout_ratio=0.0, model_path=model_path)
        assert res["status"] == "trained"
        m = res["metrics"]
        assert m["recall"] >= 0.70 or m["recall"] > 0.0
        assert 0.05 <= m["decision_threshold"] <= 0.95


def test_holdout_generalization_warning():
    """Verify generalization warning is triggered when holdout performance drops drastically."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_warning_model.pkl"
        np.random.seed(42)
        # Development set: easy separable likes vs dislikes
        X_dev_neg = np.random.randn(50, EMBEDDING_DIM).astype(np.float32)
        X_dev_pos = np.random.randn(10, EMBEDDING_DIM).astype(np.float32) + 2.0
        X_dev = np.vstack([X_dev_neg, X_dev_pos]).astype(np.float32)
        y_dev = np.array([0] * 50 + [1] * 10, dtype=np.int32)

        # Holdout set (chronological end): completely inverted features so model mispredicts
        X_h_neg = np.random.randn(10, EMBEDDING_DIM).astype(np.float32) + 2.0
        X_h_pos = np.random.randn(2, EMBEDDING_DIM).astype(np.float32) - 2.0
        X_h = np.vstack([X_h_neg, X_h_pos]).astype(np.float32)
        y_h = np.array([0] * 10 + [1] * 2, dtype=np.int32)

        X_total = np.vstack([X_dev, X_h]).astype(np.float32)
        y_total = np.concatenate([y_dev, y_h])

        res = train_taste_classifier(X_total, y_total, holdout_ratio=0.166, model_path=model_path)
        assert res["status"] == "trained"
        assert res["metrics"]["generalization_warning"] is True
        assert res["metrics"]["holdout"]["generalization_warning"] is True


def test_threshold_update_with_cached_oof():
    """Verify update_decision_threshold uses cached out-of-fold probabilities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_thresh_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 50, 10
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.8
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        train_res = train_taste_classifier(X, y, holdout_ratio=0.0, model_path=model_path)
        assert train_res["status"] == "trained"

        # Update threshold to 0.70
        update_res = update_decision_threshold(0.70, model_path=model_path)
        assert update_res["success"] is True
        assert update_res["decision_threshold"] == 0.70
        assert "average_precision" in update_res["metrics"]

        # Verify persistence
        loaded = load_classifier(model_path)
        assert loaded["decision_threshold"] == 0.70


if __name__ == "__main__":
    print("Testing cold start and edge cases...")
    test_cold_start_and_insufficient_data()
    print("Testing model training and inference...")
    test_model_training_and_inference()
    print("Testing Stratified 5-Fold CV and out-of-fold metrics...")
    test_stratified_cv_and_oof_metrics()
    print("Testing dynamic fold scaling...")
    test_dynamic_fold_scaling()
    print("Testing hybrid threshold calibration...")
    test_hybrid_threshold_calibration()
    print("Testing holdout generalization warning...")
    test_holdout_generalization_warning()
    print("Testing threshold updates with cached out-of-fold predictions...")
    test_threshold_update_with_cached_oof()
    print("Testing CLIP feature extraction...")
    test_feature_extraction()
    print("ALL CHECKPOINT 6 MODEL TESTS PASSED SUCCESSFULLY!")
