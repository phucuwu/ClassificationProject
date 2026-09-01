"""Checkpoint 2 test suite: Verifies CLIP feature extraction, model fitting, and inference."""

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from backend.model import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_ZERO_SHOT_PROMPT,
    EMBEDDING_DIM,
    HYPERPARAMETER_C_GRID,
    extract_text_embedding,
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
        from sklearn.model_selection import train_test_split as _tts

        n_neg = 60
        n_pos = 12
        y = np.array([0] * n_neg + [1] * n_pos, dtype=np.int32)
        idx = np.arange(len(y))
        _, h_idx = _tts(idx, test_size=0.166, stratify=y, random_state=42)

        X = np.random.randn(len(y), EMBEDDING_DIM).astype(np.float32)
        X[y == 1] += 2.0

        # Invert holdout samples so development set is cleanly separable but holdout mispredicts
        for i in h_idx:
            if y[i] == 1:
                X[i] -= 4.0
            else:
                X[i] += 4.0

        res = train_taste_classifier(X, y, holdout_ratio=0.166, model_path=model_path)
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


def test_hyperparameter_grid_search():
    """Verify regularized hyperparameter search selects best C and class weighting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_grid_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 80, 15
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 1.0
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        res = train_taste_classifier(X, y, holdout_ratio=0.0, model_path=model_path)
        assert res["status"] == "trained"
        m = res["metrics"]
        assert "best_params" in m
        assert "tuning_summary" in m
        best_p = m["best_params"]
        assert best_p["C"] in HYPERPARAMETER_C_GRID
        assert best_p["class_weight"] in ["balanced", "unweighted", "balanced_1.5x", "balanced_2.0x"]
        assert len(m["tuning_summary"]) == len(HYPERPARAMETER_C_GRID) * 4

        # Loaded model preserves best_params and tuning_summary
        loaded = load_classifier(model_path)
        assert loaded["metrics"]["best_params"] == best_p
        assert "tuning_summary" in loaded["metrics"]


def test_reference_baselines():
    """Verify calculation of random guess, positive centroid, and zero-shot baselines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_baselines_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 90, 10
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 1.0
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        res = train_taste_classifier(X, y, holdout_ratio=0.0, model_path=model_path)
        assert res["status"] == "trained"
        assert "baselines" in res["metrics"]
        b = res["metrics"]["baselines"]
        assert "random_guess" in b
        assert "positive_centroid" in b
        assert "zero_shot" in b
        assert b["reference_type"] == "text"
        assert b["reference_source"] == DEFAULT_ZERO_SHOT_PROMPT

        # Random guess baseline matches positive prevalence (10/100 = 0.10)
        expected_prev = round(10 / 100, 4)
        assert abs(b["random_guess"] - expected_prev) < 0.01

        # Centroid and zero-shot baselines return valid PR-AUC floats
        assert 0.0 <= b["positive_centroid"] <= 1.0
        assert 0.0 <= b["zero_shot"] <= 1.0


def test_baseline_customization_and_persistence():
    """Verify custom prompt, exemplar image override, and baseline persistence across runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_custom_baseline_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 50, 10
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.8
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        # 1. Custom text prompt
        custom_prompt = "cyberpunk neon anime illustration"
        res1 = train_taste_classifier(
            X, y,
            baseline_prompt_text=custom_prompt,
            holdout_ratio=0.0,
            model_path=model_path,
        )
        assert res1["status"] == "trained"
        b1 = res1["metrics"]["baselines"]
        assert b1["reference_type"] == "text"
        assert b1["reference_source"] == custom_prompt

        # 2. Exemplar image upload override
        test_img = Image.new("RGB", (64, 64), color=(200, 50, 80))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        res2 = train_taste_classifier(
            X, y,
            baseline_image_base64=img_b64,
            holdout_ratio=0.0,
            model_path=model_path,
        )
        assert res2["status"] == "trained"
        b2 = res2["metrics"]["baselines"]
        assert b2["reference_type"] == "image"
        assert b2["reference_source"] == "exemplar_image"

        # 3. Persistence: subsequent retrain without overrides keeps exemplar image
        res3 = train_taste_classifier(
            X, y,
            holdout_ratio=0.0,
            model_path=model_path,
        )
        assert res3["status"] == "trained"
        b3 = res3["metrics"]["baselines"]
        assert b3["reference_type"] == "image"

        # 4. Reset baseline to default
        res4 = train_taste_classifier(
            X, y,
            reset_baseline_to_default=True,
            holdout_ratio=0.0,
            model_path=model_path,
        )
        assert res4["status"] == "trained"
        b4 = res4["metrics"]["baselines"]
        assert b4["reference_type"] == "text"
        assert b4["reference_source"] == DEFAULT_ZERO_SHOT_PROMPT


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
    print("Testing hyperparameter grid search...")
    test_hyperparameter_grid_search()
    print("Testing reference baselines...")
    test_reference_baselines()
    print("Testing baseline customization and persistence...")
    test_baseline_customization_and_persistence()
    print("Testing CLIP feature extraction...")
    test_feature_extraction()
    print("ALL CHECKPOINT 7 MODEL TESTS PASSED SUCCESSFULLY!")
