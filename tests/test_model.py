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


if __name__ == "__main__":
    print("Testing cold start and edge cases...")
    test_cold_start_and_insufficient_data()
    print("Testing model training and inference...")
    test_model_training_and_inference()
    print("Testing CLIP feature extraction (downloading/loading model if first time)...")
    test_feature_extraction()
    print("ALL CHECKPOINT 2 TESTS PASSED SUCCESSFULLY!")
