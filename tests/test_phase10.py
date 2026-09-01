"""Phase 10 Test Suite: Model Serialization and Architecture Hardening.

Verifies:
1. Model persistence in structured JSON format (coefficients, intercept, metrics, threshold).
2. Automatic migration of legacy pickle model artifacts to JSON.
3. Vectorized inference parity between JSON model weights and scikit-learn LogisticRegression.
4. Active decision threshold updates directly against JSON artifacts.
5. Offline backbone benchmark evaluation engine and cross-validation metrics.
6. Backend REST API endpoints (GET and POST /api/benchmark).
"""

import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

import backend.model as model_mod
from backend.app import app
from backend.model import (
    DEFAULT_DECISION_THRESHOLD,
    load_classifier,
    predict_taste,
    save_classifier_json,
    train_taste_classifier,
    update_decision_threshold,
)
from tasks.benchmark_backbones import evaluate_backbone_cv, run_backbone_benchmark


def _generate_synthetic_dataset(n_samples: int = 60, pos_ratio: float = 0.10, dim: int = 768):
    """Generate synthetic L2-normalized feature matrix and label vector."""
    np.random.seed(42)
    n_pos = max(2, int(n_samples * pos_ratio))
    n_neg = n_samples - n_pos

    # Direction vectors for classes
    pos_direction = np.zeros(dim, dtype=np.float32)
    pos_direction[0] = 1.0

    neg_direction = np.zeros(dim, dtype=np.float32)
    neg_direction[1] = 1.0

    X_pos = pos_direction + np.random.normal(0, 0.05, (n_pos, dim)).astype(np.float32)
    X_pos /= np.linalg.norm(X_pos, axis=1, keepdims=True)

    X_neg = neg_direction + np.random.normal(0, 0.05, (n_neg, dim)).astype(np.float32)
    X_neg /= np.linalg.norm(X_neg, axis=1, keepdims=True)

    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int32)
    return X, y


def test_json_model_serialization_and_deserialization():
    """Verify train_taste_classifier saves structured JSON artifact with full parameter recovery."""
    X, y = _generate_synthetic_dataset(n_samples=50, pos_ratio=0.12)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "model.json"
        train_res = train_taste_classifier(X, y, holdout_ratio=0.0, model_path=json_path)
        assert train_res["status"] == "trained"
        assert json_path.exists(), "Expected model.json artifact to be created"

        # Verify raw JSON format
        with open(json_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)

        assert raw_json["format_version"] == "1.0"
        assert "coefficients" in raw_json
        assert isinstance(raw_json["coefficients"], list)
        assert len(raw_json["coefficients"][0]) == 768
        assert "intercept" in raw_json
        assert "decision_threshold" in raw_json
        assert "metrics" in raw_json
        assert "pr_auc" in raw_json["metrics"]
        assert raw_json["positive_count"] == int(np.sum(y == 1))
        assert raw_json["negative_count"] == int(np.sum(y == 0))

        # Verify load_classifier deserializes accurately
        loaded = load_classifier(json_path)
        assert loaded is not None
        assert "classifier" in loaded
        assert isinstance(loaded["classifier"], LogisticRegression)
        assert loaded["classifier"].coef_.shape == (1, 768)
        assert loaded["decision_threshold"] == raw_json["decision_threshold"]


def test_legacy_pickle_migration():
    """Verify load_classifier automatically loads legacy .pkl files and migrates to .json."""
    X, y = _generate_synthetic_dataset(n_samples=40, pos_ratio=0.15)

    clf = LogisticRegression(class_weight="balanced", random_state=42)
    clf.fit(X, y)

    legacy_payload = {
        "classifier": clf,
        "decision_threshold": 0.35,
        "metrics": {"pr_auc": 0.85, "recall": 0.80},
        "sample_count": 40,
        "positive_count": 6,
        "negative_count": 34,
        "oof_probabilities": [0.1, 0.8],
        "y_oof": [0, 1],
        "oof_score_map": {1: 0.1, 2: 0.8},
        "reference_embedding": np.zeros(768, dtype=np.float32).tolist(),
        "reference_type": "text",
        "reference_source": "goth aesthetic alternative indie girl style",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        pkl_path = Path(tmpdir) / "test_legacy_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(legacy_payload, f)

        assert pkl_path.exists()
        json_migrated_path = Path(tmpdir) / "test_legacy_model.json"
        assert not json_migrated_path.exists()

        # Loading legacy path triggers automatic JSON migration
        loaded = load_classifier(pkl_path)
        assert loaded is not None
        assert loaded["decision_threshold"] == 0.35
        assert loaded["sample_count"] == 40
        assert json_migrated_path.exists(), "Expected auto-migrated test_legacy_model.json to be created"

        # Verify migrated JSON content
        with open(json_migrated_path, "r", encoding="utf-8") as f:
            migrated_json = json.load(f)
        assert migrated_json["format_version"] == "1.0"
        assert len(migrated_json["coefficients"][0]) == 768


def test_vectorized_inference_parity():
    """Verify predict_taste produces identical predictions between scikit-learn and direct NumPy."""
    X, y = _generate_synthetic_dataset(n_samples=50, pos_ratio=0.10)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "parity_model.json"
        train_taste_classifier(X, y, holdout_ratio=0.0, model_path=json_path)

        loaded = load_classifier(json_path)
        clf: LogisticRegression = loaded["classifier"]

        # Test on 10 random vision embeddings
        np.random.seed(123)
        test_vectors = np.random.randn(10, 768).astype(np.float32)
        test_vectors /= np.linalg.norm(test_vectors, axis=1, keepdims=True)

        for i in range(10):
            vec = test_vectors[i]
            sk_prob = float(clf.predict_proba(vec.reshape(1, -1))[0, 1])

            pred_res = predict_taste(vec, model_path=json_path)
            np_prob = pred_res["prediction_score"]

            assert np.isclose(sk_prob, np_prob, atol=1e-4), f"Probability mismatch at sample {i}: {sk_prob} vs {np_prob}"


def test_update_decision_threshold_json():
    """Verify update_decision_threshold modifies decision_threshold and metrics in JSON artifact."""
    X, y = _generate_synthetic_dataset(n_samples=50, pos_ratio=0.12)

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = Path(tmpdir) / "threshold_test.json"
        train_taste_classifier(X, y, holdout_ratio=0.0, model_path=json_path)

        # Update threshold to 0.45
        res = update_decision_threshold(0.45, model_path=json_path)
        assert res["success"] is True
        assert res["decision_threshold"] == 0.45

        # Re-load and verify persistence
        loaded = load_classifier(json_path)
        assert loaded["decision_threshold"] == 0.45
        assert loaded["metrics"]["decision_threshold"] == 0.45


def test_backbone_benchmark_cv_engine():
    """Verify evaluate_backbone_cv correctly scores feature matrices with cross-validation."""
    X, y = _generate_synthetic_dataset(n_samples=40, pos_ratio=0.15, dim=768)

    metrics = evaluate_backbone_cv(X, y, n_splits=3)
    assert "error" not in metrics
    assert "pr_auc" in metrics
    assert "f2_score" in metrics
    assert "recall" in metrics
    assert "precision" in metrics
    assert "best_c" in metrics
    assert "best_class_weight" in metrics
    assert metrics["sample_count"] == 40
    assert metrics["positive_count"] == int(np.sum(y == 1))
    assert 0.0 <= metrics["pr_auc"] <= 1.0


def test_api_benchmark_endpoints():
    """Verify REST API GET and POST /api/benchmark."""
    client = TestClient(app)

    # 1. GET status
    res_get = client.get("/api/benchmark")
    assert res_get.status_code == 200
    get_data = res_get.json()
    assert "status" in get_data
    assert "percent" in get_data

    # 2. POST trigger benchmark (limit 10 for rapid test execution)
    res_post = client.post(
        "/api/benchmark",
        json={"models": ["clip-ViT-L-14"], "limit": 10},
    )
    assert res_post.status_code == 200
    post_data = res_post.json()
    assert post_data["status"] in ("started", "already_running")


def run_all_tests():
    """Run all Phase 10 test functions."""
    print("Running test_json_model_serialization_and_deserialization...")
    test_json_model_serialization_and_deserialization()
    print("Running test_legacy_pickle_migration...")
    test_legacy_pickle_migration()
    print("Running test_vectorized_inference_parity...")
    test_vectorized_inference_parity()
    print("Running test_update_decision_threshold_json...")
    test_update_decision_threshold_json()
    print("Running test_backbone_benchmark_cv_engine...")
    test_backbone_benchmark_cv_engine()
    print("Running test_api_benchmark_endpoints...")
    test_api_benchmark_endpoints()
    print("\nALL PHASE 10 TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_tests()
