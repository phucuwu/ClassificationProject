"""Phase 8 Test Suite: Taste Consistency and Data Quality Tooling.

Tests:
1. Positive class outlier detection via centroid distance and out-of-fold score thresholding.
2. Near-duplicate artwork detection via cosine similarity >= 0.98.
3. Near-duplicate consolidation on ingestion in POST /api/record.
4. Outlier querying and filtering in GET /api/samples?outliers_only=true.
"""

import base64
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import backend.database as db_mod
import backend.model as model_mod
from backend.app import app
from backend.database import find_near_duplicate, init_db, insert_sample, update_sample_record
from backend.model import detect_negative_outliers, detect_positive_outliers


def _make_test_image_b64(color=(120, 80, 200)) -> str:
    """Generate a test JPEG as base64 string."""
    img = Image.new("RGB", (32, 32), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def test_detect_positive_outliers_distance_and_score():
    """Verify positive class outlier detection flags distance > 2 std and low scores."""
    np.random.seed(42)
    # Create a cluster of 15 similar positive vision embeddings (high cosine similarity)
    base_vector = np.zeros(768, dtype=np.float32)
    base_vector[0] = 1.0  # Main direction

    tight_cluster = []
    for _ in range(15):
        noise = np.random.normal(0, 0.02, 768).astype(np.float32)
        vec = base_vector + noise
        vec = vec / np.linalg.norm(vec)
        tight_cluster.append(vec)

    # Add an outlier positive vector pointing in a very different direction
    distant_vector = np.zeros(768, dtype=np.float32)
    distant_vector[1] = 1.0  # Orthogonal direction
    distant_vector = distant_vector / np.linalg.norm(distant_vector)
    tight_cluster.append(distant_vector)

    X_pos = np.vstack(tight_cluster)
    y_pos = np.ones(len(X_pos), dtype=np.int32)
    sample_ids = [100 + i for i in range(len(X_pos))]

    # Distant vector is at index 15 (sample_id 115)
    # Set high prediction scores for everyone except one cluster member (sample_id 103)
    oof_scores = {sid: 0.85 for sid in sample_ids}
    oof_scores[103] = 0.12  # Low score outlier (< 0.20)

    outliers = detect_positive_outliers(
        X_pos,
        y_pos,
        sample_ids=sample_ids,
        oof_score_map=oof_scores,
    )

    assert len(outliers) == 16
    # 1. Distant vector (id 115) should be flagged due to distance > 2 std
    assert outliers[115]["is_outlier"] is True
    assert outliers[115]["outlier_reason"] in ("distance", "both")
    assert outliers[115]["centroid_distance"] > outliers[115]["distance_threshold"]

    # 2. Low score vector (id 103) should be flagged due to score < 0.20
    assert outliers[103]["is_outlier"] is True
    assert outliers[103]["outlier_reason"] in ("low_score", "both")
    assert outliers[103]["prediction_score"] == 0.12

    # 3. Regular tight cluster samples with high scores should NOT be outliers
    normal_sample = outliers[100]
    assert normal_sample["is_outlier"] is False
    assert normal_sample["outlier_reason"] is None


def test_find_near_duplicate_detection():
    """Verify find_near_duplicate identifies cosine similarity >= 0.98."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test.db"
        init_db(test_db)

        # Insert a canonical sample
        v1 = np.zeros(768, dtype=np.float32)
        v1[0] = 1.0
        sid1 = insert_sample("hash_orig", "data/images/hash_orig.jpg", v1, label=1, mode="manual", db_path=test_db)

        # 1. Test identical embedding (sim = 1.0)
        dup_match = find_near_duplicate(v1, threshold=0.98, db_path=test_db)
        assert dup_match is not None
        assert dup_match["id"] == sid1
        assert dup_match["similarity"] >= 0.99

        # 2. Test perturbed embedding with similarity ~ 0.99
        v_near = v1.copy()
        v_near[1] = 0.1
        v_near = v_near / np.linalg.norm(v_near)
        cos_sim = float(v1 @ v_near)
        assert cos_sim >= 0.98
        dup_near = find_near_duplicate(v_near, threshold=0.98, db_path=test_db)
        assert dup_near is not None
        assert dup_near["id"] == sid1

        # 3. Test orthogonal embedding (sim = 0.0)
        v_diff = np.zeros(768, dtype=np.float32)
        v_diff[5] = 1.0
        no_dup = find_near_duplicate(v_diff, threshold=0.98, db_path=test_db)
        assert no_dup is None


def test_record_near_duplicate_consolidation_api():
    """Verify POST /api/record consolidates near-duplicates into the existing sample row."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_db_path = tmp_path / "test_dataset.db"
        test_img_dir = tmp_path / "images"
        test_model_path = tmp_path / "test_model.pkl"

        orig_db = db_mod.DEFAULT_DB_PATH
        orig_images = db_mod.IMAGES_DIR
        orig_model = model_mod.MODEL_PATH

        db_mod.DEFAULT_DB_PATH = test_db_path
        db_mod.IMAGES_DIR = test_img_dir
        model_mod.MODEL_PATH = test_model_path

        try:
            init_db(test_db_path)
            client = TestClient(app)

            # Mock extract_vision_embedding to return controlled vectors
            canonical_vec = np.zeros(768, dtype=np.float32)
            canonical_vec[0] = 1.0

            near_dup_vec = canonical_vec.copy()
            near_dup_vec[1] = 0.05
            near_dup_vec = near_dup_vec / np.linalg.norm(near_dup_vec)

            orthogonal_vec = np.zeros(768, dtype=np.float32)
            orthogonal_vec[10] = 1.0

            with patch("backend.app.extract_vision_embedding") as mock_extract:
                # 1. Record original sample
                mock_extract.return_value = canonical_vec
                img_b64 = _make_test_image_b64((100, 100, 100))
                resp1 = client.post("/api/record", json={"image_base64": img_b64, "label": 0, "mode": "manual"})
                assert resp1.status_code == 201
                data1 = resp1.json()
                assert data1["status"] == "success"
                orig_id = data1["id"]

                # 2. Record a near-duplicate (similarity > 0.98) with an updated label
                mock_extract.return_value = near_dup_vec
                img_b64_2 = _make_test_image_b64((105, 100, 100))
                resp2 = client.post("/api/record", json={"image_base64": img_b64_2, "label": 1, "mode": "supervised", "reviewed": 1})
                assert resp2.status_code == 201
                data2 = resp2.json()
                assert data2["status"] == "consolidated"
                assert data2["id"] == orig_id
                assert data2["duplicate_of"] == orig_id
                assert data2["similarity"] >= 0.98

                # Verify database still has only 1 sample, updated to label=1, reviewed=1
                samples_resp = client.get("/api/samples")
                samples = samples_resp.json()
                assert len(samples) == 1
                assert samples[0]["id"] == orig_id
                assert samples[0]["label"] == 1
                assert samples[0]["reviewed"] == 1

                # 3. Record an orthogonal image (sim < 0.98)
                mock_extract.return_value = orthogonal_vec
                img_b64_3 = _make_test_image_b64((0, 255, 0))
                resp3 = client.post("/api/record", json={"image_base64": img_b64_3, "label": 0, "mode": "auto"})
                assert resp3.status_code == 201
                data3 = resp3.json()
                assert data3["status"] == "success"
                assert data3["id"] != orig_id

                # Now database has exactly 2 samples
                samples_after = client.get("/api/samples").json()
                assert len(samples_after) == 2

        finally:
            db_mod.DEFAULT_DB_PATH = orig_db
            db_mod.IMAGES_DIR = orig_images
            model_mod.MODEL_PATH = orig_model


def test_samples_outliers_query_endpoint():
    """Verify GET /api/samples?outliers_only=true filters and tags inconsistent positive ratings."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test.db"
        test_model = Path(tmpdir) / "test_model.pkl"
        init_db(test_db)

        orig_db = db_mod.DEFAULT_DB_PATH
        orig_model = model_mod.MODEL_PATH
        db_mod.DEFAULT_DB_PATH = test_db
        model_mod.MODEL_PATH = test_model

        try:
            # Insert a tight cluster of 5 positive samples
            base_vec = np.zeros(768, dtype=np.float32)
            base_vec[0] = 1.0

            for i in range(5):
                noise = np.zeros(768, dtype=np.float32)
                noise[1] = 0.01 * i
                v = (base_vec + noise) / np.linalg.norm(base_vec + noise)
                insert_sample(f"pos_{i}", f"data/images/pos_{i}.jpg", v, label=1, mode="manual", prediction_score=0.9, db_path=test_db)

            # Insert an outlier positive sample (distant direction)
            outlier_vec = np.zeros(768, dtype=np.float32)
            outlier_vec[2] = 1.0
            outlier_id = insert_sample("pos_outlier", "data/images/pos_outlier.jpg", outlier_vec, label=1, mode="manual", prediction_score=0.9, db_path=test_db)

            # Insert 5 negative samples
            neg_vec = np.zeros(768, dtype=np.float32)
            neg_vec[3] = 1.0
            for i in range(5):
                insert_sample(f"neg_{i}", f"data/images/neg_{i}.jpg", neg_vec, label=0, mode="manual", prediction_score=0.1, db_path=test_db)

            client = TestClient(app)

            # Query all samples: regular samples have is_outlier=False, outlier has is_outlier=True
            all_samples = client.get("/api/samples?limit=50").json()
            outlier_found = [s for s in all_samples if s["id"] == outlier_id]
            assert len(outlier_found) == 1
            assert outlier_found[0]["is_outlier"] is True
            assert outlier_found[0]["outlier_reason"] in ("distance", "both")
            assert outlier_found[0]["centroid_distance"] is not None

            # Query with outliers_only=true
            filtered_resp = client.get("/api/samples?outliers_only=true")
            assert filtered_resp.status_code == 200
            filtered = filtered_resp.json()
            assert len(filtered) == 1
            assert filtered[0]["id"] == outlier_id
            assert filtered[0]["is_outlier"] is True

        finally:
            db_mod.DEFAULT_DB_PATH = orig_db
            model_mod.MODEL_PATH = orig_model


def test_detect_negative_outliers_distance_and_score():
    """Verify negative class outlier detection flags distance > 2 std and high scores."""
    np.random.seed(42)
    # Create a cluster of 15 similar negative vision embeddings
    base_vector = np.zeros(768, dtype=np.float32)
    base_vector[0] = 1.0

    tight_cluster = []
    for _ in range(15):
        noise = np.random.normal(0, 0.02, 768).astype(np.float32)
        vec = base_vector + noise
        vec = vec / np.linalg.norm(vec)
        tight_cluster.append(vec)

    # Add an outlier negative vector pointing in a very different direction
    distant_vector = np.zeros(768, dtype=np.float32)
    distant_vector[1] = 1.0
    distant_vector = distant_vector / np.linalg.norm(distant_vector)
    tight_cluster.append(distant_vector)

    X_neg = np.vstack(tight_cluster)
    y_neg = np.zeros(len(X_neg), dtype=np.int32)
    sample_ids = [200 + i for i in range(len(X_neg))]

    # Distant vector is at index 15 (sample_id 215)
    # Set low prediction scores for cluster except one with a high score (sample_id 205)
    oof_scores = {sid: 0.05 for sid in sample_ids}
    oof_scores[205] = 0.65  # High score outlier (>= 0.35)

    outliers = detect_negative_outliers(
        X_neg,
        y_neg,
        sample_ids=sample_ids,
        oof_score_map=oof_scores,
        score_threshold=0.35,
    )

    assert len(outliers) == 16
    # 1. Distant vector (id 215) flagged due to distance > 2 std
    assert outliers[215]["is_outlier"] is True
    assert outliers[215]["outlier_type"] == "dislike"
    assert outliers[215]["outlier_reason"] in ("distance", "both")

    # 2. High score vector (id 205) flagged due to score >= 0.35
    assert outliers[205]["is_outlier"] is True
    assert outliers[205]["outlier_type"] == "dislike"
    assert outliers[205]["outlier_reason"] in ("high_score", "both")
    assert outliers[205]["prediction_score"] == 0.65

    # 3. Regular tight cluster samples with low scores should NOT be outliers
    normal_sample = outliers[200]
    assert normal_sample["is_outlier"] is False
    assert normal_sample["outlier_reason"] is None


def test_samples_inconsistent_dislikes_query_endpoint():
    """Verify GET /api/samples?quality=inconsistent_dislikes filters to negative class outliers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test.db"
        test_model = Path(tmpdir) / "test_model.pkl"
        init_db(test_db)

        orig_db = db_mod.DEFAULT_DB_PATH
        orig_model = model_mod.MODEL_PATH
        db_mod.DEFAULT_DB_PATH = test_db
        model_mod.MODEL_PATH = test_model

        try:
            # Insert 5 regular negative samples
            base_vec = np.zeros(768, dtype=np.float32)
            base_vec[0] = 1.0

            for i in range(5):
                noise = np.zeros(768, dtype=np.float32)
                noise[1] = 0.01 * i
                v = (base_vec + noise) / np.linalg.norm(base_vec + noise)
                insert_sample(f"neg_{i}", f"data/images/neg_{i}.jpg", v, label=0, mode="manual", prediction_score=0.05, db_path=test_db)

            # Insert an outlier negative sample (distant direction)
            outlier_vec = np.zeros(768, dtype=np.float32)
            outlier_vec[2] = 1.0
            outlier_id = insert_sample("neg_outlier", "data/images/neg_outlier.jpg", outlier_vec, label=0, mode="manual", prediction_score=0.05, db_path=test_db)

            # Insert 5 positive samples
            pos_vec = np.zeros(768, dtype=np.float32)
            pos_vec[3] = 1.0
            for i in range(5):
                insert_sample(f"pos_{i}", f"data/images/pos_{i}.jpg", pos_vec, label=1, mode="manual", prediction_score=0.9, db_path=test_db)

            client = TestClient(app)

            # Query with quality=inconsistent_dislikes
            filtered_resp = client.get("/api/samples?quality=inconsistent_dislikes")
            assert filtered_resp.status_code == 200
            filtered = filtered_resp.json()
            assert len(filtered) == 1
            assert filtered[0]["id"] == outlier_id
            assert filtered[0]["label"] == 0
            assert filtered[0]["is_outlier"] is True
            assert filtered[0]["outlier_type"] == "dislike"
            assert filtered[0]["outlier_reason"] in ("distance", "both")

        finally:
            db_mod.DEFAULT_DB_PATH = orig_db
            model_mod.MODEL_PATH = orig_model
