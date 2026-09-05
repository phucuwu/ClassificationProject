"""Checkpoint 3 test suite: Verifies all backend REST API endpoints using FastAPI TestClient."""

import base64
import io
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import backend.database as db_mod
import backend.model as model_mod
from backend.app import app


def _create_test_image_b64(color=(200, 100, 50)) -> str:
    """Helper to generate a base64 encoded test JPEG image."""
    img = Image.new("RGB", (64, 64), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def test_api_endpoints_end_to_end():
    """Verify all REST API endpoints end-to-end with an isolated temporary test database."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_db_path = tmp_path / "test_dataset.db"
        test_img_dir = tmp_path / "images"
        test_model_path = tmp_path / "test_model.pkl"

        # Patch paths in modules
        orig_db = db_mod.DEFAULT_DB_PATH
        orig_images = db_mod.IMAGES_DIR
        orig_model = model_mod.MODEL_PATH

        db_mod.DEFAULT_DB_PATH = test_db_path
        db_mod.IMAGES_DIR = test_img_dir
        model_mod.MODEL_PATH = test_model_path

        try:
            db_mod.init_db(test_db_path)
            client = TestClient(app)

            # 1. Test POST /api/predict in cold start state
            img_b64_1 = _create_test_image_b64(color=(255, 0, 0))
            pred_resp = client.post("/api/predict", json={"image_base64": img_b64_1})
            assert pred_resp.status_code == 200
            pred_data = pred_resp.json()
            assert pred_data["model_loaded"] is False
            assert pred_data["prediction_score"] is None

            # 2. Test POST /api/record (Like sample)
            rec_resp1 = client.post(
                "/api/record",
                json={
                    "image_base64": img_b64_1,
                    "label": 1,
                    "mode": "manual",
                    "reviewed": 1,
                },
            )
            assert rec_resp1.status_code == 201
            data1 = rec_resp1.json()
            assert data1["status"] == "success"
            assert data1["label"] == 1
            id1 = data1["id"]

            # 3. Test POST /api/record (Dislike sample)
            img_b64_2 = _create_test_image_b64(color=(0, 255, 0))
            rec_resp2 = client.post(
                "/api/record",
                json={
                    "image_base64": img_b64_2,
                    "label": 0,
                    "mode": "manual",
                    "reviewed": 1,
                },
            )
            assert rec_resp2.status_code == 201
            data2 = rec_resp2.json()
            id2 = data2["id"]

            # 4. Test POST /api/record (Auto sample pending review)
            img_b64_3 = _create_test_image_b64(color=(0, 0, 255))
            rec_resp3 = client.post(
                "/api/record",
                json={
                    "image_base64": img_b64_3,
                    "label": 1,
                    "mode": "auto",
                    "prediction_score": 0.85,
                    "reviewed": 0,
                },
            )
            assert rec_resp3.status_code == 201
            data3 = rec_resp3.json()
            id3 = data3["id"]

            # 5. Test GET /api/samples
            samples_resp = client.get("/api/samples")
            assert samples_resp.status_code == 200
            samples_list = samples_resp.json()
            assert len(samples_list) == 3
            assert samples_list[0]["image_base64"].startswith("data:image/jpeg;base64,")

            # Filter for unreviewed auto samples
            unreviewed_resp = client.get("/api/samples", params={"reviewed": 0, "mode": "auto"})
            assert unreviewed_resp.status_code == 200
            unreviewed_list = unreviewed_resp.json()
            assert len(unreviewed_list) == 1
            assert unreviewed_list[0]["id"] == id3

            # 6. Test POST /api/review (Bulk update)
            rev_resp = client.post(
                "/api/review",
                json={
                    "updates": [
                        {"id": id3, "label": 0, "reviewed": 1}
                    ]
                },
            )
            assert rev_resp.status_code == 200
            assert rev_resp.json()["updated_count"] == 1

            # 7. Test POST /api/train
            train_resp = client.post("/api/train", json={"target_recall": 0.90})
            assert train_resp.status_code == 200
            train_data = train_resp.json()
            assert train_data["status"] == "trained"
            assert train_data["sample_count"] == 3
            assert "metrics" in train_data

            # 8. Test POST /api/predict after model training
            pred_resp2 = client.post("/api/predict", json={"image_base64": img_b64_1})
            assert pred_resp2.status_code == 200
            pred_data2 = pred_resp2.json()
            assert pred_data2["model_loaded"] is True
            assert isinstance(pred_data2["prediction_score"], float)
            assert pred_data2["decision"] in (0, 1)

            # 9. Test POST /api/threshold (Update active decision threshold)
            thresh_resp = client.post("/api/threshold", json={"threshold": 0.42})
            assert thresh_resp.status_code == 200
            thresh_data = thresh_resp.json()
            assert thresh_data["success"] is True
            assert thresh_data["decision_threshold"] == 0.42

            # 10. Test POST /api/train with custom threshold override
            train_custom_resp = client.post("/api/train", json={"threshold": 0.28})
            assert train_custom_resp.status_code == 200
            assert train_custom_resp.json()["metrics"]["decision_threshold"] == 0.28

            # 11. Test GET /api/metrics
            metrics_resp = client.get("/api/metrics")
            assert metrics_resp.status_code == 200
            metrics_data = metrics_resp.json()
            assert "statistics" in metrics_data
            assert "model_status" in metrics_data
            assert metrics_data["statistics"]["total_samples"] == 3
            assert metrics_data["model_status"]["model_loaded"] is True
            assert metrics_data["model_status"]["decision_threshold"] == 0.28

            # 12. Test DELETE /api/samples/{id}
            del_resp1 = client.delete(f"/api/samples/{id1}")
            assert del_resp1.status_code == 200
            assert del_resp1.json()["status"] == "success"

            # 13. Test POST /api/samples/batch-delete
            del_resp2 = client.post("/api/samples/batch-delete", json={"ids": [id2, id3]})
            assert del_resp2.status_code == 200
            assert del_resp2.json()["deleted_count"] == 2

            # Verify samples empty
            metrics_after = client.get("/api/metrics").json()
            assert metrics_after["statistics"]["total_samples"] == 0

        finally:
            db_mod.DEFAULT_DB_PATH = orig_db
            db_mod.IMAGES_DIR = orig_images
            model_mod.MODEL_PATH = orig_model


# ---------------------------------------------------------------------------
# Phase 2: provenance-aware recording/review, exact-hash dedup, capture removal
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent


@contextmanager
def _isolated_backend(tmpdir: str | Path):
    """Patch backend paths to an isolated temp dir and yield a TestClient."""
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
        db_mod.init_db(test_db_path)
        yield TestClient(app)
    finally:
        db_mod.DEFAULT_DB_PATH = orig_db
        db_mod.IMAGES_DIR = orig_images
        model_mod.MODEL_PATH = orig_model


def _samples_by_id(client: TestClient) -> dict[int, dict]:
    resp = client.get("/api/samples", params={"limit": 200})
    assert resp.status_code == 200
    return {s["id"]: s for s in resp.json()}


def test_phase2_record_rejects_invalid_payloads():
    """Constrained /api/record and /api/review types reject bad labels, modes, scores, and review flags."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with _isolated_backend(tmpdir) as client:
            img = _create_test_image_b64()
            bad_record_payloads = [
                {"image_base64": img, "label": 2, "mode": "manual"},
                {"image_base64": img, "label": "like", "mode": "manual"},
                {"image_base64": img, "label": 1, "mode": "turbo"},
                {"image_base64": img, "label": 1, "mode": "manual", "prediction_score": 1.5},
                {"image_base64": img, "label": 1, "mode": "manual", "prediction_score": -0.1},
                {"image_base64": img, "label": 1, "mode": "manual", "reviewed": 2},
                {"label": 1, "mode": "manual"},
                {"image_base64": "", "label": 1, "mode": "manual"},
            ]
            for payload in bad_record_payloads:
                resp = client.post("/api/record", json=payload)
                assert resp.status_code in (400, 422), payload

            bad_review = client.post(
                "/api/review",
                json={"updates": [{"id": 1, "label": 5, "reviewed": 1}]},
            )
            assert bad_review.status_code in (400, 422)


def test_phase2_record_forces_provenance_and_review_state():
    """The server derives provenance/review state from mode, ignoring caller hints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with _isolated_backend(tmpdir) as client:
            manual = client.post(
                "/api/record",
                json={
                    "image_base64": _create_test_image_b64(color=(200, 10, 10)),
                    "label": 1,
                    "mode": "manual",
                    "reviewed": 0,
                },
            )
            assert manual.status_code == 201
            assert manual.json()["reviewed"] == 1
            assert manual.json()["label_provenance"] == "manual_rating"

            supervised = client.post(
                "/api/record",
                json={
                    "image_base64": _create_test_image_b64(color=(10, 200, 10)),
                    "label": 0,
                    "mode": "supervised",
                    "prediction_score": 0.2,
                    "reviewed": 0,
                },
            )
            assert supervised.status_code == 201
            assert supervised.json()["reviewed"] == 1
            assert supervised.json()["label_provenance"] == "supervised_confirmation"

            auto = client.post(
                "/api/record",
                json={
                    "image_base64": _create_test_image_b64(color=(10, 10, 200)),
                    "label": 1,
                    "mode": "auto",
                    "prediction_score": 0.9,
                    "reviewed": 1,
                },
            )
            assert auto.status_code == 201
            assert auto.json()["reviewed"] == 0
            assert auto.json()["label_provenance"] == "auto_decision"

            rows = _samples_by_id(client)
            assert rows[manual.json()["id"]]["label_provenance"] == "manual_rating"
            assert rows[manual.json()["id"]]["reviewed"] == 1
            assert rows[supervised.json()["id"]]["label_provenance"] == "supervised_confirmation"
            assert rows[supervised.json()["id"]]["reviewed"] == 1
            assert rows[auto.json()["id"]]["label_provenance"] == "auto_decision"
            assert rows[auto.json()["id"]]["reviewed"] == 0


def test_phase2_auto_excluded_until_review_confirmation():
    """Full auto Samples stay out of training until review flips them to review_confirmation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with _isolated_backend(tmpdir) as client:
            rec = client.post(
                "/api/record",
                json={
                    "image_base64": _create_test_image_b64(color=(90, 90, 10)),
                    "label": 1,
                    "mode": "auto",
                    "prediction_score": 0.88,
                    "reviewed": 0,
                },
            )
            assert rec.status_code == 201
            auto_id = rec.json()["id"]

            _, _, ids_before = db_mod.load_training_matrix(
                return_ids=True, db_path=db_mod.DEFAULT_DB_PATH
            )
            assert auto_id not in ids_before

            rev = client.post(
                "/api/review",
                json={"updates": [{"id": auto_id, "label": 1, "reviewed": 1}]},
            )
            assert rev.status_code == 200
            assert rev.json()["updated_count"] == 1

            rows = _samples_by_id(client)
            assert rows[auto_id]["reviewed"] == 1
            assert rows[auto_id]["label_provenance"] == "review_confirmation"

            _, _, ids_after = db_mod.load_training_matrix(
                return_ids=True, db_path=db_mod.DEFAULT_DB_PATH
            )
            assert auto_id in ids_after


def test_phase2_exact_hash_dedup_preserves_confirmed_label():
    """Re-recording the exact Primary image creates no second row and keeps the confirmed label."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with _isolated_backend(tmpdir) as client:
            img = _create_test_image_b64(color=(10, 20, 30))
            first = client.post(
                "/api/record",
                json={"image_base64": img, "label": 1, "mode": "manual", "reviewed": 1},
            )
            assert first.status_code == 201
            first_id = first.json()["id"]

            second = client.post(
                "/api/record",
                json={
                    "image_base64": img,
                    "label": 0,
                    "mode": "auto",
                    "prediction_score": 0.05,
                    "reviewed": 0,
                },
            )
            assert second.status_code == 201
            assert second.json()["status"] == "duplicate"
            assert second.json()["id"] == first_id
            assert second.json()["duplicate_of"] == first_id

            rows = _samples_by_id(client)
            assert len(rows) == 1
            assert rows[first_id]["label"] == 1
            assert rows[first_id]["reviewed"] == 1
            assert rows[first_id]["label_provenance"] == "manual_rating"


def test_phase2_near_match_creates_separate_sample_without_mutation():
    """Similarity >= 0.98 on a different hash creates a separate Sample and mutates nothing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with _isolated_backend(tmpdir) as client:
            canonical = np.zeros(768, dtype=np.float32)
            canonical[0] = 1.0
            near = canonical.copy()
            near[1] = 0.05
            near = near / np.linalg.norm(near)

            with patch("backend.app.extract_vision_embedding") as mock_extract:
                mock_extract.side_effect = [canonical, near]
                first = client.post(
                    "/api/record",
                    json={
                        "image_base64": _create_test_image_b64(color=(60, 60, 60)),
                        "label": 1,
                        "mode": "manual",
                        "reviewed": 1,
                    },
                )
                assert first.status_code == 201
                assert first.json()["status"] == "success"
                first_id = first.json()["id"]

                second = client.post(
                    "/api/record",
                    json={
                        "image_base64": _create_test_image_b64(color=(200, 30, 30)),
                        "label": 0,
                        "mode": "supervised",
                        "prediction_score": 0.1,
                        "reviewed": 1,
                    },
                )
                assert second.status_code == 201
                assert second.json()["status"] == "success"
                second_id = second.json()["id"]
                assert second_id != first_id

            rows = _samples_by_id(client)
            assert len(rows) == 2
            assert rows[first_id]["label"] == 1
            assert rows[first_id]["mode"] == "manual"
            assert rows[first_id]["reviewed"] == 1
            assert rows[first_id]["label_provenance"] == "manual_rating"
            assert rows[second_id]["label"] == 0
            assert rows[second_id]["reviewed"] == 1
            assert rows[second_id]["label_provenance"] == "supervised_confirmation"


def test_phase2_capture_endpoint_removed_and_cors_restricted():
    """No /api/capture route exists and CORS allows only local dashboard origins."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with _isolated_backend(tmpdir) as client:
            capture_post = client.post(
                "/api/capture", json={"x": 0, "y": 0, "width": 10, "height": 10}
            )
            # No API route serves desktop capture anymore (unknown paths fall
            # through to the static mount, which answers POST with 405).
            assert capture_post.status_code in (404, 405)
            assert client.get("/api/capture").status_code in (404, 405)

            route_paths = [
                getattr(route, "path", "") for route in app.routes
            ]
            assert "/api/capture" not in route_paths

            evil = client.get("/api/metrics", headers={"Origin": "https://evil.example.com"})
            assert evil.status_code == 200
            assert "access-control-allow-origin" not in {k.lower() for k in evil.headers}

            local = client.get("/api/metrics", headers={"Origin": "http://localhost:8000"})
            assert local.status_code == 200
            assert local.headers.get("access-control-allow-origin") == "http://localhost:8000"


def test_phase2_no_mss_dependency():
    """mss usage and the desktop-capture endpoint are fully removed."""
    requirements = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert not any(
        line.strip().lower().startswith("mss") for line in requirements.splitlines()
    )
    backend_source = (REPO_ROOT / "backend" / "app.py").read_text(encoding="utf-8")
    assert "import mss" not in backend_source
    assert "/api/capture" not in backend_source


def test_phase2_userscript_fail_closed_contract():
    """Userscript narrows extraction to the active card and fails closed without capture."""
    userscript = (REPO_ROOT / "userscript" / "taste_collector.user.js").read_text(
        encoding="utf-8", errors="replace"
    )
    assert "@connect      *" not in userscript
    assert "localhost" in userscript
    assert "127.0.0.1" in userscript
    assert "/api/capture" not in userscript
    assert "requestScreenCapture" not in userscript
    assert 'querySelector("canvas")' not in userscript
    assert "main, body" not in userscript
    assert "|| document.body" not in userscript
    for marker in (
        "validatePrimaryImage",
        "revalidateCapturedArtwork",
        "logExtractionFailure",
        "isConnected",
        "isAssociatedWithActiveCard",
    ):
        assert marker in userscript
    assert userscript.count("revalidateCapturedArtwork") >= 6


if __name__ == "__main__":
    test_api_endpoints_end_to_end()
    print("ALL CHECKPOINT 3 TESTS PASSED SUCCESSFULLY!")
