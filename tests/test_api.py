"""Checkpoint 3 test suite: Verifies all backend REST API endpoints using FastAPI TestClient."""

import base64
import io
import tempfile
from pathlib import Path

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


if __name__ == "__main__":
    test_api_endpoints_end_to_end()
    print("ALL CHECKPOINT 3 TESTS PASSED SUCCESSFULLY!")
