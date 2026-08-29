"""Checkpoint 5 test suite: Simulates full end-to-end lifecycle across manual, supervised, and full auto modes."""

import base64
import io
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

import backend.database as db_mod
import backend.model as model_mod
from backend.app import app


def _generate_test_image(color=(150, 100, 200)) -> str:
    """Helper to generate a base64 encoded JPEG."""
    img = Image.new("RGB", (64, 64), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def test_complete_system_workflow():
    """Verify entire workflow: manual data gathering -> train -> supervised -> full auto -> review queue."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        orig_db = db_mod.DEFAULT_DB_PATH
        orig_images = db_mod.IMAGES_DIR
        orig_model = model_mod.MODEL_PATH

        db_mod.DEFAULT_DB_PATH = tmp_path / "test_dataset.db"
        db_mod.IMAGES_DIR = tmp_path / "images"
        model_mod.MODEL_PATH = tmp_path / "test_model.pkl"

        try:
            db_mod.init_db(db_mod.DEFAULT_DB_PATH)
            client = TestClient(app)

            # --- Step 1: Manual Mode (Data Gathering) ---
            # Ingest 10 dislikes and 2 likes
            for i in range(10):
                img_b64 = _generate_test_image(color=(50 + i * 10, 50, 50))
                resp = client.post("/api/record", json={
                    "image_base64": img_b64,
                    "label": 0,
                    "mode": "manual",
                    "reviewed": 1,
                })
                assert resp.status_code == 201

            for i in range(2):
                img_b64 = _generate_test_image(color=(200, 180 + i * 20, 220))
                resp = client.post("/api/record", json={
                    "image_base64": img_b64,
                    "label": 1,
                    "mode": "manual",
                    "reviewed": 1,
                })
                assert resp.status_code == 201

            stats_initial = client.get("/api/metrics").json()["statistics"]
            assert stats_initial["total_samples"] == 12
            assert stats_initial["positive_count"] == 2
            assert stats_initial["negative_count"] == 10

            # --- Step 2: Train Model ---
            train_resp = client.post("/api/train", json={"target_recall": 0.90})
            assert train_resp.status_code == 200
            train_data = train_resp.json()
            assert train_data["status"] == "trained"
            assert "metrics" in train_data
            assert "decision_threshold" in train_data["metrics"]

            # --- Step 3: Supervised Mode (Prediction + Confirmation) ---
            super_img = _generate_test_image(color=(210, 190, 230))
            pred_resp = client.post("/api/predict", json={"image_base64": super_img})
            assert pred_resp.status_code == 200
            pred_data = pred_resp.json()
            assert pred_data["model_loaded"] is True
            assert isinstance(pred_data["prediction_score"], float)

            # User accepts/records the supervised prediction
            rec_super = client.post("/api/record", json={
                "image_base64": super_img,
                "label": pred_data["decision"],
                "mode": "supervised",
                "prediction_score": pred_data["prediction_score"],
                "reviewed": 1,
            })
            assert rec_super.status_code == 201

            # --- Step 4: Full Auto Mode (Prediction + Log to Review Queue) ---
            auto_img = _generate_test_image(color=(40, 60, 80))
            auto_pred = client.post("/api/predict", json={"image_base64": auto_img}).json()

            rec_auto = client.post("/api/record", json={
                "image_base64": auto_img,
                "label": auto_pred["decision"],
                "mode": "auto",
                "prediction_score": auto_pred["prediction_score"],
                "reviewed": 0,
            })
            assert rec_auto.status_code == 201
            auto_sample_id = rec_auto.json()["id"]

            # Check Review Queue has 1 pending auto sample
            queue = client.get("/api/samples", params={"mode": "auto", "reviewed": 0}).json()
            assert len(queue) == 1
            assert queue[0]["id"] == auto_sample_id

            # --- Step 5: Dashboard Review Action ---
            rev_resp = client.post("/api/review", json={
                "updates": [
                    {"id": auto_sample_id, "label": 0, "reviewed": 1}
                ]
            })
            assert rev_resp.status_code == 200
            assert rev_resp.json()["updated_count"] == 1

            # Verify queue is now empty
            queue_after = client.get("/api/samples", params={"mode": "auto", "reviewed": 0}).json()
            assert len(queue_after) == 0

            # --- Step 6: Final Verification ---
            final_metrics = client.get("/api/metrics").json()
            assert final_metrics["statistics"]["total_samples"] == 14
            assert final_metrics["statistics"]["pending_auto_review_count"] == 0
            assert final_metrics["model_status"]["model_loaded"] is True

        finally:
            db_mod.DEFAULT_DB_PATH = orig_db
            db_mod.IMAGES_DIR = orig_images
            model_mod.MODEL_PATH = orig_model


if __name__ == "__main__":
    test_complete_system_workflow()
    print("ALL CHECKPOINT 5 END-TO-END TESTS PASSED SUCCESSFULLY!")
