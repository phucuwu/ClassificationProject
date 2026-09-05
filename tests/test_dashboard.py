"""Checkpoint 4 test suite: Verifies static asset serving for the Developer Dashboard."""

import base64
import io
import tempfile
from pathlib import Path

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


def test_dashboard_static_assets():
    """Verify that Developer Dashboard HTML, CSS, and JS are properly served at root."""
    client = TestClient(app)

    # 1. Test GET / (index.html)
    resp_index = client.get("/")
    assert resp_index.status_code == 200
    assert "Art Taste Classifier" in resp_index.text
    assert "Review Queue" in resp_index.text

    # 2. Test GET /style.css
    resp_css = client.get("/style.css")
    assert resp_css.status_code == 200
    assert "dashboard-layout" in resp_css.text

    # 3. Test GET /app.js
    resp_js = client.get("/app.js")
    assert resp_js.status_code == 200
    assert "handleGlobalKeydown" in resp_js.text


def test_dashboard_trust_ui_elements():
    """Verify the dashboard serves trust UI: warning/stale banners, eligible counts, provenance."""
    client = TestClient(app)

    resp_index = client.get("/")
    assert resp_index.status_code == 200
    for element_id in (
        "effectiveness-warning-banner",
        "stale-model-banner",
        "model-trust-meta",
        "stat-eligible-samples",
        "holdout-summary-container",
    ):
        assert element_id in resp_index.text, f"index.html missing #{element_id}"

    resp_js = client.get("/app.js")
    assert resp_js.status_code == 200
    for hook in (
        "renderTrustState",
        "temporal_holdout",
        "provenanceBadge",
        "label_provenance",
        "warning_reasons",
        "Temporal Effectiveness",
    ):
        assert hook in resp_js.text, f"app.js missing trust hook {hook!r}"

    resp_css = client.get("/style.css")
    assert resp_css.status_code == 200
    for cls in ("trust-banner", "provenance-badge", "trust-meta-card"):
        assert cls in resp_css.text, f"style.css missing .{cls}"


def _seed_training_eligible_samples(client: TestClient) -> None:
    """Seed 2 manual Dislikes + 1 manual Like (training-eligible) + 1 auto decision."""
    manual_dislikes = [_create_test_image_b64(color=(10, 20, 30)), _create_test_image_b64(color=(11, 21, 31))]
    for img_b64 in manual_dislikes:
        resp = client.post("/api/record", json={"image_base64": img_b64, "label": 0, "mode": "manual", "reviewed": 1})
        assert resp.status_code == 201
    resp = client.post(
        "/api/record",
        json={"image_base64": _create_test_image_b64(color=(200, 30, 30)), "label": 1, "mode": "manual", "reviewed": 1},
    )
    assert resp.status_code == 201
    resp = client.post(
        "/api/record",
        json={
            "image_base64": _create_test_image_b64(color=(30, 30, 200)),
            "label": 0,
            "mode": "auto",
            "prediction_score": 0.2,
            "reviewed": 0,
        },
    )
    assert resp.status_code == 201


def test_metrics_exposes_effectiveness_and_stale_status():
    """Verify /api/metrics reports eligible counts, effectiveness, and stale-model state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_db_path = tmp_path / "test_dashboard.db"
        test_img_dir = tmp_path / "images"
        test_model_path = tmp_path / "test_model.json"

        orig_db = db_mod.DEFAULT_DB_PATH
        orig_images = db_mod.IMAGES_DIR
        orig_model = model_mod.MODEL_PATH
        db_mod.DEFAULT_DB_PATH = test_db_path
        db_mod.IMAGES_DIR = test_img_dir
        model_mod.MODEL_PATH = test_model_path

        try:
            db_mod.init_db(test_db_path)
            client = TestClient(app)
            _seed_training_eligible_samples(client)

            train_resp = client.post("/api/train", json={"holdout_ratio": 0.0})
            assert train_resp.status_code == 200
            assert train_resp.json()["status"] == "trained"

            metrics = client.get("/api/metrics").json()
            stats = metrics["statistics"]
            model_status = metrics["model_status"]

            # Training-eligible counts and provenance are exposed
            assert stats["training_eligible_count"] == 3
            assert stats["training_eligible_positive_count"] == 1
            assert stats["training_eligible_negative_count"] == 2
            assert metrics["training_eligible"]["sample_count"] == 3
            assert metrics["provenance_counts"]["manual_rating"] == 3
            assert metrics["provenance_counts"]["auto_decision"] == 1

            # Effectiveness section: temporal unavailable on tiny data, warning on
            assert model_status["training_eligible"]["sample_count"] == 3
            assert model_status["effectiveness"]["status"] == "temporal_evaluation_unavailable"
            assert model_status["warning_active"] is True
            assert "temporal_evaluation_unavailable" in model_status["warning_reasons"]
            assert model_status["threshold_source"] in ("calibrated", "explicit", "target_recall")

            # Fresh model matches the eligible collection: not stale
            assert model_status["stale_model"] is False
            assert model_status["stale_reason"] is None

            # Adding another training-eligible Sample makes the saved model stale
            resp = client.post(
                "/api/record",
                json={
                    "image_base64": _create_test_image_b64(color=(12, 22, 32)),
                    "label": 0,
                    "mode": "manual",
                    "reviewed": 1,
                },
            )
            assert resp.status_code == 201
            stale_metrics = client.get("/api/metrics").json()
            assert stale_metrics["statistics"]["training_eligible_count"] == 4
            assert stale_metrics["model_status"]["stale_model"] is True
            assert stale_metrics["model_status"]["stale_reason"] is not None
        finally:
            db_mod.DEFAULT_DB_PATH = orig_db
            db_mod.IMAGES_DIR = orig_images
            model_mod.MODEL_PATH = orig_model


def test_review_queue_and_inspector_show_provenance():
    """Verify the review queue and inspector expose label provenance per Sample."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        test_db_path = tmp_path / "test_prov.db"
        test_img_dir = tmp_path / "images"
        test_model_path = tmp_path / "test_model.json"

        orig_db = db_mod.DEFAULT_DB_PATH
        orig_images = db_mod.IMAGES_DIR
        orig_model = model_mod.MODEL_PATH
        db_mod.DEFAULT_DB_PATH = test_db_path
        db_mod.IMAGES_DIR = test_img_dir
        model_mod.MODEL_PATH = test_model_path

        try:
            db_mod.init_db(test_db_path)
            client = TestClient(app)
            _seed_training_eligible_samples(client)

            queue = client.get("/api/samples", params={"mode": "auto", "reviewed": 0}).json()
            assert len(queue) == 1
            assert queue[0]["label_provenance"] == "auto_decision"

            inspector = client.get("/api/samples", params={"limit": 50}).json()
            assert len(inspector) == 4
            by_prov = {s["label_provenance"] for s in inspector}
            assert "manual_rating" in by_prov
            assert "auto_decision" in by_prov
            for sample in inspector:
                assert sample["label_provenance"] in (
                    "manual_rating",
                    "supervised_confirmation",
                    "review_confirmation",
                    "auto_decision",
                )
        finally:
            db_mod.DEFAULT_DB_PATH = orig_db
            db_mod.IMAGES_DIR = orig_images
            model_mod.MODEL_PATH = orig_model


def test_userscript_full_auto_acknowledgement_hook():
    """Verify the userscript gates every Full auto activation on an explicit acknowledgement."""
    userscript_path = Path(__file__).resolve().parent.parent / "userscript" / "taste_collector.user.js"
    source = userscript_path.read_text(encoding="utf-8")

    # Acknowledgement machinery exists and consults backend effectiveness state
    for hook in (
        "fetchEffectivenessState",
        "requestFullAutoAcknowledgement",
        "tryEnableFullAuto",
        "fullAutoAcknowledged",
        "/api/metrics",
        "warningActive",
    ):
        assert hook in source, f"userscript missing acknowledgement hook {hook!r}"

    # Explicit user decision via a confirmation dialog before enabling Full auto
    assert "window.confirm" in source
    assert "acknowledge" in source.lower()

    # Activation path goes through the gate (not straight into the auto loop)
    activation_block = source[source.index('"a"'):]
    assert "tryEnableFullAuto()" in activation_block


if __name__ == "__main__":
    test_dashboard_static_assets()
    test_dashboard_trust_ui_elements()
    test_metrics_exposes_effectiveness_and_stale_status()
    test_review_queue_and_inspector_show_provenance()
    test_userscript_full_auto_acknowledgement_hook()
    print("ALL CHECKPOINT 4 TESTS PASSED SUCCESSFULLY!")
