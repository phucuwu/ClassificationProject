"""Checkpoint 4 test suite: Verifies static asset serving for the Developer Dashboard."""

from fastapi.testclient import TestClient
from backend.app import app


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


if __name__ == "__main__":
    test_dashboard_static_assets()
    print("ALL CHECKPOINT 4 TESTS PASSED SUCCESSFULLY!")
