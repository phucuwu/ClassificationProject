"""Test suite for vision embedding scatter plot projection and endpoint."""

import tempfile
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.database import init_db, insert_sample, load_embedding_scatter_data


def test_load_embedding_scatter_data_empty():
    """Verify loading scatter data from an empty database returns empty results."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_dataset.db"
        init_db(test_db)

        meta, X = load_embedding_scatter_data(test_db)
        assert len(meta) == 0
        assert X.shape == (0, 768)


def test_load_embedding_scatter_data_with_samples():
    """Verify sample metadata and feature embeddings load correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_dataset.db"
        init_db(test_db)

        vec1 = np.random.randn(768).astype(np.float32)
        vec2 = np.random.randn(768).astype(np.float32)

        insert_sample(
            image_hash="hash_1",
            file_path="data/images/hash_1.jpg",
            embedding=vec1,
            label=1,
            mode="manual",
            prediction_score=0.92,
            reviewed=1,
            db_path=test_db,
        )
        insert_sample(
            image_hash="hash_2",
            file_path="data/images/hash_2.jpg",
            embedding=vec2,
            label=0,
            mode="auto",
            prediction_score=0.15,
            reviewed=0,
            db_path=test_db,
        )

        meta, X = load_embedding_scatter_data(test_db)
        assert len(meta) == 2
        assert X.shape == (2, 768)
        assert meta[0]["label"] == 1
        assert meta[0]["prediction_score"] == 0.92
        assert meta[1]["label"] == 0
        assert meta[1]["mode"] == "auto"


def test_scatter_endpoint_pca_and_tsne(monkeypatch):
    """Verify /api/embeddings/scatter returns 2D projected coordinates with PCA and t-SNE."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_dataset.db"
        init_db(test_db)

        import backend.database as db_mod
        monkeypatch.setattr(db_mod, "DEFAULT_DB_PATH", test_db)

        # 1. Empty dataset test
        client = TestClient(app)
        res_empty = client.get("/api/embeddings/scatter?method=pca")
        assert res_empty.status_code == 200
        data_empty = res_empty.json()
        assert data_empty["status"] == "empty"
        assert data_empty["total_points"] == 0
        assert data_empty["points"] == []

        # 2. Populate dataset with 5 synthetic samples
        for i in range(5):
            vec = np.random.randn(768).astype(np.float32)
            insert_sample(
                image_hash=f"hash_{i}",
                file_path=f"data/images/hash_{i}.jpg",
                embedding=vec,
                label=1 if i % 2 == 0 else 0,
                mode="manual",
                prediction_score=0.8 if i % 2 == 0 else 0.2,
                reviewed=1,
                db_path=test_db,
            )

        # 3. Test PCA projection
        res_pca = client.get("/api/embeddings/scatter?method=pca")
        assert res_pca.status_code == 200
        data_pca = res_pca.json()
        assert data_pca["status"] == "success"
        assert data_pca["total_points"] == 5
        assert len(data_pca["points"]) == 5
        assert "variance_ratio" in data_pca
        assert len(data_pca["variance_ratio"]) == 2

        point = data_pca["points"][0]
        assert "x" in point and "y" in point
        assert "image_url" in point
        assert point["image_url"] == f"/images/{point['image_hash']}.jpg"
        assert point["label"] in (0, 1)

        # 4. Test t-SNE projection
        res_tsne = client.get("/api/embeddings/scatter?method=tsne")
        assert res_tsne.status_code == 200
        data_tsne = res_tsne.json()
        assert data_tsne["status"] == "success"
        assert data_tsne["total_points"] == 5
        assert len(data_tsne["points"]) == 5


if __name__ == "__main__":
    test_load_embedding_scatter_data_empty()
    test_load_embedding_scatter_data_with_samples()
    import unittest.mock
    class MonkeyPatch:
        def setattr(self, obj, attr, val):
            setattr(obj, attr, val)
    test_scatter_endpoint_pca_and_tsne(MonkeyPatch())
    print("ALL SCATTER PLOT TESTS PASSED SUCCESSFULLY!")

