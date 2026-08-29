"""Checkpoint 1 test suite: Verifies SQLite schema, WAL mode, CRUD, and feature matrix export."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from backend.database import (
    get_dataset_statistics,
    get_db_connection,
    get_samples,
    init_db,
    insert_sample,
    load_training_matrix,
    update_sample_reviews,
)


def test_database_initialization_and_wal():
    """Verify that dataset database initializes tables and operates in WAL mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_dataset.db"
        init_db(test_db)

        conn = get_db_connection(test_db)
        cursor = conn.cursor()

        # Check journal mode
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"

        # Check table existence
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='samples';")
        assert cursor.fetchone() is not None, "Table 'samples' was not created"
        conn.close()


def test_sample_crud_and_matrix_export():
    """Verify sample insertion, query filtering, bulk updates, and training matrix extraction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_dataset.db"
        init_db(test_db)

        # 1. Insert synthetic samples
        dim = 768
        vec1 = np.random.randn(dim).astype(np.float32)
        vec2 = np.random.randn(dim).astype(np.float32)
        vec3 = np.random.randn(dim).astype(np.float32)

        id1 = insert_sample(
            image_hash="hash_like_1",
            file_path="data/images/hash_like_1.jpg",
            embedding=vec1,
            label=1,
            mode="manual",
            reviewed=1,
            db_path=test_db,
        )
        assert id1 > 0

        id2 = insert_sample(
            image_hash="hash_dislike_1",
            file_path="data/images/hash_dislike_1.jpg",
            embedding=vec2,
            label=0,
            mode="manual",
            reviewed=1,
            db_path=test_db,
        )
        assert id2 > 0

        id3 = insert_sample(
            image_hash="hash_auto_pending",
            file_path="data/images/hash_auto_pending.jpg",
            embedding=vec3,
            label=1,
            mode="auto",
            prediction_score=0.75,
            reviewed=0,
            db_path=test_db,
        )
        assert id3 > 0

        # 2. Test get_samples query filtering
        all_samples = get_samples(db_path=test_db)
        assert len(all_samples) == 3

        unreviewed = get_samples(reviewed=0, db_path=test_db)
        assert len(unreviewed) == 1
        assert unreviewed[0]["image_hash"] == "hash_auto_pending"

        # 3. Test bulk review updates
        updated_count = update_sample_reviews(
            updates=[{"id": id3, "label": 0, "reviewed": 1}],
            db_path=test_db,
        )
        assert updated_count == 1

        rechecked = get_samples(reviewed=0, db_path=test_db)
        assert len(rechecked) == 0

        # 4. Test feature matrix (X) and label vector (y) loading
        X, y = load_training_matrix(db_path=test_db)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape == (3, 768)
        assert y.shape == (3,)
        assert X.dtype == np.float32
        assert y.dtype == np.int32
        np.testing.assert_allclose(X[0], vec1, atol=1e-6)

        # 5. Test dataset statistics
        stats = get_dataset_statistics(db_path=test_db)
        assert stats["total_samples"] == 3
        assert stats["positive_count"] == 1  # id1 was 1, id2 was 0, id3 was updated to 0
        assert stats["negative_count"] == 2
        assert stats["positive_ratio"] == round(1 / 3, 4)


if __name__ == "__main__":
    test_database_initialization_and_wal()
    test_sample_crud_and_matrix_export()
    print("ALL CHECKPOINT 1 TESTS PASSED SUCCESSFULLY!")
