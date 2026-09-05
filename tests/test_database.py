"""Checkpoint 1 test suite: Verifies SQLite schema, WAL mode, CRUD, and feature matrix export.

Phase 1 contract: label provenance, training-eligible Feature matrix, idempotent
migration with Sample invariants, and provenance-aware dataset statistics.
"""

import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.database import (
    get_dataset_statistics,
    get_db_connection,
    get_samples,
    init_db,
    insert_sample,
    load_training_matrix,
    migrate_dataset_database,
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

        # Check label_provenance column exists
        cursor.execute("PRAGMA table_info(samples);")
        columns = {row[1] for row in cursor.fetchall()}
        assert "label_provenance" in columns, "Column 'label_provenance' is missing"

        # Check CHECK constraints are enforced on fresh schema
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='samples';")
        create_sql = cursor.fetchone()[0]
        assert "label_provenance" in create_sql
        assert "length(embedding)" in create_sql
        assert "manual_rating" in create_sql
        assert "supervised_confirmation" in create_sql
        assert "review_confirmation" in create_sql
        assert "auto_decision" in create_sql
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

        # Provenance is derived on insert.
        by_hash = {s["image_hash"]: s for s in get_samples(db_path=test_db, limit=50)}
        assert by_hash["hash_like_1"]["label_provenance"] == "manual_rating"
        assert by_hash["hash_dislike_1"]["label_provenance"] == "manual_rating"
        assert by_hash["hash_auto_pending"]["label_provenance"] == "auto_decision"

        # 2. Test get_samples query filtering
        all_samples = get_samples(db_path=test_db)
        assert len(all_samples) == 3

        unreviewed = get_samples(reviewed=0, db_path=test_db)
        assert len(unreviewed) == 1
        assert unreviewed[0]["image_hash"] == "hash_auto_pending"

        # Auto decisions are excluded from the training-eligible matrix before review.
        X_pre, y_pre, ids_pre = load_training_matrix(db_path=test_db, return_ids=True)
        assert set(ids_pre) == {id1, id2}
        assert X_pre.shape == (2, 768)

        # 3. Test bulk review updates (explicit human review converts provenance).
        updated_count = update_sample_reviews(
            updates=[{"id": id3, "label": 0, "reviewed": 1}],
            db_path=test_db,
        )
        assert updated_count == 1

        rechecked = get_samples(reviewed=0, db_path=test_db)
        assert len(rechecked) == 0

        reviewed_row = get_samples(label_provenance="review_confirmation", db_path=test_db)
        assert len(reviewed_row) == 1
        assert reviewed_row[0]["id"] == id3

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
        assert stats["training_eligible_count"] == 3
        assert stats["provenance_counts"]["manual_rating"] == 2
        assert stats["provenance_counts"]["review_confirmation"] == 1
        assert stats["provenance_counts"]["auto_decision"] == 0

        # 6. Test delete_samples
        from backend.database import delete_samples
        deleted_count, _ = delete_samples([id1, id2], db_path=test_db)
        assert deleted_count == 2
        remaining = get_samples(db_path=test_db)
        assert len(remaining) == 1
        assert remaining[0]["id"] == id3


def test_invalid_writes_rejected():
    """Invalid labels, scores, modes, review flags, provenance, and embeddings are rejected."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_dataset.db"
        init_db(test_db)
        vec = np.zeros(768, dtype=np.float32)

        with pytest.raises(ValueError):
            insert_sample("h_bad_label", "data/images/h_bad_label.jpg", vec,
                          label=2, mode="manual", reviewed=1, db_path=test_db)
        for bad_score in (-0.1, 1.5, float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValueError):
                insert_sample(f"h_bad_score_{bad_score}", "data/images/x.jpg", vec,
                              label=1, mode="manual", reviewed=1,
                              prediction_score=bad_score, db_path=test_db)
        with pytest.raises(ValueError):
            insert_sample("h_bad_mode", "data/images/h_bad_mode.jpg", vec,
                          label=1, mode="bogus", reviewed=1, db_path=test_db)
        with pytest.raises(ValueError):
            insert_sample("h_bad_reviewed", "data/images/h_bad_reviewed.jpg", vec,
                          label=1, mode="manual", reviewed=2, db_path=test_db)
        with pytest.raises(ValueError):
            insert_sample("h_bad_prov", "data/images/h_bad_prov.jpg", vec,
                          label=1, mode="manual", reviewed=1,
                          label_provenance="bogus", db_path=test_db)
        with pytest.raises(ValueError):
            insert_sample("h_bad_emb", "data/images/h_bad_emb.jpg", b"short",
                          label=1, mode="manual", reviewed=1, db_path=test_db)
        with pytest.raises(ValueError):
            insert_sample("h_bad_dim", "data/images/h_bad_dim.jpg",
                          np.zeros(10, dtype=np.float32),
                          label=1, mode="manual", reviewed=1, db_path=test_db)

        with pytest.raises(ValueError):
            update_sample_reviews(
                updates=[{"id": 1, "label": 5, "reviewed": 1}], db_path=test_db)

        # Database-level CHECK constraints reject raw invalid writes.
        good_bytes = vec.tobytes()
        conn = get_db_connection(test_db)
        try:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO samples (image_hash, file_path, label, mode, reviewed, label_provenance, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?);",
                    ("h_raw_label", "data/images/h_raw_label.jpg", 5, "manual", 1, "manual_rating", good_bytes),
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO samples (image_hash, file_path, label, mode, reviewed, label_provenance, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?);",
                    ("h_raw_emb", "data/images/h_raw_emb.jpg", 1, "manual", 1, "manual_rating", b"short"),
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO samples (image_hash, file_path, label, mode, reviewed, label_provenance, embedding)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?);",
                    ("h_raw_prov", "data/images/h_raw_prov.jpg", 1, "manual", 1, "bogus", good_bytes),
                )
        finally:
            conn.close()


def _create_legacy_db(path: Path) -> dict[str, dict]:
    """Build a legacy samples table without label_provenance or CHECK constraints."""
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute(
        """
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_hash TEXT UNIQUE NOT NULL,
            file_path TEXT NOT NULL,
            label INTEGER,
            prediction_score REAL,
            mode TEXT NOT NULL,
            reviewed INTEGER DEFAULT 0,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    vec_manual = np.zeros(768, dtype=np.float32)
    vec_manual[0] = 1.0
    vec_sup = np.zeros(768, dtype=np.float32)
    vec_sup[1] = 1.0
    vec_auto = np.zeros(768, dtype=np.float32)
    vec_auto[2] = 1.0
    rows = [
        ("legacy_manual", "data/images/legacy_manual.jpg", 1, 0.9, "manual", 1, vec_manual.tobytes(), "2024-01-01 00:00:01"),
        ("legacy_supervised", "data/images/legacy_supervised.jpg", 0, 0.2, "supervised", 1, vec_sup.tobytes(), "2024-01-02 00:00:02"),
        ("legacy_auto_reviewed", "data/images/legacy_auto_reviewed.jpg", 1, 0.8, "auto", 1, vec_auto.tobytes(), "2024-01-03 00:00:03"),
        ("legacy_auto_pending", "data/images/legacy_auto_pending.jpg", 0, 0.1, "auto", 0, vec_auto.tobytes(), "2024-01-04 00:00:04"),
    ]
    expected: dict[str, dict] = {}
    with conn:
        for image_hash, file_path, label, score, mode, reviewed, emb, created in rows:
            cur = conn.execute(
                "INSERT INTO samples (image_hash, file_path, label, prediction_score, mode, reviewed, embedding, created_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
                (image_hash, file_path, label, score, mode, reviewed, emb, created),
            )
            expected[image_hash] = {
                "id": cur.lastrowid,
                "file_path": file_path,
                "label": label,
                "embedding": emb,
                "created_at": created,
            }
    conn.close()
    return expected


def test_legacy_migration_classifies_and_preserves():
    """Legacy migration preserves rows and classifies provenance; auto rows become unreviewed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "legacy.db"
        expected = _create_legacy_db(test_db)

        init_db(test_db)

        samples = {s["image_hash"]: s for s in get_samples(db_path=test_db, limit=50)}
        assert samples["legacy_manual"]["label_provenance"] == "manual_rating"
        assert samples["legacy_manual"]["reviewed"] == 1
        assert samples["legacy_supervised"]["label_provenance"] == "supervised_confirmation"
        assert samples["legacy_supervised"]["reviewed"] == 1
        assert samples["legacy_auto_reviewed"]["label_provenance"] == "auto_decision"
        assert samples["legacy_auto_reviewed"]["reviewed"] == 0
        assert samples["legacy_auto_pending"]["label_provenance"] == "auto_decision"
        assert samples["legacy_auto_pending"]["reviewed"] == 0

        # IDs, hashes, paths, labels, embeddings, and timestamps are preserved.
        conn = get_db_connection(test_db)
        try:
            for image_hash, exp in expected.items():
                row = conn.execute(
                    "SELECT id, file_path, label, embedding, created_at FROM samples WHERE image_hash = ?;",
                    (image_hash,),
                ).fetchone()
                assert int(row["id"]) == exp["id"]
                assert row["file_path"] == exp["file_path"]
                assert row["label"] == exp["label"]
                assert bytes(row["embedding"]) == exp["embedding"]
                assert str(row["created_at"]) == exp["created_at"]
        finally:
            conn.close()

        # Training-eligible matrix contains only the two human-confirmed legacy rows.
        _, _, ids = load_training_matrix(db_path=test_db, return_ids=True)
        assert set(ids) == {expected["legacy_manual"]["id"], expected["legacy_supervised"]["id"]}

        # Migration is repeatable without changing already migrated rows.
        snapshot_before = sorted(
            (s["id"], s["label_provenance"], s["reviewed"], s["label"]) for s in get_samples(db_path=test_db, limit=50)
        )
        assert migrate_dataset_database(test_db) is False
        init_db(test_db)
        snapshot_after = sorted(
            (s["id"], s["label_provenance"], s["reviewed"], s["label"]) for s in get_samples(db_path=test_db, limit=50)
        )
        assert snapshot_before == snapshot_after


def test_auto_decision_excluded_until_review_confirmation():
    """Reviewed auto decisions stay out of the matrix until provenance becomes review_confirmation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_dataset.db"
        init_db(test_db)
        vec = np.zeros(768, dtype=np.float32)
        vec[0] = 1.0

        manual_id = insert_sample("m1", "data/images/m1.jpg", vec, label=1,
                                  mode="manual", reviewed=1, db_path=test_db)
        auto_id = insert_sample("a1", "data/images/a1.jpg", vec, label=1,
                                mode="auto", prediction_score=0.9, reviewed=0, db_path=test_db)

        _, _, ids = load_training_matrix(db_path=test_db, return_ids=True)
        assert ids == [manual_id]

        # Even a reviewed=1 auto_decision row is excluded when provenance is not human-confirmed.
        conn = get_db_connection(test_db)
        try:
            with conn:
                conn.execute(
                    "UPDATE samples SET reviewed = 1 WHERE id = ?;", (auto_id,)
                )
        finally:
            conn.close()
        _, _, ids_still = load_training_matrix(db_path=test_db, return_ids=True)
        assert ids_still == [manual_id]

        stats = get_dataset_statistics(db_path=test_db)
        assert stats["training_eligible_count"] == 1
        assert stats["provenance_counts"]["auto_decision"] == 1


def test_review_confirmation_included_in_matrix():
    """An explicit human review moves an auto decision into the training-eligible matrix."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_dataset.db"
        init_db(test_db)
        vec = np.zeros(768, dtype=np.float32)
        vec[3] = 1.0

        auto_id = insert_sample("a2", "data/images/a2.jpg", vec, label=0,
                                mode="auto", prediction_score=0.2, reviewed=0, db_path=test_db)
        X_empty, _, ids_empty = load_training_matrix(db_path=test_db, return_ids=True)
        assert ids_empty == []
        assert X_empty.shape == (0, 768)

        updated = update_sample_reviews(
            updates=[{"id": auto_id, "label": 1, "reviewed": 1,
                      "label_provenance": "review_confirmation"}],
            db_path=test_db,
        )
        assert updated == 1

        X, y, ids = load_training_matrix(db_path=test_db, return_ids=True)
        assert ids == [auto_id]
        assert X.shape == (1, 768)
        assert y.tolist() == [1]

        stats = get_dataset_statistics(db_path=test_db)
        assert stats["training_eligible_count"] == 1
        assert stats["training_eligible_positive_count"] == 1
        assert stats["provenance_counts"]["review_confirmation"] == 1

        row = get_samples(db_path=test_db)[0]
        assert row["label_provenance"] == "review_confirmation"
        assert row["reviewed"] == 1


def test_dataset_statistics_provenance_counts():
    """Statistics expose training-eligible counts and counts by provenance."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = Path(tmpdir) / "test_dataset.db"
        init_db(test_db)
        vec = np.zeros(768, dtype=np.float32)

        insert_sample("m1", "data/images/m1.jpg", vec, label=1,
                      mode="manual", reviewed=1, db_path=test_db)
        insert_sample("s1", "data/images/s1.jpg", vec, label=0,
                      mode="supervised", reviewed=1, db_path=test_db)
        insert_sample("a1", "data/images/a1.jpg", vec, label=1,
                      mode="auto", prediction_score=0.9, reviewed=0, db_path=test_db)

        stats = get_dataset_statistics(db_path=test_db)
        assert stats["total_samples"] == 3
        assert stats["training_eligible_count"] == 2
        assert stats["provenance_counts"] == {
            "manual_rating": 1,
            "supervised_confirmation": 1,
            "review_confirmation": 0,
            "auto_decision": 1,
        }
        assert stats["manual_rating_count"] == 1
        assert stats["supervised_confirmation_count"] == 1
        assert stats["review_confirmation_count"] == 0
        assert stats["auto_decision_count"] == 1


if __name__ == "__main__":
    test_database_initialization_and_wal()
    test_sample_crud_and_matrix_export()
    print("ALL CHECKPOINT 1 TESTS PASSED SUCCESSFULLY!")
