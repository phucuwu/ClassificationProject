"""Dataset database management for the local art taste classifier.

Handles SQLite database initialization, sample ingestion, label updates,
review queue queries, and feature matrix loading for model training.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

# Default file paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
DEFAULT_DB_PATH = DATA_DIR / "dataset.db"


def get_db_connection(db_path: Path | str | None = None) -> sqlite3.Connection:
    """Create and return a SQLite connection configured with WAL mode."""
    actual_path = Path(db_path) if db_path is not None else Path(DEFAULT_DB_PATH)
    actual_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(actual_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(db_path: Path | str | None = None) -> None:
    """Initialize dataset database schema and directories."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_hash TEXT UNIQUE NOT NULL,
                    file_path TEXT NOT NULL,
                    label INTEGER,               -- 1 = Like, 0 = Dislike, NULL = Unlabeled
                    prediction_score REAL,       -- Model score P(Like) from 0.0 to 1.0
                    mode TEXT NOT NULL,          -- 'manual', 'supervised', 'auto'
                    reviewed INTEGER DEFAULT 0,  -- 1 = Confirmed, 0 = Pending Review
                    embedding BLOB NOT NULL,     -- 768 float32 values (3072 bytes)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_reviewed ON samples(reviewed);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_mode ON samples(mode);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_image_hash ON samples(image_hash);"
            )
    finally:
        conn.close()


def insert_sample(
    image_hash: str,
    file_path: str,
    embedding: np.ndarray | bytes,
    label: int | None,
    mode: str,
    prediction_score: float | None = None,
    reviewed: int = 0,
    db_path: Path | str | None = None,
) -> int:
    """Insert or update a sample in the dataset database.

    Args:
        image_hash: SHA-256 hash of the primary image bytes.
        file_path: Local file path where the image is stored.
        embedding: 768-dimensional float32 NumPy array or raw bytes.
        label: 1 for Like (positive class), 0 for Dislike (negative class), or None.
        mode: Operating mode ('manual', 'supervised', 'auto').
        prediction_score: Optional model output probability.
        reviewed: 1 if confirmed by user, 0 if pending review.
        db_path: Optional path to SQLite database file.

    Returns:
        The integer ID of the inserted or updated sample row.
    """
    if isinstance(embedding, np.ndarray):
        embedding_bytes = embedding.astype(np.float32).tobytes()
    else:
        embedding_bytes = embedding

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO samples (
                    image_hash, file_path, label, prediction_score, mode, reviewed, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_hash) DO UPDATE SET
                    label = COALESCE(excluded.label, samples.label),
                    prediction_score = COALESCE(excluded.prediction_score, samples.prediction_score),
                    mode = excluded.mode,
                    reviewed = excluded.reviewed
                RETURNING id;
                """,
                (
                    image_hash,
                    file_path,
                    label,
                    prediction_score,
                    mode,
                    reviewed,
                    embedding_bytes,
                ),
            )
            row = cursor.fetchone()
            return int(row["id"]) if row else 0
    finally:
        conn.close()


def get_samples(
    mode: str | None = None,
    reviewed: int | None = None,
    label: int | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Query samples with optional filtering for the review queue or inspector."""
    conn = get_db_connection(db_path)
    try:
        conditions: list[str] = []
        params: list[Any] = []

        if mode is not None:
            conditions.append("mode = ?")
            params.append(mode)
        if reviewed is not None:
            conditions.append("reviewed = ?")
            params.append(reviewed)
        if label is not None:
            conditions.append("label = ?")
            params.append(label)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT id, image_hash, file_path, label, prediction_score, mode, reviewed, created_at
            FROM samples
            {where_clause}
            ORDER BY id DESC
            LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def update_sample_reviews(
    updates: list[dict[str, Any]],
    db_path: Path | str | None = None,
) -> int:
    """Bulk update labels and review flags for samples in the review queue.

    Args:
        updates: List of dicts, each with 'id', 'label', and optional 'reviewed'.
        db_path: Path to SQLite database file.

    Returns:
        Number of rows updated.
    """
    if not updates:
        return 0

    conn = get_db_connection(db_path)
    try:
        updated_count = 0
        with conn:
            for item in updates:
                sample_id = item["id"]
                label = item.get("label")
                reviewed = item.get("reviewed", 1)

                cursor = conn.execute(
                    """
                    UPDATE samples
                    SET label = ?, reviewed = ?
                    WHERE id = ?;
                    """,
                    (label, reviewed, sample_id),
                )
                updated_count += cursor.rowcount
        return updated_count
    finally:
        conn.close()


def load_training_matrix(
    db_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix (X) and label vector (y) from labeled samples.

    Returns:
        X: 2D NumPy array of shape (N, 768) with float32 embeddings.
        y: 1D NumPy array of shape (N,) with integer labels (0 or 1).
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT embedding, label
            FROM samples
            WHERE label IS NOT NULL
            ORDER BY id ASC;
            """
        )
        rows = cursor.fetchall()
        if not rows:
            return np.empty((0, 768), dtype=np.float32), np.empty((0,), dtype=np.int32)

        embeddings_list = [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
        labels_list = [int(row["label"]) for row in rows]

        X = np.vstack(embeddings_list).astype(np.float32)
        y = np.array(labels_list, dtype=np.int32)
        return X, y
    finally:
        conn.close()


def get_dataset_statistics(
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute aggregate dataset statistics for the developer dashboard."""
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT
                COUNT(*) AS total_samples,
                SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS positive_count,
                SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS negative_count,
                SUM(CASE WHEN label IS NULL THEN 1 ELSE 0 END) AS unlabeled_count,
                SUM(CASE WHEN reviewed = 0 AND mode = 'auto' THEN 1 ELSE 0 END) AS pending_auto_review_count,
                SUM(CASE WHEN mode = 'manual' THEN 1 ELSE 0 END) AS manual_mode_count,
                SUM(CASE WHEN mode = 'supervised' THEN 1 ELSE 0 END) AS supervised_mode_count,
                SUM(CASE WHEN mode = 'auto' THEN 1 ELSE 0 END) AS auto_mode_count
            FROM samples;
            """
        )
        row = cursor.fetchone()
        if not row:
            return {
                "total_samples": 0,
                "positive_count": 0,
                "negative_count": 0,
                "unlabeled_count": 0,
                "pending_auto_review_count": 0,
                "manual_mode_count": 0,
                "supervised_mode_count": 0,
                "auto_mode_count": 0,
                "positive_ratio": 0.0,
            }

        total = row["total_samples"] or 0
        pos = row["positive_count"] or 0
        neg = row["negative_count"] or 0
        ratio = (pos / (pos + neg)) if (pos + neg) > 0 else 0.0

        return {
            "total_samples": total,
            "positive_count": pos,
            "negative_count": neg,
            "unlabeled_count": row["unlabeled_count"] or 0,
            "pending_auto_review_count": row["pending_auto_review_count"] or 0,
            "manual_mode_count": row["manual_mode_count"] or 0,
            "supervised_mode_count": row["supervised_mode_count"] or 0,
            "auto_mode_count": row["auto_mode_count"] or 0,
            "positive_ratio": round(ratio, 4),
        }
    finally:
        conn.close()
