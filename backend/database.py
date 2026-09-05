"""Dataset database management for the local art taste classifier.

Handles SQLite database initialization, sample ingestion, label updates,
review queue queries, and feature matrix loading for model training.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Literal, overload

import numpy as np

# Default file paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
DEFAULT_DB_PATH = DATA_DIR / "dataset.db"

# Label provenance contract (see agent-docs/glossary.md).
VALID_MODES = ("manual", "supervised", "auto")
VALID_PROVENANCE = (
    "manual_rating",
    "supervised_confirmation",
    "review_confirmation",
    "auto_decision",
)
TRAINING_ELIGIBLE_PROVENANCE = (
    "manual_rating",
    "supervised_confirmation",
    "review_confirmation",
)
EMBEDDING_DIM = 768
EMBEDDING_NUM_BYTES = EMBEDDING_DIM * 4  # 3072 bytes for 768 float32 values

_SAMPLES_DDL = """
CREATE TABLE samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_hash TEXT UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    label INTEGER CHECK (label IS NULL OR label IN (0, 1)),
    prediction_score REAL CHECK (prediction_score IS NULL OR (prediction_score >= 0.0 AND prediction_score <= 1.0)),
    mode TEXT NOT NULL CHECK (mode IN ('manual', 'supervised', 'auto')),
    reviewed INTEGER NOT NULL DEFAULT 0 CHECK (reviewed IN (0, 1)),
    label_provenance TEXT NOT NULL DEFAULT 'auto_decision' CHECK (label_provenance IN ('manual_rating', 'supervised_confirmation', 'review_confirmation', 'auto_decision')),
    embedding BLOB NOT NULL CHECK (length(embedding) = 3072),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _validate_label(label: int | None) -> None:
    if label is not None and label not in (0, 1):
        raise ValueError(f"Invalid label {label!r}: expected 0, 1, or None.")


def _validate_prediction_score(prediction_score: float | None) -> None:
    if prediction_score is None:
        return
    try:
        score = float(prediction_score)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prediction_score {prediction_score!r}: not a number.") from exc
    if not np.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError(
            f"Invalid prediction_score {prediction_score!r}: expected finite float in [0.0, 1.0]."
        )


def _validate_mode(mode: str) -> None:
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode {mode!r}: expected one of {VALID_MODES}.")


def _validate_reviewed(reviewed: int) -> None:
    if reviewed not in (0, 1):
        raise ValueError(f"Invalid reviewed {reviewed!r}: expected 0 or 1.")


def _validate_provenance(label_provenance: str) -> None:
    if label_provenance not in VALID_PROVENANCE:
        raise ValueError(
            f"Invalid label_provenance {label_provenance!r}: expected one of {VALID_PROVENANCE}."
        )


def _validate_embedding_bytes(embedding_bytes: bytes) -> None:
    if not isinstance(embedding_bytes, (bytes, bytearray, memoryview)):
        raise ValueError("Invalid embedding: expected 768 float32 values (3072 bytes).")
    if len(embedding_bytes) != EMBEDDING_NUM_BYTES:
        raise ValueError(
            f"Invalid embedding length {len(embedding_bytes)} bytes: "
            f"expected {EMBEDDING_NUM_BYTES} bytes (768 float32 values)."
        )


def _derive_provenance(mode: str, reviewed: int, label_provenance: str | None) -> str:
    """Derive label provenance when the caller does not supply it explicitly."""
    if label_provenance is not None:
        _validate_provenance(label_provenance)
        return label_provenance
    if mode == "manual" and reviewed == 1:
        return "manual_rating"
    if mode == "supervised" and reviewed == 1:
        return "supervised_confirmation"
    return "auto_decision"


def _table_columns(conn: sqlite3.Connection) -> set[str]:
    cursor = conn.execute("PRAGMA table_info(samples);")
    return {row["name"] for row in cursor.fetchall()}


def _samples_create_sql(conn: sqlite3.Connection) -> str:
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'samples';")
    row = cursor.fetchone()
    return row["sql"] if row and row["sql"] else ""


def _needs_schema_rebuild(conn: sqlite3.Connection) -> bool:
    columns = _table_columns(conn)
    if "label_provenance" not in columns:
        return True
    sql = _samples_create_sql(conn)
    checks = (
        "label_provenance" in sql
        and "length(embedding)" in sql
        and "manual_rating" in sql
        and "supervised_confirmation" in sql
        and "review_confirmation" in sql
        and "auto_decision" in sql
    )
    return not checks


def migrate_dataset_database(db_path: Path | str | None = None) -> bool:
    """Apply the idempotent label-provenance migration to a Dataset database.

    Introduces the ``label_provenance`` column with ``CHECK`` constraints,
    classifies legacy rows (reviewed Manual-mode rows become ``manual_rating``,
    reviewed Supervised-mode rows become ``supervised_confirmation``, all
    Auto-mode rows become unreviewed ``auto_decision``), and preserves row IDs,
    image hashes, paths, labels, embeddings, and timestamps.

    Returns:
        True when a table rebuild was performed, False when the schema was
        already current.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'samples';"
        )
        if cursor.fetchone() is None:
            return False
        if not _needs_schema_rebuild(conn):
            with conn:
                conn.execute(
                    """
                    UPDATE samples
                    SET label_provenance = CASE
                        WHEN mode = 'manual' AND reviewed = 1 AND (label_provenance IS NULL OR label_provenance = '')
                            THEN 'manual_rating'
                        WHEN mode = 'supervised' AND reviewed = 1 AND (label_provenance IS NULL OR label_provenance = '')
                            THEN 'supervised_confirmation'
                        WHEN (label_provenance IS NULL OR label_provenance = '')
                            THEN 'auto_decision'
                        ELSE label_provenance
                    END
                    WHERE label_provenance IS NULL OR label_provenance = '';
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_samples_provenance ON samples(label_provenance);"
                )
            return False

        has_provenance = "label_provenance" in _table_columns(conn)
        with conn:
            conn.execute("PRAGMA foreign_keys = OFF;")
            conn.execute(_SAMPLES_DDL.replace("CREATE TABLE samples", "CREATE TABLE samples_new"))
            if has_provenance:
                conn.execute(
                    """
                    INSERT INTO samples_new (
                        id, image_hash, file_path, label, prediction_score,
                        mode, reviewed, label_provenance, embedding, created_at
                    )
                    SELECT
                        id, image_hash, file_path, label, prediction_score,
                        mode,
                        CASE WHEN mode = 'auto' AND reviewed != 0 THEN 0 ELSE reviewed END,
                        CASE
                            WHEN label_provenance IN ('manual_rating', 'supervised_confirmation', 'review_confirmation', 'auto_decision')
                                THEN CASE WHEN mode = 'auto' AND label_provenance != 'review_confirmation' THEN 'auto_decision' ELSE label_provenance END
                            WHEN mode = 'manual' AND reviewed = 1 THEN 'manual_rating'
                            WHEN mode = 'supervised' AND reviewed = 1 THEN 'supervised_confirmation'
                            ELSE 'auto_decision'
                        END,
                        embedding, created_at
                    FROM samples;
                    """
                )
            else:
                conn.execute(
                    """
                    INSERT INTO samples_new (
                        id, image_hash, file_path, label, prediction_score,
                        mode, reviewed, label_provenance, embedding, created_at
                    )
                    SELECT
                        id, image_hash, file_path, label, prediction_score,
                        mode,
                        CASE WHEN mode = 'auto' THEN 0 ELSE reviewed END,
                        CASE
                            WHEN mode = 'manual' AND reviewed = 1 THEN 'manual_rating'
                            WHEN mode = 'supervised' AND reviewed = 1 THEN 'supervised_confirmation'
                            ELSE 'auto_decision'
                        END,
                        embedding, created_at
                    FROM samples;
                    """
                )
            conn.execute("DROP TABLE samples;")
            conn.execute("ALTER TABLE samples_new RENAME TO samples;")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_reviewed ON samples(reviewed);"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_mode ON samples(mode);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label);")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_image_hash ON samples(image_hash);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_provenance ON samples(label_provenance);"
            )
            conn.execute("PRAGMA foreign_keys = ON;")
        return True
    finally:
        conn.close()


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
    """Initialize dataset database schema and directories.

    Fresh Dataset databases are created with label provenance and Sample
    invariants enforced. Legacy databases are migrated idempotently: reviewed
    Manual-mode Samples become ``manual_rating``, reviewed Supervised-mode
    Samples become ``supervised_confirmation``, and all Auto-mode Samples
    become unreviewed ``auto_decision``.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    conn = get_db_connection(db_path)
    try:
        with conn:
            # NOTE: on legacy databases this statement is a no-op because the
            # table already exists; the migration below rebuilds it.
            conn.execute(_SAMPLES_DDL.replace("CREATE TABLE samples", "CREATE TABLE IF NOT EXISTS samples"))
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

    migrate_dataset_database(db_path)

    conn = get_db_connection(db_path)
    try:
        with conn:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_samples_provenance ON samples(label_provenance);"
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
    label_provenance: str | None = None,
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
        label_provenance: Optional explicit label provenance. When omitted it is
            derived from the operating mode and review status (reviewed Manual
            mode becomes ``manual_rating``, reviewed Supervised mode becomes
            ``supervised_confirmation``, anything else becomes ``auto_decision``).
            Manual- and Supervised-mode inserts carry a human label by
            definition, so when provenance is not explicitly supplied and the
            label is not None, ``reviewed`` is coerced to ``1``.

    Returns:
        The integer ID of the inserted or updated sample row.
    """
    _validate_label(label)
    _validate_mode(mode)
    _validate_reviewed(reviewed)
    _validate_prediction_score(prediction_score)
    if label_provenance is None and mode in ("manual", "supervised") and label is not None:
        reviewed = 1
    provenance = _derive_provenance(mode, reviewed, label_provenance)

    if isinstance(embedding, np.ndarray):
        if embedding.size != EMBEDDING_DIM:
            raise ValueError(
                f"Invalid embedding size {embedding.size}: expected {EMBEDDING_DIM} float32 values."
            )
        embedding_bytes = embedding.astype(np.float32).tobytes()
    else:
        embedding_bytes = bytes(embedding)
    _validate_embedding_bytes(embedding_bytes)

    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO samples (
                    image_hash, file_path, label, prediction_score, mode, reviewed, label_provenance, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_hash) DO UPDATE SET
                    label = COALESCE(excluded.label, samples.label),
                    prediction_score = COALESCE(excluded.prediction_score, samples.prediction_score),
                    mode = excluded.mode,
                    reviewed = excluded.reviewed,
                    label_provenance = excluded.label_provenance
                RETURNING id;
                """,
                (
                    image_hash,
                    file_path,
                    label,
                    prediction_score,
                    mode,
                    reviewed,
                    provenance,
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
    sample_ids: list[int] | None = None,
    label_provenance: str | None = None,
    limit: int = 50,
    offset: int = 0,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Query samples with optional filtering for the review queue or inspector."""
    conn = get_db_connection(db_path)
    try:
        conditions: list[str] = []
        params: list[Any] = []

        if sample_ids is not None:
            if not sample_ids:
                return []
            placeholders = ",".join("?" for _ in sample_ids)
            conditions.append(f"id IN ({placeholders})")
            params.extend(sample_ids)
        if mode is not None:
            conditions.append("mode = ?")
            params.append(mode)
        if reviewed is not None:
            conditions.append("reviewed = ?")
            params.append(reviewed)
        if label is not None:
            conditions.append("label = ?")
            params.append(label)
        if label_provenance is not None:
            _validate_provenance(label_provenance)
            conditions.append("label_provenance = ?")
            params.append(label_provenance)

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        query = f"""
            SELECT id, image_hash, file_path, label, prediction_score, mode, reviewed, label_provenance, created_at
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

    An explicit human review converts ``auto_decision`` provenance to
    ``review_confirmation``. Rows that already carry human-confirmed provenance
    keep it unless the caller passes an explicit ``label_provenance``.

    Args:
        updates: List of dicts, each with 'id', 'label', optional 'reviewed',
            and optional 'label_provenance'.
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
                explicit_provenance = item.get("label_provenance")

                _validate_label(label)
                _validate_reviewed(reviewed)
                if explicit_provenance is not None:
                    _validate_provenance(explicit_provenance)

                if explicit_provenance is not None:
                    provenance_expr = "?"
                    provenance_param: Any = explicit_provenance
                elif reviewed == 1:
                    # Explicit review: auto decisions become review confirmations;
                    # existing human-confirmed provenance is preserved.
                    provenance_expr = (
                        "CASE WHEN label_provenance = 'auto_decision' "
                        "THEN 'review_confirmation' ELSE label_provenance END"
                    )
                    provenance_param = None
                else:
                    provenance_expr = "label_provenance"
                    provenance_param = None

                if provenance_param is not None:
                    cursor = conn.execute(
                        f"""
                        UPDATE samples
                        SET label = ?, reviewed = ?, label_provenance = {provenance_expr}
                        WHERE id = ?;
                        """,
                        (label, reviewed, provenance_param, sample_id),
                    )
                else:
                    cursor = conn.execute(
                        f"""
                        UPDATE samples
                        SET label = ?, reviewed = ?, label_provenance = {provenance_expr}
                        WHERE id = ?;
                        """,
                        (label, reviewed, sample_id),
                    )
                updated_count += cursor.rowcount
        return updated_count
    finally:
        conn.close()


def delete_samples(
    sample_ids: list[int],
    db_path: Path | str | None = None,
) -> tuple[int, list[str]]:
    """Delete samples by IDs from the database and remove local image files.

    Args:
        sample_ids: List of integer sample IDs to delete.
        db_path: Optional path to SQLite database file.

    Returns:
        Tuple of (number of deleted rows, list of file paths removed).
    """
    if not sample_ids:
        return 0, []

    conn = get_db_connection(db_path)
    try:
        placeholders = ",".join("?" for _ in sample_ids)
        with conn:
            cursor = conn.execute(
                f"SELECT id, file_path FROM samples WHERE id IN ({placeholders});",
                sample_ids,
            )
            rows = cursor.fetchall()
            file_paths = [row["file_path"] for row in rows]

            del_cursor = conn.execute(
                f"DELETE FROM samples WHERE id IN ({placeholders});",
                sample_ids,
            )
            deleted_count = del_cursor.rowcount

        # Clean up local image files if they exist
        for rel_path in file_paths:
            p = Path(rel_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists() and p.is_file():
                try:
                    p.unlink()
                except OSError:
                    pass

        return deleted_count, file_paths
    finally:
        conn.close()


@overload
def load_training_matrix(
    db_path: Path | str | None = None,
    return_ids: Literal[False] = False,
    training_eligible_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    ...


@overload
def load_training_matrix(
    db_path: Path | str | None = None,
    return_ids: Literal[True] = ...,
    training_eligible_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    ...


@overload
def load_training_matrix(
    db_path: Path | str | None = None,
    return_ids: bool = False,
    training_eligible_only: bool = True,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[int]]:
    ...


def load_training_matrix(
    db_path: Path | str | None = None,
    return_ids: bool = False,
    training_eligible_only: bool = True,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract feature matrix (X) and label vector (y) from training-eligible samples.

    By default only training-eligible Samples are returned: rows with a binary
    label, ``reviewed = 1``, and label provenance ``manual_rating``,
    ``supervised_confirmation``, or ``review_confirmation``. Samples with
    ``auto_decision`` provenance are excluded until an explicit human review
    changes their provenance to ``review_confirmation``. Outlier analysis
    callers use this same default matrix as model training.

    Args:
        db_path: Optional path to SQLite database file.
        return_ids: If True, also returns list of integer sample IDs.
        training_eligible_only: When False, return all labeled samples
            regardless of review status and provenance (legacy inspection only).

    Returns:
        X: 2D NumPy array of shape (N, 768) with float32 embeddings.
        y: 1D NumPy array of shape (N,) with integer labels (0 or 1).
        (Optional) sample_ids: List of integer sample IDs when return_ids=True.
    """
    conn = get_db_connection(db_path)
    try:
        if training_eligible_only:
            cursor = conn.execute(
                """
                SELECT id, embedding, label
                FROM samples
                WHERE label IS NOT NULL
                  AND reviewed = 1
                  AND label_provenance IN ('manual_rating', 'supervised_confirmation', 'review_confirmation')
                ORDER BY id ASC;
                """
            )
        else:
            cursor = conn.execute(
                """
                SELECT id, embedding, label
                FROM samples
                WHERE label IS NOT NULL
                ORDER BY id ASC;
                """
            )
        rows = cursor.fetchall()
        if not rows:
            empty_X = np.empty((0, 768), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.int32)
            return (empty_X, empty_y, []) if return_ids else (empty_X, empty_y)

        sample_ids = [int(row["id"]) for row in rows]
        embeddings_list = [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
        labels_list = [int(row["label"]) for row in rows]

        X = np.vstack(embeddings_list).astype(np.float32)
        y = np.array(labels_list, dtype=np.int32)
        if return_ids:
            return X, y, sample_ids
        return X, y
    finally:
        conn.close()


def get_dataset_statistics(
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute aggregate dataset statistics for the developer dashboard.

    Existing aggregate keys keep their all-Sample semantics. Additional keys
    expose training-eligible Samples and counts by label provenance.
    """
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
                SUM(CASE WHEN mode = 'auto' THEN 1 ELSE 0 END) AS auto_mode_count,
                SUM(CASE WHEN label IS NOT NULL AND reviewed = 1
                           AND label_provenance IN ('manual_rating', 'supervised_confirmation', 'review_confirmation')
                         THEN 1 ELSE 0 END) AS training_eligible_count,
                SUM(CASE WHEN label IS NOT NULL AND reviewed = 1
                           AND label_provenance IN ('manual_rating', 'supervised_confirmation', 'review_confirmation')
                           AND label = 1
                         THEN 1 ELSE 0 END) AS training_eligible_positive_count,
                SUM(CASE WHEN label IS NOT NULL AND reviewed = 1
                           AND label_provenance IN ('manual_rating', 'supervised_confirmation', 'review_confirmation')
                           AND label = 0
                         THEN 1 ELSE 0 END) AS training_eligible_negative_count,
                SUM(CASE WHEN label_provenance = 'manual_rating' THEN 1 ELSE 0 END) AS manual_rating_count,
                SUM(CASE WHEN label_provenance = 'supervised_confirmation' THEN 1 ELSE 0 END) AS supervised_confirmation_count,
                SUM(CASE WHEN label_provenance = 'review_confirmation' THEN 1 ELSE 0 END) AS review_confirmation_count,
                SUM(CASE WHEN label_provenance = 'auto_decision' THEN 1 ELSE 0 END) AS auto_decision_count
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
                "training_eligible_count": 0,
                "training_eligible_positive_count": 0,
                "training_eligible_negative_count": 0,
                "manual_rating_count": 0,
                "supervised_confirmation_count": 0,
                "review_confirmation_count": 0,
                "auto_decision_count": 0,
                "provenance_counts": {
                    "manual_rating": 0,
                    "supervised_confirmation": 0,
                    "review_confirmation": 0,
                    "auto_decision": 0,
                },
            }

        total = row["total_samples"] or 0
        pos = row["positive_count"] or 0
        neg = row["negative_count"] or 0
        ratio = (pos / (pos + neg)) if (pos + neg) > 0 else 0.0

        manual_rating = row["manual_rating_count"] or 0
        supervised_confirmation = row["supervised_confirmation_count"] or 0
        review_confirmation = row["review_confirmation_count"] or 0
        auto_decision = row["auto_decision_count"] or 0

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
            "training_eligible_count": row["training_eligible_count"] or 0,
            "training_eligible_positive_count": row["training_eligible_positive_count"] or 0,
            "training_eligible_negative_count": row["training_eligible_negative_count"] or 0,
            "manual_rating_count": manual_rating,
            "supervised_confirmation_count": supervised_confirmation,
            "review_confirmation_count": review_confirmation,
            "auto_decision_count": auto_decision,
            "provenance_counts": {
                "manual_rating": manual_rating,
                "supervised_confirmation": supervised_confirmation,
                "review_confirmation": review_confirmation,
                "auto_decision": auto_decision,
            },
        }
    finally:
        conn.close()


def load_embedding_scatter_data(
    db_path: Path | str | None = None,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Retrieve all samples with metadata and the (N, 768) vision embedding matrix.

    Returns:
        samples_metadata: List of dicts with id, image_hash, file_path, label,
                          prediction_score, mode, reviewed, and created_at.
        X: 2D NumPy array of shape (N, 768) with float32 embeddings.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, image_hash, file_path, label, prediction_score, mode, reviewed, label_provenance, created_at, embedding
            FROM samples
            WHERE embedding IS NOT NULL
            ORDER BY id ASC;
            """
        )
        rows = cursor.fetchall()
        if not rows:
            return [], np.empty((0, 768), dtype=np.float32)

        metadata_list: list[dict[str, Any]] = []
        embeddings_list: list[np.ndarray] = []

        for row in rows:
            metadata_list.append(
                {
                    "id": row["id"],
                    "image_hash": row["image_hash"],
                    "file_path": row["file_path"],
                    "label": row["label"],
                    "prediction_score": row["prediction_score"],
                    "mode": row["mode"],
                    "reviewed": row["reviewed"],
                    "label_provenance": row["label_provenance"],
                    "created_at": row["created_at"],
                }
            )
            embeddings_list.append(np.frombuffer(row["embedding"], dtype=np.float32))

        X = np.vstack(embeddings_list).astype(np.float32)
        return metadata_list, X
    finally:
        conn.close()


def find_near_duplicate(
    embedding: np.ndarray,
    threshold: float = 0.98,
    db_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Find an existing sample in the dataset database with cosine similarity >= threshold.

    Args:
        embedding: 768-dimensional float32 vision embedding.
        threshold: Cosine similarity threshold (default 0.98).
        db_path: Optional path to SQLite database file.

    Returns:
        Dictionary with id, similarity, image_hash, file_path, label, reviewed if found, else None.
    """
    metadata, X = load_embedding_scatter_data(db_path=db_path)
    if len(X) == 0:
        return None

    v = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm

    similarities = X @ v
    max_idx = int(np.argmax(similarities))
    max_sim = float(similarities[max_idx])

    if max_sim >= threshold:
        matched = metadata[max_idx]
        return {
            "id": matched["id"],
            "similarity": round(max_sim, 4),
            "image_hash": matched["image_hash"],
            "file_path": matched["file_path"],
            "label": matched["label"],
            "mode": matched["mode"],
            "reviewed": matched["reviewed"],
            "label_provenance": matched.get("label_provenance"),
        }
    return None


def update_sample_record(
    sample_id: int,
    label: int | None = None,
    mode: str | None = None,
    prediction_score: float | None = None,
    reviewed: int | None = None,
    db_path: Path | str | None = None,
    label_provenance: str | None = None,
) -> bool:
    """Update metadata for an existing sample record.

    Args:
        sample_id: Sample integer database ID.
        label: Optional updated label (1, 0, or None).
        mode: Optional updated mode ('manual', 'supervised', 'auto').
        prediction_score: Optional model prediction score.
        reviewed: Optional reviewed flag (1 or 0).
        db_path: Optional path to SQLite database file.
        label_provenance: Optional explicit label provenance.

    Returns:
        True if the row was updated, False otherwise.
    """
    updates: list[str] = []
    params: list[Any] = []

    if label is not None:
        _validate_label(label)
        updates.append("label = ?")
        params.append(label)
    if mode is not None:
        _validate_mode(mode)
        updates.append("mode = ?")
        params.append(mode)
    if prediction_score is not None:
        _validate_prediction_score(prediction_score)
        updates.append("prediction_score = ?")
        params.append(prediction_score)
    if reviewed is not None:
        _validate_reviewed(reviewed)
        updates.append("reviewed = ?")
        params.append(reviewed)
    if label_provenance is not None:
        _validate_provenance(label_provenance)
        updates.append("label_provenance = ?")
        params.append(label_provenance)

    if not updates:
        return False

    params.append(sample_id)
    conn = get_db_connection(db_path)
    try:
        with conn:
            cursor = conn.execute(
                f"UPDATE samples SET {', '.join(updates)} WHERE id = ?;",
                params,
            )
            return cursor.rowcount > 0
    finally:
        conn.close()


def get_recent_samples(
    limit: int = 100,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the most recent N samples for rolling session drift analysis.

    Args:
        limit: Number of recent samples to query (default 100).
        db_path: Optional path to SQLite database file.

    Returns:
        List of sample dictionaries with id, label, prediction_score, mode, reviewed, and created_at.
    """
    conn = get_db_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, label, prediction_score, mode, reviewed, created_at
            FROM samples
            ORDER BY id DESC
            LIMIT ?;
            """,
            (limit,),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()



