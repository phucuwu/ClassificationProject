"""Vision embedding backbone benchmark evaluation engine.

Evaluates alternative vision backbones against baseline CLIP (clip-ViT-L-14)
using stratified cross-validation, hyperparameter grid search, and
Precision-Recall metrics on subjective art taste classification.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, average_precision_score, fbeta_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold

# Suppress Hugging Face Hub token notice
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")

# Default file paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "dataset.db"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_FILE = CACHE_DIR / "backbone_benchmark_results.json"

# Candidate backbones
SUPPORTED_BACKBONES: dict[str, dict[str, Any]] = {
    "clip-ViT-L-14": {
        "id": "clip-ViT-L-14",
        "name": "CLIP ViT-L/14 (Baseline)",
        "family": "clip",
        "dim": 768,
        "hf_id": "clip-ViT-L-14",
    },
    "google/siglip-so400m-patch14-384": {
        "id": "google/siglip-so400m-patch14-384",
        "name": "Google SigLIP SO400M (384px)",
        "family": "siglip",
        "dim": 1152,
        "hf_id": "google/siglip-so400m-patch14-384",
    },
    "facebook/dinov2-base": {
        "id": "facebook/dinov2-base",
        "name": "Meta DINOv2 Base",
        "family": "dinov2",
        "dim": 768,
        "hf_id": "facebook/dinov2-base",
    },
}

HYPERPARAMETER_C_GRID = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def get_device(requested_device: str | None = None) -> torch.device:
    """Return compute device based on availability and optional override."""
    if requested_device:
        return torch.device(requested_device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset_samples(
    db_path: Path | str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load labeled samples with image file paths from SQLite database."""
    actual_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    if not actual_path.exists():
        return []

    conn = sqlite3.connect(str(actual_path))
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT id, file_path, label, embedding
            FROM samples
            WHERE label IS NOT NULL
            ORDER BY id ASC
        """
        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"

        rows = conn.execute(query).fetchall()
        samples = []
        for r in rows:
            # Resolve relative file paths
            img_path = PROJECT_ROOT / r["file_path"] if not Path(r["file_path"]).is_absolute() else Path(r["file_path"])
            samples.append({
                "id": int(r["id"]),
                "file_path": str(img_path),
                "label": int(r["label"]),
                "raw_embedding": bytes(r["embedding"]) if r["embedding"] is not None else None,
            })
        return samples
    finally:
        conn.close()


def extract_clip_embeddings_from_db(
    samples: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract baseline CLIP embeddings directly from database BLOBs."""
    valid_ids: list[int] = []
    valid_labels: list[int] = []
    valid_embeddings: list[np.ndarray] = []

    for s in samples:
        blob = s.get("raw_embedding")
        if blob and len(blob) == 768 * 4:  # 768 float32s
            vec = np.frombuffer(blob, dtype=np.float32).copy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            valid_embeddings.append(vec)
            valid_ids.append(s["id"])
            valid_labels.append(s["label"])

    if not valid_embeddings:
        return np.empty((0, 768), dtype=np.float32), np.empty((0,), dtype=np.int32), []

    return (
        np.vstack(valid_embeddings).astype(np.float32),
        np.array(valid_labels, dtype=np.int32),
        valid_ids,
    )


def extract_backbone_embeddings(
    model_id: str,
    samples: list[dict[str, Any]],
    batch_size: int = 32,
    device: torch.device | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract vision embeddings for given backbone model across samples."""
    target_device = device or get_device()
    valid_samples = [s for s in samples if Path(s["file_path"]).exists()]
    total_count = len(valid_samples)

    if total_count == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32), []

    if model_id == "clip-ViT-L-14":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_id, device=str(target_device))
        embeddings_list: list[np.ndarray] = []
        labels_list: list[int] = []
        ids_list: list[int] = []

        for i in range(0, total_count, batch_size):
            batch = valid_samples[i : i + batch_size]
            pil_images = []
            batch_labels = []
            batch_ids = []
            for item in batch:
                try:
                    img = Image.open(item["file_path"]).convert("RGB")
                    pil_images.append(img)
                    batch_labels.append(item["label"])
                    batch_ids.append(item["id"])
                except Exception:
                    continue

            if pil_images:
                batch_emb = model.encode(
                    pil_images,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=len(pil_images),
                    show_progress_bar=False,
                )
                embeddings_list.append(batch_emb.astype(np.float32))
                labels_list.extend(batch_labels)
                ids_list.extend(batch_ids)

            if progress_callback:
                progress_callback(min(i + batch_size, total_count), total_count, model_id)

        if not embeddings_list:
            return np.empty((0, 768), dtype=np.float32), np.empty((0,), dtype=np.int32), []
        return np.vstack(embeddings_list), np.array(labels_list, dtype=np.int32), ids_list

    # Hugging Face transformers backbones (SigLIP, DINOv2)
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(target_device)
    model.eval()

    embeddings_list = []
    labels_list = []
    ids_list = []

    for i in range(0, total_count, batch_size):
        batch = valid_samples[i : i + batch_size]
        pil_images = []
        batch_labels = []
        batch_ids = []
        for item in batch:
            try:
                img = Image.open(item["file_path"]).convert("RGB")
                pil_images.append(img)
                batch_labels.append(item["label"])
                batch_ids.append(item["id"])
            except Exception:
                continue

        if pil_images:
            inputs = processor(images=pil_images, return_tensors="pt").to(target_device)
            with torch.no_grad():
                if hasattr(model, "get_image_features"):
                    features = model.get_image_features(**inputs)
                else:
                    outputs = model(**inputs)
                    features = outputs.last_hidden_state[:, 0, :]
                # L2 normalize
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                emb_np = features.cpu().numpy().astype(np.float32)

            embeddings_list.append(emb_np)
            labels_list.extend(batch_labels)
            ids_list.extend(batch_ids)

        if progress_callback:
            progress_callback(min(i + batch_size, total_count), total_count, model_id)

    # Free GPU memory
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not embeddings_list:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int32), []
    return np.vstack(embeddings_list), np.array(labels_list, dtype=np.int32), ids_list


def get_cached_embeddings(
    model_id: str,
    samples: list[dict[str, Any]],
    force_extract: bool = False,
    batch_size: int = 32,
    device: torch.device | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], float]:
    """Retrieve embeddings from disk cache or extract fresh embeddings."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = model_id.replace("/", "_").replace("-", "_")
    cache_path = CACHE_DIR / f"benchmark_{safe_name}.npz"

    current_ids = [s["id"] for s in samples]

    start_time = time.time()
    # Check disk cache
    if not force_extract and cache_path.exists():
        try:
            with np.load(cache_path) as data:
                cached_ids = data["sample_ids"].tolist()
                cached_embs = data["embeddings"]
                cached_labels = data["labels"]

                # If cached sample IDs match requested samples, use cache
                if cached_ids == current_ids:
                    return cached_embs, cached_labels, cached_ids, 0.0
        except Exception:
            pass

    # For clip baseline, use database BLOBs if not force-extracting
    if model_id == "clip-ViT-L-14" and not force_extract:
        X, y, sample_ids = extract_clip_embeddings_from_db(samples)
        if len(y) == len(samples):
            extraction_duration = round(time.time() - start_time, 2)
            np.savez_compressed(
                cache_path,
                sample_ids=np.array(sample_ids, dtype=np.int32),
                embeddings=X,
                labels=y,
            )
            return X, y, sample_ids, extraction_duration

    # Extract fresh embeddings
    X, y, sample_ids = extract_backbone_embeddings(
        model_id=model_id,
        samples=samples,
        batch_size=batch_size,
        device=device,
        progress_callback=progress_callback,
    )
    extraction_duration = round(time.time() - start_time, 2)

    if len(y) > 0:
        np.savez_compressed(
            cache_path,
            sample_ids=np.array(sample_ids, dtype=np.int32),
            embeddings=X,
            labels=y,
        )

    return X, y, sample_ids, extraction_duration


def evaluate_backbone_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    min_recall_floor: float = 0.70,
) -> dict[str, Any]:
    """Evaluate taste classification metrics using Stratified 5-Fold Cross-Validation."""
    if len(y) == 0 or np.sum(y == 1) < 2 or np.sum(y == 0) < 2:
        return {
            "error": "Insufficient labeled samples for cross-validation.",
            "sample_count": len(y),
            "positive_count": int(np.sum(y == 1)) if len(y) > 0 else 0,
            "negative_count": int(np.sum(y == 0)) if len(y) > 0 else 0,
        }

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    ratio = float(neg_count / pos_count) if pos_count > 0 else 1.0

    weight_candidates = [
        ("balanced", "balanced"),
        ("unweighted", None),
        ("balanced_1.5x", {0: 1.0, 1: float(1.5 * ratio)}),
        ("balanced_2.0x", {0: 1.0, 1: float(2.0 * ratio)}),
    ]

    effective_splits = min(n_splits, pos_count)
    if effective_splits < 2:
        return {"error": "Need at least 2 positive samples to perform stratified cross-validation."}

    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)

    # 1. Hyperparameter tuning across folds
    best_c = 1.0
    best_weight_key = "balanced"
    best_weight_param: Any = "balanced"
    best_grid_prauc = -1.0

    for c_val in HYPERPARAMETER_C_GRID:
        for w_key, w_param in weight_candidates:
            fold_scores = []
            for train_idx, val_idx in skf.split(X, y):
                clf = LogisticRegression(
                    class_weight=w_param,
                    C=c_val,
                    max_iter=1000,
                    solver="lbfgs",
                    random_state=42,
                )
                clf.fit(X[train_idx], y[train_idx])
                probs = clf.predict_proba(X[val_idx])[:, 1]
                prec, rec, _ = precision_recall_curve(y[val_idx], probs)
                f_score = float(auc(rec, prec)) if len(rec) > 1 else 0.0
                fold_scores.append(f_score)

            mean_score = float(np.mean(fold_scores))
            if mean_score > best_grid_prauc:
                best_grid_prauc = mean_score
                best_c = c_val
                best_weight_key = w_key
                best_weight_param = w_param

    # 2. Generate out-of-fold probability predictions with best parameters
    oof_probabilities = np.zeros(len(y), dtype=np.float32)
    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(
            class_weight=best_weight_param,
            C=best_c,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X[train_idx], y[train_idx])
        oof_probabilities[val_idx] = clf.predict_proba(X[val_idx])[:, 1]

    # 3. Calculate evaluation metrics on out-of-fold probabilities
    precisions, recalls, pr_thresholds = precision_recall_curve(y, oof_probabilities)
    pr_auc_score = float(auc(recalls, precisions)) if len(recalls) > 1 else 0.0
    ap_score = float(average_precision_score(y, oof_probabilities))

    # 4. Calibrate decision threshold maximizing F2 with recall floor
    candidate_cutoffs = np.linspace(0.05, 0.95, 91)
    valid_candidates = []
    all_candidates = []

    for cutoff in candidate_cutoffs:
        preds = (oof_probabilities >= cutoff).astype(int)
        c_tp = np.sum((preds == 1) & (y == 1))
        c_fn = np.sum((preds == 0) & (y == 1))
        c_rec = float(c_tp / (c_tp + c_fn)) if (c_tp + c_fn) > 0 else 0.0
        c_f2 = float(fbeta_score(y, preds, beta=2, zero_division=0))
        all_candidates.append((c_f2, c_rec, float(cutoff)))
        if c_rec >= min_recall_floor:
            valid_candidates.append((c_f2, c_rec, float(cutoff)))

    if valid_candidates:
        valid_candidates.sort(key=lambda item: (item[0], item[2]), reverse=True)
        calibrated_threshold = valid_candidates[0][2]
    else:
        all_candidates.sort(key=lambda item: (item[1], item[0]), reverse=True)
        calibrated_threshold = all_candidates[0][2]

    calibrated_threshold = float(round(np.clip(calibrated_threshold, 0.05, 0.95), 2))

    # Metrics at calibrated threshold
    binary_preds = (oof_probabilities >= calibrated_threshold).astype(int)
    tp = int(np.sum((binary_preds == 1) & (y == 1)))
    fp = int(np.sum((binary_preds == 1) & (y == 0)))
    tn = int(np.sum((binary_preds == 0) & (y == 0)))
    fn = int(np.sum((binary_preds == 0) & (y == 1)))

    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    f2 = float(fbeta_score(y, binary_preds, beta=2, zero_division=0))

    return {
        "pr_auc": round(pr_auc_score, 4),
        "average_precision": round(ap_score, 4),
        "recall": round(rec, 4),
        "precision": round(prec, 4),
        "f2_score": round(f2, 4),
        "decision_threshold": calibrated_threshold,
        "sample_count": len(y),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
        "best_c": best_c,
        "best_class_weight": best_weight_key,
        "folds": effective_splits,
    }


def run_backbone_benchmark(
    models: list[str] | None = None,
    limit: int | None = None,
    batch_size: int = 32,
    force_extract: bool = False,
    db_path: Path | str | None = None,
    device: str | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Execute offline benchmark comparing vision backbones on labeled dataset."""
    selected_models = models or list(SUPPORTED_BACKBONES.keys())
    device_obj = get_device(device)

    samples = load_dataset_samples(db_path=db_path, limit=limit)
    total_samples = len(samples)

    if total_samples == 0:
        return {
            "status": "error",
            "message": "No labeled samples found in dataset database.",
            "results": [],
        }

    benchmark_start = time.time()
    results: list[dict[str, Any]] = []

    for idx, model_id in enumerate(selected_models):
        meta = SUPPORTED_BACKBONES.get(model_id, {
            "id": model_id,
            "name": model_id,
            "dim": 768,
        })

        if progress_callback:
            progress_callback({
                "status": "running",
                "current_model": meta["name"],
                "model_id": model_id,
                "model_index": idx + 1,
                "total_models": len(selected_models),
                "processed_samples": 0,
                "total_samples": total_samples,
                "percent": round((idx / len(selected_models)) * 100, 1),
                "message": f"Extracting embeddings for {meta['name']}...",
            })

        def sub_progress(processed: int, total: int, m_name: str) -> None:
            if progress_callback:
                model_base = (idx / len(selected_models)) * 100
                model_chunk = (processed / total) * (100 / len(selected_models))
                progress_callback({
                    "status": "running",
                    "current_model": meta["name"],
                    "model_id": model_id,
                    "model_index": idx + 1,
                    "total_models": len(selected_models),
                    "processed_samples": processed,
                    "total_samples": total,
                    "percent": round(model_base + model_chunk, 1),
                    "message": f"Extracting {meta['name']}: {processed}/{total} samples",
                })

        # 1. Extract or load embeddings
        X, y, valid_ids, extract_time = get_cached_embeddings(
            model_id=model_id,
            samples=samples,
            force_extract=force_extract,
            batch_size=batch_size,
            device=device_obj,
            progress_callback=sub_progress,
        )

        # 2. Evaluate with Stratified CV
        eval_start = time.time()
        metrics = evaluate_backbone_cv(X, y)
        eval_duration = round(time.time() - eval_start, 2)

        speed = round(len(X) / extract_time, 1) if extract_time > 0 else None

        res_item = {
            "model_id": model_id,
            "model_name": meta["name"],
            "embedding_dim": int(X.shape[1]) if len(X) > 0 else meta.get("dim", 768),
            "sample_count": len(X),
            "positive_count": int(np.sum(y == 1)) if len(y) > 0 else 0,
            "negative_count": int(np.sum(y == 0)) if len(y) > 0 else 0,
            "extraction_duration_seconds": extract_time,
            "evaluation_duration_seconds": eval_duration,
            "samples_per_second": speed,
            "metrics": metrics,
            "is_baseline": (model_id == "clip-ViT-L-14"),
        }
        results.append(res_item)

    # Determine top performing model by PR-AUC (with F2 tiebreak)
    valid_results = [r for r in results if "error" not in r.get("metrics", {})]
    if valid_results:
        valid_results.sort(
            key=lambda r: (
                r["metrics"].get("pr_auc", 0.0),
                r["metrics"].get("f2_score", 0.0),
            ),
            reverse=True,
        )
        winner_id = valid_results[0]["model_id"]
        for r in results:
            r["is_winner"] = (r["model_id"] == winner_id)

    total_duration = round(time.time() - benchmark_start, 2)
    output_payload = {
        "status": "completed",
        "sample_count": total_samples,
        "total_duration_seconds": total_duration,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": str(device_obj),
        "results": results,
    }

    # Cache results to disk
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2)
    except Exception:
        pass

    if progress_callback:
        progress_callback({
            "status": "completed",
            "current_model": None,
            "percent": 100.0,
            "message": "Benchmark completed successfully.",
            "results": results,
        })

    return output_payload


def format_cli_table(results: list[dict[str, Any]]) -> str:
    """Format benchmark results into a clean terminal text table."""
    headers = ["Model Backbone", "Dim", "PR-AUC", "F2 Score", "Recall", "Prec", "Best C", "Weight", "Speed"]
    rows = []

    for r in results:
        m = r.get("metrics", {})
        if "error" in m:
            rows.append([r["model_name"], str(r["embedding_dim"]), "ERR", "-", "-", "-", "-", "-", "-"])
            continue

        speed_str = f"{r['samples_per_second']} s/s" if r.get("samples_per_second") else "cached"
        name_str = ("* " if r.get("is_winner") else "  ") + r["model_name"]
        rows.append([
            name_str,
            str(r["embedding_dim"]),
            f"{m.get('pr_auc', 0.0):.4f}",
            f"{m.get('f2_score', 0.0):.4f}",
            f"{m.get('recall', 0.0):.4f}",
            f"{m.get('precision', 0.0):.4f}",
            str(m.get("best_c", "-")),
            str(m.get("best_class_weight", "-")),
            speed_str,
        ])

    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"

    lines = [separator]
    lines.append("| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |")
    lines.append(separator)
    for r in rows:
        lines.append("| " + " | ".join(r[i].ljust(col_widths[i]) for i in range(len(headers))) + " |")
    lines.append(separator)
    return "\n".join(lines)


def main() -> None:
    """CLI entry point for offline vision backbone benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark vision backbones for the subjective art taste classifier."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(SUPPORTED_BACKBONES.keys()),
        help="List of model backbone IDs to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples evaluated (default: all labeled samples).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for feature extraction (default: 32).",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Bypass embedding cache and re-extract all vision embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override compute device (e.g. 'cuda', 'cpu').",
    )

    args = parser.parse_args()

    print("\nStarting Vision Backbone Benchmark...")
    print(f"Candidate backbones: {args.models}")
    print(f"Device: {get_device(args.device)}")
    if args.limit:
        print(f"Sample limit: {args.limit}")

    def cli_progress(p: dict[str, Any]) -> None:
        pct = p.get("percent", 0.0)
        msg = p.get("message", "")
        sys.stdout.write(f"\r[{pct:>5.1f}%] {msg:<60}")
        sys.stdout.flush()

    output = run_backbone_benchmark(
        models=args.models,
        limit=args.limit,
        batch_size=args.batch_size,
        force_extract=args.force_extract,
        device=args.device,
        progress_callback=cli_progress,
    )

    print("\n\n" + format_cli_table(output["results"]))
    print(f"\nResults written to: {RESULTS_FILE}\n")


if __name__ == "__main__":
    main()
