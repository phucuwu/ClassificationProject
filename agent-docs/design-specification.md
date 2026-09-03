# Design specification: Art taste classifier

This document is the definitive design and implementation specification for the local art taste classification system.

---

## 1. System overview and data flow

The project is a local, privacy-first pipeline designed to learn and automate the user's subjective taste profile on artworks in a target art library website.

```
+-------------------------------------------------------------------------------+
|                             Art library website                               |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |                        Tampermonkey userscript                        |   |
|   |   - Hotkeys: Left Arrow (Dislike), Right Arrow (Like), S (Super), A (Auto)|
|   |   - Extraction: Primary image (DOM -> Canvas -> /api/capture)         |   |
|   |   - UI badge: Prediction score & confidence in Supervised mode        |   |
|   |   - Action dispatcher: Triggers Left Arrow (0) or Right Arrow (1)     |   |
|   +-----------------------------------+-----------------------------------+   |
+---------------------------------------|---------------------------------------+
                                        | JSON (GM_xmlhttpRequest)
                                        v
+-------------------------------------------------------------------------------+
|                         Local FastAPI backend (:8000)                         |
|                                                                               |
|   +--------------------+     +---------------------+     +----------------+   |
|   |    API endpoints   |     |    CLIP extractor   |     | Logistic Reg.  |   |
|   |  - /api/record     |---->|  - clip-ViT-L-14    |---->|  - Grid search |   |
|   |  - /api/predict    |     |  - 768-dim float32  |     |  - C & weights |   |
|   |  - /api/capture    |     |  - L2 normalized    |     |  - Hybrid F2   |   |
|   |  - /api/train      |     +---------------------+     +----------------+   |
|   |  - /api/threshold  |                                         |            |
|   |  - /api/samples    |                                         v            |
|   |  - /api/review     |                              +-------------------+   |
|   |  - /api/metrics    |                              | Structured JSON   |   |
|   |  - /api/benchmark  |                              | - data/model.json |   |
|   |  - /api/logs       |                              +-------------------+   |
|   +--------------------+                                         |            |
|             |                                                    v            |
|             v                                         +-------------------+   |
|   +---------------------------------------------+     | SQLite database   |   |
|   | Data quality tools                          |     | - data/dataset.db |   |
|   |  - Near-duplicate consolidation (sim >= 0.98|---->| - WAL mode        |   |
|   |  - Outlier detection (distance > 2σ)        |     | - BLOB embeddings |   |
|   |  - Session drift monitoring ([0.05, 0.10])  |     +-------------------+   |
|   +---------------------------------------------+               |             |
|                                                                 v             |
|   +-----------------------------------------------------------------------+   |
|   | Developer web dashboard (HTML, Vanilla CSS, JavaScript)               |   |
|   |  - Review queue: Grid for unreviewed automated decisions              |   |
|   |  - Samples: Dataset inspector with outlier filter & batch deletion    |   |
|   |  - Embedding space: Interactive 2D scatter plot (PCA and t-SNE)       |   |
|   |  - Model & metrics: PR curves, threshold slider, retrain, benchmark   |   |
|   |  - Console: Live activity log stream                                  |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
```

---

## 2. Storage architecture and file layout

### Repository layout
```
ClassificationProject/
├── agent-docs/
│   ├── design-specification.md    # Design and API specifications
│   ├── glossary.md                # Canonical terminology and definitions
│   └── system-architecture.md     # Component architecture and system design
├── backend/
│   ├── app.py                     # FastAPI application and route handlers
│   ├── database.py                # SQLite database management and queries
│   ├── model.py                   # CLIP feature extraction and classification
│   └── static/                    # Developer dashboard frontend assets
│       ├── app.js
│       ├── index.html
│       └── style.css
├── data/
│   ├── cache/                     # Benchmark results and cached computations
│   ├── dataset.db                 # SQLite dataset database (WAL mode)
│   ├── images/                    # Stored primary artwork images ({image_hash}.jpg)
│   └── model.json                 # Serialized model weights and evaluation metrics
├── tasks/
│   ├── benchmark_backbones.py     # Vision backbone benchmark engine
│   └── todo.md                    # Task checklist
├── tests/
│   ├── test_api.py                # REST API endpoint tests
│   ├── test_dashboard.py          # Dashboard static asset tests
│   ├── test_database.py           # SQLite database CRUD tests
│   ├── test_e2e.py                # End-to-end workflow tests
│   ├── test_model.py              # ML pipeline and threshold calibration tests
│   ├── test_phase8.py             # Outlier and near-duplicate tests
│   ├── test_phase10.py            # JSON serialization and benchmark tests
│   └── test_scatter.py            # Embedding space projection tests
├── userscript/
│   └── taste_collector.user.js    # Tampermonkey browser script
├── requirements.txt
└── README.md
```

### SQLite schema (`data/dataset.db`)

```sql
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_hash TEXT UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    label INTEGER,               -- 1 = Like, 0 = Dislike, NULL = Unlabeled
    prediction_score REAL,       -- P(Like) from 0.0 to 1.0
    mode TEXT NOT NULL,          -- 'manual', 'supervised', 'auto'
    reviewed INTEGER DEFAULT 0,  -- 1 = Confirmed, 0 = Pending review
    embedding BLOB NOT NULL,     -- 768 float32 values (3072 bytes)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_samples_reviewed ON samples(reviewed);
CREATE INDEX IF NOT EXISTS idx_samples_mode ON samples(mode);
CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label);
CREATE INDEX IF NOT EXISTS idx_samples_image_hash ON samples(image_hash);
```

---

## 3. Backend API contract

### `POST /api/record`
Ingests a sample with a base64 encoded image, label, and mode. Extracts and stores the 768-dimensional vision embedding. Automatically checks for near-duplicates before creating new rows.
* **Payload:**
  ```json
  {
    "image_base64": "data:image/jpeg;base64,...",
    "label": 1,
    "mode": "manual",
    "prediction_score": 0.84,
    "reviewed": 1,
    "image_set_count": 1,
    "negative_sample_rate": 0.05
  }
  ```
* **Response (New Sample):**
  ```json
  {
    "status": "success",
    "id": 142,
    "image_hash": "a8f3b1c9...",
    "label": 1,
    "reviewed": 1
  }
  ```
* **Response (Consolidated Near-Duplicate):**
  ```json
  {
    "status": "consolidated",
    "id": 89,
    "duplicate_of": 89,
    "similarity": 0.9872,
    "image_hash": "b2c7e1...",
    "label": 1,
    "reviewed": 1
  }
  ```

### `POST /api/predict`
Calculates the vision embedding for a base64 image and computes the prediction score and binary decision.
* **Payload:**
  ```json
  {
    "image_base64": "data:image/jpeg;base64,...",
    "threshold": 0.35
  }
  ```
* **Response:**
  ```json
  {
    "prediction_score": 0.84,
    "decision": 1,
    "threshold": 0.35,
    "model_loaded": true
  }
  ```
* **Response (Cold Start):**
  ```json
  {
    "prediction_score": null,
    "decision": null,
    "threshold": null,
    "model_loaded": false,
    "message": "Model not trained yet."
  }
  ```

### `POST /api/capture`
Captures an OS desktop screen region using `mss` when the browser DOM cannot access raw image bytes directly.
* **Payload:**
  ```json
  {
    "x": 100,
    "y": 150,
    "width": 600,
    "height": 800
  }
  ```
* **Response:**
  ```json
  {
    "status": "captured",
    "image_base64": "data:image/jpeg;base64,..."
  }
  ```

### `POST /api/train`
Executes Stratified 5-Fold Cross-Validation, hyperparameter grid search, and decision threshold calibration on all labeled samples. Fits the final Logistic Regression model and saves weights to `data/model.json`.
* **Payload (Optional overrides):**
  ```json
  {
    "target_recall": null,
    "threshold": null,
    "min_recall_floor": 0.70,
    "holdout_ratio": 0.15,
    "baseline_prompt_text": null,
    "baseline_image_base64": null,
    "reset_baseline_to_default": false
  }
  ```
* **Response:**
  ```json
  {
    "status": "trained",
    "sample_count": 250,
    "positive_count": 22,
    "negative_count": 228,
    "metrics": {
      "pr_auc": 0.884,
      "average_precision": 0.881,
      "recall": 0.909,
      "precision": 0.625,
      "f2_score": 0.833,
      "decision_threshold": 0.32,
      "evaluation_type": "stratified_cv",
      "folds": 5,
      "best_params": { "C": 1.0, "class_weight": "balanced" },
      "baselines": {
        "random_guess": 0.088,
        "positive_centroid": 0.762,
        "zero_shot": 0.612,
        "reference_type": "text",
        "reference_source": "goth aesthetic alternative indie girl style"
      },
      "confusion_matrix": {
        "true_positives": 20,
        "false_positives": 12,
        "true_negatives": 216,
        "false_negatives": 2
      }
    }
  }
  ```

### `POST /api/threshold`
Updates the active decision threshold in `data/model.json` and recalculates confusion matrix metrics without full retraining.
* **Payload:**
  ```json
  {
    "threshold": 0.40
  }
  ```
* **Response:**
  ```json
  {
    "success": true,
    "decision_threshold": 0.40,
    "metrics": { "recall": 0.864, "precision": 0.679, "f2_score": 0.819 }
  }
  ```

### `GET /api/samples`
Queries stored samples with pagination, mode filtering, review status filtering, label filtering, and taste consistency outlier detection.
* **Query Parameters:** `mode`, `reviewed`, `label`, `quality` (`inconsistent_likes` or `inconsistent_dislikes`), `outliers_only`, `limit`, `offset`.
* **Response:**
  ```json
  [
    {
      "id": 142,
      "image_hash": "a8f3b1...",
      "file_path": "data/images/a8f3b1.jpg",
      "image_base64": "data:image/jpeg;base64,...",
      "label": 1,
      "prediction_score": 0.84,
      "mode": "auto",
      "reviewed": 0,
      "is_outlier": false,
      "outlier_type": null,
      "outlier_reason": null,
      "centroid_distance": 0.42,
      "created_at": "2026-09-03 10:30:00"
    }
  ]
  ```

### `POST /api/review`
Bulk updates labels and sets review status for samples in the review queue.
* **Payload:**
  ```json
  {
    "updates": [
      { "id": 142, "label": 1, "reviewed": 1 },
      { "id": 143, "label": 0, "reviewed": 1 }
    ]
  }
  ```
* **Response:**
  ```json
  {
    "status": "success",
    "updated_count": 2
  }
  ```

### `DELETE /api/samples/{sample_id}`
Deletes a single sample from the SQLite database and deletes the corresponding image file from disk.
* **Response:**
  ```json
  {
    "status": "success",
    "deleted_id": 142
  }
  ```

### `POST /api/samples/batch-delete`
Deletes multiple samples from the database and disk in a single transaction.
* **Payload:**
  ```json
  {
    "ids": [142, 143, 144]
  }
  ```
* **Response:**
  ```json
  {
    "status": "success",
    "deleted_count": 3,
    "deleted_ids": [142, 143, 144]
  }
  ```

### `GET /api/metrics`
Returns dataset statistics, class balance, and active model metrics.
* **Response:**
  ```json
  {
    "statistics": {
      "total_samples": 250,
      "positive_count": 22,
      "negative_count": 228,
      "unlabeled_count": 0,
      "pending_auto_review_count": 5,
      "positive_ratio": 0.088
    },
    "model_status": {
      "model_loaded": true,
      "decision_threshold": 0.32,
      "positive_count": 22,
      "negative_count": 228,
      "metrics": { ... }
    }
  }
  ```

### `GET /api/embeddings/scatter`
Extracts 2D coordinates for dataset sample vision embeddings using PCA or t-SNE dimensionality reduction.
* **Query Parameters:** `method` (`pca` or `tsne`).
* **Response:**
  ```json
  {
    "status": "success",
    "total_points": 250,
    "method": "pca",
    "variance_ratio": [0.1842, 0.0915],
    "points": [
      {
        "id": 142,
        "image_hash": "a8f3b1...",
        "image_url": "/images/a8f3b1.jpg",
        "label": 1,
        "prediction_score": 0.84,
        "mode": "auto",
        "reviewed": 0,
        "x": 1.4521,
        "y": -0.8924,
        "created_at": "2026-09-03 10:30:00"
      }
    ]
  }
  ```

### `POST /api/benchmark`
Launches an asynchronous vision backbone benchmark evaluation in the background.
* **Payload:**
  ```json
  {
    "models": ["clip-ViT-L-14", "google/siglip-so400m-patch14-384", "facebook/dinov2-base"],
    "limit": null,
    "batch_size": 32,
    "force_extract": false
  }
  ```
* **Response:**
  ```json
  {
    "status": "started",
    "message": "Vision backbone benchmark started in background."
  }
  ```

### `GET /api/benchmark`
Queries the live progress and results of the vision backbone benchmark evaluation.
* **Response:**
  ```json
  {
    "status": "completed",
    "percent": 100.0,
    "sample_count": 250,
    "total_duration_seconds": 18.4,
    "results": [
      {
        "id": "clip-ViT-L-14",
        "name": "CLIP ViT-L/14 (Baseline)",
        "dim": 768,
        "pr_auc": 0.884,
        "f2_score": 0.833,
        "recall": 0.909,
        "precision": 0.625
      }
    ]
  }
  ```

### `GET /api/logs` and `POST /api/logs/clear`
Retrieves or clears recent in-memory activity logs.
* `GET /api/logs?limit=100&level=INFO&mode=manual`: Returns circular buffer entries in reverse chronological order.
* `POST /api/logs/clear`: Empties the activity log buffer.

---

## 4. Machine learning pipeline details

1. **Feature extraction:**
   * Model: `clip-ViT-L-14` (768 dimensions) via `sentence-transformers`.
   * Normalization: Unit $L_2$ norm applied to each vision embedding.
2. **Hyperparameter grid search:**
   * $C \in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]$
   * Class weight configurations: `balanced`, `unweighted`, `balanced_1.5x`, `balanced_2.0x`.
   * Evaluated through Stratified 5-Fold Cross-Validation.
3. **Decision threshold calibration:**
   * Hybrid $F_2$ search with a minimum recall floor (default 0.70).
   * Supports target recall constraints or manual threshold setting.
4. **Outlier and consistency analysis:**
   * Centroid distance calculation: $\text{distance} = 1.0 - \text{cosine\_sim}(v, \text{centroid})$.
   * Flags positive samples with distance $> \text{mean} + 2\sigma$ or out-of-fold score $< 0.20$.
   * Flags negative samples with distance $> \text{mean} + 2\sigma$ or score $\ge \theta$.
5. **Near-duplicate detection:**
   * Compares incoming embedding dot product against stored normalized embeddings.
   * If $\ge 0.98$, updates existing record rather than inserting a duplicate.
6. **Model serialization:**
   * Saved to `data/model.json` with format version `1.0`. Contains weight array, intercept scalar, threshold, metrics dictionary, and baseline metadata.

---

## 5. Tampermonkey userscript design

### Controls
* `ArrowLeft`: Dislike (Manual mode)
* `ArrowRight`: Like (Manual mode)
* `S`: Toggle Supervised mode
* `A`: Toggle Full auto mode
* `Y` / `N`: Confirm or flip prediction in Supervised mode

### Image extraction waterfall
1. Primary DOM `<img>` tag in the active artwork container.
2. CSS `background-image` style parsing if rendered inside a background element.
3. Canvas export (`canvas.toDataURL('image/jpeg')`) if rendered inside an HTML5 `<canvas>`.
4. Desktop capture fallback via `POST /api/capture`.

### Automated actions
Dispatches synthetic keyboard events (`keydown`) with `ArrowLeft` or `ArrowRight` to advance the library website upon manual or automated rating decisions.

---

## 6. Developer web dashboard

Single-page web interface served at `http://localhost:8000`:

* **Review queue tab:** Displays unreviewed automated decisions (`reviewed = 0`). Supports keyboard navigation (`1` for Like, `0` for Dislike), click-to-toggle badges, and bulk confirmation.
* **Samples tab (Dataset inspector):** Comprehensive list of stored samples with mode and label filters. Includes quality filter for inconsistent likes and dislikes (outliers), select all, and batch deletion.
* **Embedding space tab:** 2D interactive canvas for PCA and t-SNE projections. Supports drag pan, wheel zoom, point coloring by label or score, thumbnail hover cards, and sample inspector drawer.
* **Model & metrics tab:** Real-time out-of-fold PR curves, threshold slider, confusion matrix, baseline comparison table, one-click retrain, and backbone benchmark comparison table.
* **Console tab:** Real-time log view showing system events, predictions, duplicate consolidations, drift warnings, and training outputs.
