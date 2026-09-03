# System architecture specification: Art taste classifier

This document defines the architecture for the local binary art taste classification system.

---

## 1. System overview

The system consists of two primary components:

1. **Local backend server (FastAPI, SQLite, PyTorch, CLIP, scikit-learn):**
   * Manages dataset storage, image processing, embedding extraction, model training, and prediction serving.
   * Serves an interactive developer dashboard for reviewing automated decisions, tuning classification thresholds, exploring embedding projections, and monitoring evaluation metrics.
2. **Browser layer (Tampermonkey userscript):**
   * Injects into the target art library website.
   * Extracts the primary image from each artwork set.
   * Supports three operating modes: manual mode, supervised mode, and full auto mode.
   * Dispatches keyboard events (Left Arrow for Dislike, Right Arrow for Like) to trigger website rating actions.

```
+--------------------------------------------------------------------+
|                         Art library website                        |
|  +--------------------------------------------------------------+  |
|  |                    Tampermonkey userscript                   |  |
|  |  - Hotkey handler (Left Arrow, Right Arrow, S, A)            |  |
|  |  - Image extractor (DOM -> Canvas -> /api/capture)           |  |
|  |  - UI overlay (Supervised predictions & threshold preview)   |  |
|  |  - Keyboard simulator (Left/Right Arrow triggers)            |  |
|  +------------------------------+-------------------------------+  |
+---------------------------------|----------------------------------+
                                  | HTTP (GM_xmlhttpRequest JSON)
                                  v
+--------------------------------------------------------------------+
|                    Local backend (FastAPI :8000)                   |
|                                                                    |
|  +--------------------------------------------------------------+  |
|  | API endpoints                                                |  |
|  |  - POST   /api/record   (Saves sample, image, embedding)     |  |
|  |  - POST   /api/predict  (Computes embedding & returns score) |  |
|  |  - POST   /api/capture  (Desktop screenshot fallback)        |  |
|  |  - POST   /api/train    (Grid search CV & threshold tune)    |  |
|  |  - POST   /api/threshold(Updates active decision threshold)  |  |
|  |  - GET    /api/samples  (Queries dataset & outlier flags)   |  |
|  |  - POST   /api/review   (Bulk update labels & review status) |  |
|  |  - DELETE /api/samples/{id} (Single sample deletion)         |  |
|  |  - POST   /api/samples/batch-delete (Batch sample deletion)  |  |
|  |  - GET    /api/metrics  (PR-AUC, F2, confusion matrix)       |  |
|  |  - GET    /api/embeddings/scatter (PCA & t-SNE 2D coords)    |  |
|  |  - POST   /api/benchmark (Vision backbone benchmark run)     |  |
|  |  - GET    /api/benchmark (Benchmark status & results)        |  |
|  |  - GET    /api/logs     (Live activity console logs)         |  |
|  |  - POST   /api/logs/clear (Clear console logs)               |  |
|  +------------------------------+-------------------------------+  |
|                                 |                                  |
|  +---------------------+  +-----+---------------+  +------------+  |
|  | Pretrained CLIP     |  | Logistic Regression |  | SQLite DB  |  |
|  | (clip-ViT-L-14)     |  | (JSON serialization)|  | (WAL mode) |  |
|  | 768-dim float32     |  | data/model.json     |  | dataset.db |  |
|  +---------------------+  +---------------------+  +------------+  |
|                                                          |         |
|  +---------------------------------------------------+   |         |
|  | Developer web dashboard (HTML, Vanilla CSS, JS)   |   |         |
|  |  - Review queue: Grid for Full Auto decisions     |<--+         |
|  |  - Samples: Dataset inspector with outlier filter |             |
|  |  - Embedding space: Interactive 2D scatter plot   |             |
|  |  - Model & metrics: PR curve, threshold, retrain  |             |
|  |  - Console: Real-time event log stream            |             |
|  +-----------------------------------------------------------------+
+--------------------------------------------------------------------+
```

---

## 2. Storage and database schema

### File storage configuration
* Engine: SQLite 3 with Write-Ahead Logging (`PRAGMA journal_mode = WAL;`) and synchronous mode normal (`PRAGMA synchronous = NORMAL;`).
* Database location: `data/dataset.db`
* Model weights location: `data/model.json` (stores coefficients, intercept, decision threshold, metrics, and reference baselines). The system also auto-migrates legacy `data/model.pkl` artifacts to JSON on load.
* Stored artwork images: `data/images/{image_hash}.jpg`
* Cache and benchmark outputs: `data/cache/backbone_benchmark_results.json`

### Table: `samples`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Unique identifier |
| `image_hash` | `TEXT UNIQUE NOT NULL` | SHA-256 hash of image bytes to prevent duplicates |
| `file_path` | `TEXT NOT NULL` | Relative path to local stored image |
| `label` | `INTEGER` | `1` (Like), `0` (Dislike), `NULL` (Pending review) |
| `prediction_score` | `REAL` | Model predicted probability $P(\text{Like})$ from 0.0 to 1.0 |
| `mode` | `TEXT NOT NULL` | `'manual'`, `'supervised'`, `'auto'` |
| `reviewed` | `INTEGER DEFAULT 0` | `1` = Confirmed by human, `0` = Pending review |
| `embedding` | `BLOB NOT NULL` | 768 `float32` vector bytes (3072 bytes) |
| `created_at` | `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` | Ingestion timestamp |

### Indexes
* `idx_samples_reviewed` on `samples(reviewed)`
* `idx_samples_mode` on `samples(mode)`
* `idx_samples_label` on `samples(label)`
* `idx_samples_image_hash` on `samples(image_hash)`

---

## 3. Machine learning and data quality pipeline

### Vision embedding model
* Model: OpenAI CLIP (`clip-ViT-L-14`, 768 dimensions) loaded via `sentence-transformers`.
* Compute device: CUDA GPU when available, otherwise CPU.
* Normalization: Vectors are $L_2$-normalized to unit length for cosine distance operations.

### Classifier and hyperparameter search
* Model family: Balanced Logistic Regression using `scikit-learn`.
* Grid search: Evaluates candidate regularization parameters $C \in [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]$ across four class weight multipliers (`balanced`, `unweighted`, `balanced_1.5x`, `balanced_2.0x`).
* Validation: Stratified 5-Fold Cross-Validation on development data generates out-of-fold predictions. When positive samples are fewer than 5, fold count scales down to $\min(5, \text{pos}, \text{neg})$.
* Ranking metric: Best hyperparameter configuration is selected by out-of-fold PR-AUC, breaking ties with average precision.
* Final model fitting: Once parameters are selected, the final classifier fits on 100% of labeled samples.
* Serialization: Model coefficients, intercept scalar, active decision threshold, evaluation metrics, and reference baselines serialize into `data/model.json`. Vectorized inference uses direct NumPy dot products without requiring scikit-learn unpickling.

### Decision threshold calibration
* Default threshold: $\theta = 0.35$ (calibrated below 0.50 to favor recall on skewed taste distributions).
* Hybrid $F_2$ calibration: Searches thresholds from 0.05 to 0.95 in steps of 0.01 to maximize the $F_2$ score while enforcing a minimum recall floor (default 0.70).
* Target recall calibration: When specified, finds the highest threshold that meets or exceeds the requested recall target.
* Explicit manual override: Allows setting the active threshold directly via `POST /api/threshold` or `POST /api/train`.

### Generalization verification
* Holdout partition: Reserves the holdout fraction (default 15%) to verify out-of-fold generalization.
* Divergence alert: If holdout PR-AUC drops by more than 0.25 below out-of-fold PR-AUC, the system records a generalization warning in model metadata and logs.

### Reference baselines
Every training run computes three reference baselines:
1. **Random guess baseline:** Positive class ratio in development data.
2. **Positive centroid baseline:** Cosine similarity against the average vector of all positive class embeddings.
3. **Zero-shot prompt baseline:** Cosine similarity against a prompt text embedding (default: "goth aesthetic alternative indie girl style") or a custom user exemplar image embedding.

### Data quality and consistency tooling
* **Near-duplicate detection:** Calculates cosine similarity across existing vision embeddings on ingestion. If similarity is greater than or equal to 0.98, `POST /api/record` consolidates the incoming record into the existing sample row, updating labels without creating duplicate database rows or disk files.
* **Positive class outlier detection:** Computes the positive class centroid in feature space. Flags samples whose distance from the centroid exceeds 2 standard deviations, or whose out-of-fold prediction score is below 0.20.
* **Negative class outlier detection:** Flags negative samples whose distance from the negative centroid exceeds 2 standard deviations, or whose prediction score meets or exceeds the active decision threshold.
* **Session drift monitoring:** Analyzes a rolling window of recent samples (default 100). Emits an activity log warning if the rolling positive ratio drops below 5% or rises above 10%.

### Vision backbone benchmark suite
Located in `tasks/benchmark_backbones.py` and exposed through `POST /api/benchmark`:
* Evaluates alternative vision backbones against the baseline `clip-ViT-L-14` (768 dimensions), including Google SigLIP SO400M (`google/siglip-so400m-patch14-384`, 1152 dimensions) and Meta DINOv2 (`facebook/dinov2-base`, 768 dimensions).
* Runs stratified cross-validation on extracted embeddings and persists comparative PR-AUC, $F_2$, recall, and precision to `data/cache/backbone_benchmark_results.json`.

---

## 4. Operating modes and browser integration

### Mode 1: Manual mode
1. User browses the library website and evaluates an artwork.
2. User presses Left Arrow (Dislike) or Right Arrow (Like).
3. The userscript extracts the primary image from the active container as a base64 string.
4. The userscript sends payload to `POST /api/record` with `label` (0 or 1), `mode = 'manual'`, and `reviewed = 1`.
5. The userscript dispatches synthetic arrow key events to advance the library website.

### Mode 2: Supervised mode
1. The userscript extracts the primary image and posts to `POST /api/predict`.
2. The backend server computes the vision embedding, runs vectorized logistic inference, and returns `prediction_score` and `decision`.
3. The userscript renders a floating HUD badge displaying the predicted classification and confidence percentage.
4. The user presses `Y` to accept the prediction, or `N` to flip and override the label.
5. The userscript sends the confirmed label to `POST /api/record` with `mode = 'supervised'` and dispatches the corresponding arrow key.

### Mode 3: Full auto mode
1. The userscript extracts the primary image and requests a prediction from `POST /api/predict`.
2. The userscript evaluates the score against the active decision threshold.
3. The userscript simulates the appropriate arrow key action immediately.
4. The userscript posts the record to `POST /api/record` with `mode = 'auto'`. All positive decisions enter the review queue (`reviewed = 0`). Dislikes are sampled at a configurable audit rate (default 5% set to `reviewed = 0`, remainder marked `reviewed = 1`).
5. The user reviews and corrects automated decisions in bulk via the developer dashboard review queue.

---

## 5. Web developer dashboard

The backend server serves a single-page dashboard at `http://localhost:8000`:

* **Review queue tab:** Card grid showing automated decisions pending review (`reviewed = 0`). Users click cards or press `1`/`0` to toggle labels, and click "Mark All Visible as Reviewed" to confirm in bulk.
* **Samples tab (Dataset inspector):** Filterable grid of all stored samples. Supports filtering by mode, label, and outlier quality status (inconsistent likes or dislikes). Includes select all, deselect, and batch deletion controls.
* **Embedding space tab:** Hardware-accelerated 2D canvas visualization of the 768-dimensional vision embedding space using PCA or t-SNE projection. Includes cursor-centered zoom, pan, point color modes (by label or prediction score), thumbnail hover previews from `/images/{image_hash}.jpg`, and a slide-out sample inspection drawer.
* **Model & metrics tab:** Visualizes out-of-fold Precision-Recall curves, a dynamic threshold tuning slider with quick preset chips, a confusion matrix table, zero-shot reference baseline controls, retrain trigger, and vision backbone benchmark controls.
* **Console tab:** Real-time stream of system activity logs, manual ratings, predictions, near-duplicate consolidations, session drift alerts, and training events.

---

## 6. Embedding space visualizer and dimensionality reduction

```
+-------------------------------------------------------------------------------+
|                      Embedding space visualizer architecture                  |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   | SQLite DB (`samples` table)                                           |   |
|   |  - 768-dim L2-normalized float32 BLOBs + Sample metadata              |   |
|   +-----------------------------------+-----------------------------------+   |
|                                       | load_embedding_scatter_data()         |
|                                       v                                       |
|   +-----------------------------------------------------------------------+   |
|   | Backend API: GET /api/embeddings/scatter?method={pca|tsne}            |   |
|   |                                                                       |   |
|   |   +-----------------------------------+---------------------------+   |   |
|   |   | PCA (Linear projection)           | t-SNE (Non-linear)        |   |   |
|   |   | - sklearn.decomposition.PCA       | - sklearn.manifold.TSNE   |   |   |
|   |   | - Maximizes global variance       | - Preserves local clusters|   |   |
|   |   | - Returns explained variance ratio| - Iterative probability fit|  |   |
|   |   +-----------------------------------+---------------------------+   |   |
|   +-----------------------------------+-----------------------------------+   |
|                                       | JSON [{x, y, label, score, ...}]      |
|                                       v                                       |
|   +-----------------------------------------------------------------------+   |
|   | HTML5 canvas visualizer (Frontend)                                    |   |
|   |  - Hardware-accelerated 2D rendering                                  |   |
|   |  - Interactive pan (drag) and cursor-centered zoom (wheel)            |   |
|   |  - Point coloring: By label (1=Emerald, 0=Rose) or prediction score   |   |
|   |  - Hover card: Artwork thumbnail preview loaded via /images/{hash}.jpg|   |
|   |  - Side inspector: Sample details with one-click relabel & delete     |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
```

### Dimensionality reduction algorithms

#### 1. PCA (Principal Component Analysis)
* Approach: Linear orthogonal projection finding eigenvectors of the sample covariance matrix.
* Structure: Preserves global variance and broad distance relationships. Distant samples in 768D stay distant in 2D.
* Determinism: Deterministic calculation with zero randomness.
* Explained variance: Returns `variance_ratio = [PC1, PC2]`, indicating the proportion of variance captured in 2D.

#### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
* Approach: Non-linear manifold learning converting Euclidean distances into probabilities and minimizing Kullback-Leibler divergence.
* Structure: Preserves local neighborhoods. Artworks with similar visual characteristics cluster together.
* Perplexity: Scaled dynamically based on sample count: $\text{perplexity} = \max(1, \min(30, \lfloor(N - 1) / 3\rfloor))$.
