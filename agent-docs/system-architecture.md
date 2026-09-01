# System Architecture Specification: Art Taste Classifier

This document defines the complete architecture for the local binary art taste classification system.

---

## 1. System Overview

The system consists of two primary components:
1. **Local Backend Server (FastAPI + SQLite + PyTorch/CLIP + scikit-learn):**
   * Manages dataset storage, image processing, embedding extraction, model training, and prediction serving.
   * Serves an interactive developer dashboard for reviewing automated decisions, tuning classification thresholds, and monitoring evaluation metrics.
2. **Browser Layer (Tampermonkey Userscript):**
   * Injects into the art library website.
   * Extracts the primary image from each artwork set.
   * Supports three operating modes: **Manual Recording**, **Supervised Auto**, and **Full Auto**.
   * Dispatches native keyboard events (Left Arrow for Dislike, Right Arrow for Like) to trigger website actions.

```
+--------------------------------------------------------------------+
|                       Art Library Website                          |
|  +--------------------------------------------------------------+  |
|  |                 Tampermonkey Userscript                      |  |
|  |  - Hotkey Handler (Left Arrow / Right Arrow / S / A)         |  |
|  |  - Image Extractor (DOM -> Canvas -> /api/capture)           |  |
|  |  - UI Overlay (Supervised predictions & threshold preview)   |  |
|  |  - Keyboard Simulator (Left/Right Arrow triggers)            |  |
|  +------------------------------+-------------------------------+  |
+---------------------------------|----------------------------------+
                                  | HTTP (GM_xmlhttpRequest JSON)
                                  v
+--------------------------------------------------------------------+
|                    Local Backend (FastAPI :8000)                   |
|                                                                    |
|  +--------------------------------------------------------------+  |
|  | API Endpoints                                                |  |
|  |  - POST /api/record   (Saves sample, image, embedding)       |  |
|  |  - POST /api/predict  (Computes embedding & returns score)   |  |
|  |  - POST /api/capture  (Desktop screenshot fallback)          |  |
|  |  - POST /api/train    (Fits Logistic Regression & evaluates) |  |
|  |  - GET  /api/samples  (Queries dataset with embedded base64) |  |
|  |  - POST /api/review   (Bulk update labels & review status)   |  |
|  |  - GET  /api/metrics  (PR-AUC, Recall@Threshold, Confusion)  |  |
|  +------------------------------+-------------------------------+  |
|                                 |                                  |
|  +---------------------+  +-----+---------------+  +------------+  |
|  | Pretrained CLIP     |  | Logistic Regression |  | SQLite DB  |  |
|  | (clip-ViT-L-14)     |  | (Balanced Weights)  |  | (WAL Mode) |  |
|  +---------------------+  +---------------------+  +------------+  |
|                                                          |         |
|  +---------------------------------------------------+   |         |
|  | Developer Web Dashboard (HTML / Vanilla CSS / JS) |   |         |
|  |  - Visual grid for reviewing Full Auto decisions  |<--+         |
|  |  - Real-time PR curve, Recall slider & Retrain button           |
|  +-----------------------------------------------------------------+
+--------------------------------------------------------------------+
```

---

## 2. Storage & Database Schema

### Database Configuration
* Engine: SQLite 3 with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`).
* Location: `data/dataset.db`
* Raw Images: `data/images/{image_hash}.jpg`

### Table: `samples`
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Unique identifier |
| `image_hash` | `TEXT UNIQUE NOT NULL` | SHA-256 hash of image bytes to prevent duplicates |
| `file_path` | `TEXT NOT NULL` | Relative path to local stored image |
| `label` | `INTEGER` | `1` (Like), `0` (Dislike), `NULL` (Pending Review) |
| `prediction_score` | `REAL` | Model predicted probability $P(\text{like})$ (0.0 to 1.0) |
| `mode` | `TEXT NOT NULL` | `'manual'`, `'supervised'`, `'auto'` |
| `reviewed` | `INTEGER DEFAULT 0` | `1` = Confirmed by human, `0` = Unreviewed |
| `embedding` | `BLOB NOT NULL` | 768 `float32` vector bytes (3072 bytes) |
| `created_at` | `TIMESTAMP DEFAULT CURRENT_TIMESTAMP` | Ingestion timestamp |

---

## 3. Machine Learning & Training Pipeline

### Embedding Model
* **Model:** OpenAI CLIP (`clip-ViT-L-14`, 768 dimensions) via `sentence-transformers` or Hugging Face `transformers`.
* Runs on CUDA if available, otherwise CPU.
* Normalizes embeddings ($L_2$ norm = 1.0) for consistent cosine similarity.

### Classifier
* **Algorithm:** `LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)` from `scikit-learn`.
* **Imbalance Handling:** Balanced class weights scale the penalty for the minority positive class (5-10% likes) proportionally to class frequency.
* **Decision Threshold:** Configurable (default 0.35) to prioritize high Recall ($>90\%$) while keeping Precision acceptable.

### Evaluation Metrics
* **Precision-Recall AUC (PR-AUC):** Informative metric for skewed binary classes.
* **F-beta Score ($F_2$):** Places twice the weight on Recall over Precision.
* **Confusion Matrix:** True Positives, False Positives, True Negatives, False Negatives across candidate thresholds.

---

## 4. Operational Modes & Browser Integration

### Mode 1: Manual Recording
1. User navigates art library.
2. User presses Left Arrow (Dislike) or Right Arrow (Like).
3. Script locates the first image in the set, extracts base64 image data, and posts payload to `/api/record` with `label` (0 or 1) and `mode='manual'`.
4. Script dispatches keyboard event (Left or Right arrow) to advance the library website.

### Mode 2: Supervised Auto
1. Script grabs the current first image and sends it to `/api/predict`.
2. Backend computes CLIP embedding, runs Logistic Regression, and returns `prediction_score` (e.g. `0.78`).
3. Script displays a floating on-screen badge showing the predicted decision and confidence.
4. User presses hotkey to confirm (`Y`) or override (`N`).
5. Script posts confirmed label to `/api/record` and simulates the corresponding arrow key.

### Mode 3: Full Auto
1. Script grabs the current first image and sends it to `/api/predict`.
2. Backend returns probability score. Script compares against active threshold.
3. Script dispatches the appropriate arrow key (Left or Right) to the website.
4. Script posts record to `/api/record` with `mode='auto'` and `reviewed=0`.
5. User reviews and adjusts auto-labeled records in bulk via the dashboard later.

---

## 5. Web Developer Dashboard

A clean local web interface served directly by FastAPI at `http://localhost:8000`:
* **Overview Tab:** Summary cards (Total samples, Total Likes, Like Ratio, Model Status, Current PR-AUC).
* **Review Queue Tab:** Grid view of unreviewed full-auto predictions. Review with mouse clicks to toggle badges or `1`/`0` keys on selected cards.
* **Samples Tab:** Dataset inspector to browse, filter, and batch delete samples.
* **Embedding Space Tab:** Interactive 2D scatter plot visualizer projecting the vision embedding space using PCA or t-SNE, with color coding by label or score and hover image previews.
* **Model & Metrics Tab:** PR curve visualization, threshold slider showing dynamic Recall/Precision tradeoff, confusion matrix, and a "Retrain Model" button.
* **Console Tab:** Real-time log stream of system and rating events.

---

## 6. Embedding Space Visualizer & Dimensionality Reduction

The developer dashboard includes an interactive 2D scatter plot to explore the 768-dimensional vision embedding space.

```
+-------------------------------------------------------------------------------+
|                      Embedding Space Visualizer Architecture                  |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   | SQLite DB (`samples` table)                                            |   |
|   |  - 768-dim L2-normalized float32 BLOBs + Sample Metadata              |   |
|   +-----------------------------------+-----------------------------------+   |
|                                       | load_embedding_scatter_data()         |
|                                       v                                       |
|   +-----------------------------------------------------------------------+   |
|   | Backend API: GET /api/embeddings/scatter?method={pca|tsne}            |   |
|   |                                                                       |   |
|   |   +-----------------------------------+---------------------------+   |   |
|   |   | PCA (Linear Projection)           | t-SNE (Non-Linear)        |   |   |
|   |   | - sklearn.decomposition.PCA       | - sklearn.manifold.TSNE   |   |   |
|   |   | - Maximize global variance        | - Preserves local clusters|   |   |
|   |   | - Returns explained variance ratio| - Iterative probability fit|  |   |
|   |   +-----------------------------------+---------------------------+   |   |
|   +-----------------------------------+-----------------------------------+   |
|                                       | JSON [{x, y, label, score, ...}]      |
|                                       v                                       |
|   +-----------------------------------------------------------------------+   |
|   | HTML5 Canvas Visualizer (Frontend)                                    |   |
|   |  - HiDPI hardware-accelerated rendering                               |   |
|   |  - Interactive pan (drag) and cursor-centered zoom (mouse wheel)       |   |
|   |  - Point coloring: By Label (1=Emerald, 0=Rose) or Prediction Score   |   |
|   |  - Hover Card: Artwork thumbnail preview loaded via /images/{hash}.jpg|   |
|   |  - Side Inspector: Sample details with one-click relabel & delete     |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
```

### Dimensionality Reduction Algorithms

#### 1. PCA (Principal Component Analysis)
* **Mathematical approach**: Linear orthogonal projection. Finds the two eigenvectors of the sample covariance matrix corresponding to the largest eigenvalues.
* **Preservation**: Preserves **global variance and broad distance relationships**. Distant clusters in 768D stay distant in 2D.
* **Deterministic**: Fixed computation with zero randomness.
* **Explained variance**: The endpoint returns `variance_ratio = [PC1, PC2]`, indicating the proportion of total information preserved in the 2D view.
* **Use case**: Assessing overall dataset balance and viewing the feature space through the lens of linear classifiers like Logistic Regression.

#### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)
* **Mathematical approach**: Non-linear manifold learning. Converts pairwise Euclidean distances into conditional probabilities in high-dimensional space and Student-t distributions in 2D space, minimizing the Kullback-Leibler (KL) divergence via gradient descent.
* **Preservation**: Preserves **local neighborhoods**. Artworks with similar visual semantics, color palettes, or artistic styles cluster tightly together.
* **Dynamic perplexity**: Automatically tuned based on sample count: $\text{perplexity} = \max(1, \min(30, \lfloor(N - 1) / 3\rfloor))$.
* **Use case**: Discovering fine-grained aesthetic sub-clusters (e.g. watercolor, anime portraits, dark landscapes) within your liked artworks.

