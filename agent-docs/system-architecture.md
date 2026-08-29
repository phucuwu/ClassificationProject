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
* **Model & Metrics Tab:** PR curve visualization, threshold slider showing dynamic Recall/Precision tradeoff, confusion matrix, and a "Retrain Model" button.
