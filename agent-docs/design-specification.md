# Final Design Specification: Local Art Taste Classifier

This document is the definitive design and implementation specification for the local art taste classification system.

---

## 1. System Overview & Data Flow

The project is a local, privacy-first pipeline designed to learn and automate the user's subjective art taste on a target art library website.

```
+-------------------------------------------------------------------------------+
|                             Art Library Website                               |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   |                        Tampermonkey Userscript                        |   |
|   |   - Hotkeys: Left Arrow (Dislike), Right Arrow (Like), S (Super), A (Auto)|
|   |   - Extraction: Grabs Primary Image (DOM -> Canvas -> /api/capture)   |   |
|   |   - UI Badge: Displays prediction & confidence during Supervised Mode |   |
|   |   - Action Dispatcher: Triggers Left Arrow (0) or Right Arrow (1)     |   |
|   +-----------------------------------+-----------------------------------+   |
+---------------------------------------|---------------------------------------+
                                        | JSON (GM_xmlhttpRequest)
                                        v
+-------------------------------------------------------------------------------+
|                         Local FastAPI Backend (:8000)                         |
|                                                                               |
|   +--------------------+     +---------------------+     +----------------+   |
|   |    API Endpoints   |     |    CLIP Extractor   |     |  Logistic Reg. |   |
|   |  - /api/record     |---->|  - clip-ViT-L-14    |---->|  - Balanced    |   |
|   |  - /api/predict    |     |  - 768-dim L2 float |     |  - Threshold   |   |
|   |  - /api/capture    |     +---------------------+     +----------------+   |
|   |  - /api/train      |                                                      |
|   |  - /api/samples    |                                                      |
|   |  - /api/review     |------------------------+                             |
|   |  - /api/metrics    |                        |                             |
|   +--------------------+                        v                             |
|                                       +-------------------+                   |
|                                       | SQLite Database   |                   |
|                                       | - data/dataset.db |                   |
|                                       | - WAL Mode        |                   |
|                                       | - BLOB Embeddings |                   |
|                                       +-------------------+                   |
|                                                 |                             |
|   +---------------------------------------------v-------------------------+   |
|   | Developer Web Dashboard (HTML / Vanilla CSS / JavaScript)             |   |
|   |  - Review Queue: Visual grid (Mouse clicks & 1/0 keys for review)     |   |
|   |  - Model Metrics: PR-AUC, Confusion Matrix, Dynamic Threshold Slider  |   |
|   |  - Controls: One-click model retraining and threshold persistence     |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
```

---

## 2. Storage Architecture & Database Schema

### File Layout
```
ClassificationProject/
├── agent-docs/
│   ├── glossary.md
│   ├── storage-options.md
│   ├── browser-interaction-options.md
│   ├── system-architecture.md
│   └── design-specification.md
├── backend/
│   ├── app.py                  # FastAPI application entry point
│   ├── database.py             # SQLite connection, initialization, queries
│   ├── model.py                # CLIP feature extractor & Logistic Regression
│   └── static/                 # Developer dashboard assets (HTML, CSS, JS)
│       ├── index.html
│       ├── app.js
│       └── style.css
├── userscript/
│   └── taste_collector.user.js # Tampermonkey script
├── data/
│   ├── dataset.db              # SQLite database (WAL mode)
│   ├── model.pkl               # Saved scikit-learn pipeline & threshold
│   └── images/                 # Stored raw image files ({image_hash}.jpg)
├── requirements.txt
└── README.md
```

### SQLite Schema (`data/dataset.db`)

```sql
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_hash TEXT UNIQUE NOT NULL,
    file_path TEXT NOT NULL,
    label INTEGER,               -- 1 = Like, 0 = Dislike, NULL = Unlabeled
    prediction_score REAL,       -- P(Like) from 0.0 to 1.0
    mode TEXT NOT NULL,          -- 'manual', 'supervised', 'auto'
    reviewed INTEGER DEFAULT 0,  -- 1 = Confirmed, 0 = Pending Review
    embedding BLOB NOT NULL,     -- 768 float32 values (3072 bytes)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_samples_reviewed ON samples(reviewed);
CREATE INDEX IF NOT EXISTS idx_samples_mode ON samples(mode);
CREATE INDEX IF NOT EXISTS idx_samples_label ON samples(label);
```

---

## 3. Backend API Contract

All image-bearing endpoints standardize on JSON payloads with Base64 data URIs.

### `POST /api/record`
Ingests a sample with base64 image bytes, label, and mode. Computes and saves the CLIP embedding automatically.
* **Payload (JSON):**
  ```json
  {
    "image_base64": "data:image/jpeg;base64,...",
    "label": 1,
    "mode": "manual",
    "prediction_score": 0.842,
    "reviewed": 1
  }
  ```
* **Response:**
  ```json
  {
    "status": "success",
    "id": 142,
    "image_hash": "a8f3b1c9...",
    "label": 1
  }
  ```

### `POST /api/predict`
Calculates CLIP embedding and predicts like/dislike for an incoming base64 image.
* **Payload (JSON):**
  ```json
  {
    "image_base64": "data:image/jpeg;base64,..."
  }
  ```
* **Response:**
  ```json
  {
    "prediction_score": 0.842,
    "decision": 1,
    "threshold": 0.35,
    "model_loaded": true
  }
  ```
* **Response (Cold Start - Model Not Trained Yet):**
  ```json
  {
    "prediction_score": null,
    "decision": null,
    "threshold": null,
    "model_loaded": false,
    "message": "Model not trained yet. Gather samples in Manual Mode first."
  }
  ```

### `POST /api/capture`
Captures an OS screen region using Python `mss`/`Pillow` when the browser DOM cannot access raw image bytes.
* **Payload (JSON):**
  ```json
  {
    "x": 200,
    "y": 150,
    "width": 800,
    "height": 600
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
Loads all labeled samples from SQLite, trains `LogisticRegression(class_weight='balanced')`, evaluates performance via cross-validation, and saves the updated model. Requires at least one positive class sample and one negative class sample.
* **Response (Success):**
  ```json
  {
    "status": "trained",
    "sample_count": 850,
    "like_count": 68,
    "dislike_count": 782,
    "metrics": {
      "pr_auc": 0.891,
      "recall": 0.941,
      "precision": 0.625,
      "f2_score": 0.852,
      "suggested_threshold": 0.35
    }
  }
  ```
* **Response (Insufficient Data):**
  ```json
  {
    "status": "insufficient_data",
    "message": "Need at least 1 Like and 1 Dislike sample to train.",
    "sample_count": 12,
    "like_count": 0,
    "dislike_count": 12
  }
  ```

### `GET /api/samples`
Queries samples for the dashboard review queue and dataset inspector, embedding base64 images directly into the response records.
* **Query Parameters:** `mode`, `reviewed`, `label`, `limit`, `offset`.
* **Response:**
  ```json
  [
    {
      "id": 142,
      "image_hash": "a8f3b1...",
      "image_base64": "data:image/jpeg;base64,...",
      "label": 1,
      "prediction_score": 0.842,
      "mode": "auto",
      "reviewed": 0,
      "created_at": "2026-08-29T19:40:00"
    }
  ]
  ```

### `POST /api/review`
Bulk updates labels and review flags for items in the Review Queue.
* **Payload:**
  ```json
  {
    "updates": [
      { "id": 142, "label": 1, "reviewed": 1 },
      { "id": 143, "label": 0, "reviewed": 1 }
    ]
  }
  ```

### `GET /api/metrics`
Returns dataset statistics, class balance, threshold curves, and confusion matrix data.

---

## 4. Machine Learning Pipeline

1. **Feature Extraction:**
   * Model: `clip-ViT-L-14` (768 dimensions).
   * Device: Auto-selects CUDA if available, otherwise CPU.
   * Normalization: $L_2$ unit sphere normalization on all extracted vectors.
2. **Classification Model:**
   * `sklearn.linear_model.LogisticRegression(class_weight='balanced', C=1.0, max_iter=1000)`
   * Resolves the 5–10% positive class skew by scaling inverse class frequencies.
3. **Threshold Calibration:**
   * Outputs probability $P(\text{like} \mid x) = \frac{1}{1 + e^{-(w \cdot x + b)}}$.
   * Predicts Like ($1$) if $P \ge \theta$, otherwise Dislike ($0$).
   * Default $\theta = 0.35$, adjustable through the dashboard.

---

## 5. Tampermonkey Userscript Design

### Hotkeys & Controls
* `ArrowLeft`: Dislike (Manual Mode)
* `ArrowRight`: Like (Manual Mode)
* `S`: Toggle Supervised Mode
* `A`: Toggle Full Auto Mode
* `Y` / `N`: Confirm or Override prediction in Supervised Mode

### Image Extraction Waterfall
1. **DOM Extraction:** Finds the primary `<img>` tag in the active artwork container. Fetches blob and converts to base64.
2. **CSS Background Fallback:** Reads computed `background-image` CSS URL if the artwork renders in a styled `<div>`.
3. **Canvas Fallback:** Calls `canvas.toDataURL('image/jpeg')` if rendered inside an HTML5 `<canvas>`.
4. **Absolute Fallback:** Sends screen coordinates of the active container to `POST /api/capture` for local OS screenshot capture.

### Action Dispatch Logic
* Dispatches synthetic keyboard events to `window` and `document.body`:
  ```javascript
  function triggerDislike() {
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', code: 'ArrowLeft', keyCode: 37, bubbles: true }));
  }
  function triggerLike() {
      document.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', code: 'ArrowRight', keyCode: 39, bubbles: true }));
  }
  ```

---

## 6. Developer Web Dashboard

Accessible at `http://localhost:8000`:
* **Review Queue View:** Responsive thumbnail grid showing images processed in Full Auto mode (`reviewed = 0`).
  * Card selection via mouse click.
  * Press `1` on selected/hovered card to mark as Like, `0` to mark as Dislike.
  * Direct mouse click on label badge toggles between Like and Dislike.
  * "Mark All Visible as Reviewed" button.
* **Model Control & Tuning:**
  * Interactive slider for Decision Threshold ($\theta$) showing estimated Precision, Recall, and Confusion Matrix in real time.
  * "Retrain Model" button that triggers background cross-validation and updates metrics.
* **Dataset Stats:** Total count, positive ratio, and ingestion history chart.
