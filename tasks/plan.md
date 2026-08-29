# Implementation Plan: Local Art Taste Classifier

## Overview
Build a local, privacy-first system to learn and automate a subjective art taste profile using CLIP vision embeddings, balanced Logistic Regression, a FastAPI backend server with SQLite WAL storage, a Tampermonkey userscript, and a developer dashboard.

## Architecture Decisions
- **Data Gathering & Cold Start:** Starts with 0 samples. The user uses Manual Mode on the library to build the initial dataset. Endpoints (`/api/predict`, `/api/train`) handle cold start gracefully before the first model training.
- **Storage:** SQLite `data/dataset.db` with WAL mode storing 768-dim float32 BLOB embeddings; local images in `data/images/`.
- **Model:** `clip-ViT-L-14` for 768-dim normalized vision embeddings + `LogisticRegression(class_weight='balanced')` with tunable decision threshold.
- **Backend:** FastAPI on `http://localhost:8000` with JSON Base64 image payloads and `mss` screen capture fallback.
- **Browser Userscript:** Tampermonkey with DOM/Canvas extraction, ArrowLeft/ArrowRight action triggers, Manual mode, Supervised HUD mode, and Full auto mode.
- **Dashboard:** Web interface served directly by FastAPI for Review Queue batch labeling (`1`/`0` keys), PR-AUC visualization, and threshold tuning.

## Phase List & Checkpoints

### Phase 1: Environment and Dataset Database Foundation
* Task 1.1: `requirements.txt`
* Task 1.2: `backend/database.py` (WAL mode, samples schema, BLOB loader)
* **Checkpoint 1:** Standalone database test script verifying CRUD and NumPy matrix reconstruction.

### Phase 2: Vision Embedding and Machine Learning Engine
* Task 2.1: `backend/model.py` (CLIP `clip-ViT-L-14` extractor)
* Task 2.2: `backend/model.py` (Balanced Logistic Regression, PR-AUC, $F_2$, thresholding)
* **Checkpoint 2:** Standalone ML pipeline test verifying feature extraction and cross-validation fitting.

### Phase 3: Backend Server and API Endpoints
* Task 3.1: `backend/app.py` (`/api/record`, `/api/predict`, `/api/capture`)
* Task 3.2: `backend/app.py` (`/api/train`, `/api/samples`, `/api/review`, `/api/metrics`)
* **Checkpoint 3:** Endpoints test suite validating JSON requests, database updates, and inference responses.

### Phase 4: Developer Web Dashboard
* Task 4.1: `backend/static/` (HTML/CSS/JS review queue grid, `1`/`0` hotkeys)
* Task 4.2: `backend/static/` (Threshold slider, confusion matrix, retrain trigger)
* **Checkpoint 4:** Browser verification of the developer dashboard UI at `http://localhost:8000`.

### Phase 5: Tampermonkey Userscript
* Task 5.1: `userscript/taste_collector.user.js` (Extraction waterfall)
* Task 5.2: `userscript/taste_collector.user.js` (Manual, Supervised, Full Auto, ArrowLeft/Right dispatch)
* **Checkpoint 5:** Live end-to-end run on the art library website.

## Risks & Mitigations
| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| First-time model download is slow | Medium | `clip-ViT-L-14` is downloaded and cached once locally. |
| DOM element hiding or DRM in library | High | Multi-stage extraction waterfall falling back to desktop screen capture via `/api/capture`. |
| Extreme class imbalance (5% likes) | High | Balanced class weighting, threshold tuning, and evaluating with PR-AUC / $F_2$ score. |
