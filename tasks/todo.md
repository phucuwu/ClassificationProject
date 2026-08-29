# Task Checklist: Local Art Taste Classifier

## Phase 1: Environment and Dataset Database Foundation
- [x] Task 1.1: Create `requirements.txt` and verify Python dependencies.
- [x] Task 1.2: Implement `backend/database.py` with SQLite WAL mode and sample CRUD operations.
- [x] **Checkpoint 1:** Verify SQLite schema, WAL mode, and fast NumPy matrix loading from BLOBs.

## Phase 2: Vision Embedding and Machine Learning Engine
- [x] Task 2.1: Implement `backend/model.py` with `clip-ViT-L-14` feature extractor.
- [x] Task 2.2: Implement balanced Logistic Regression training, PR-AUC, $F_2$ evaluation, and threshold inference in `backend/model.py`.
- [x] **Checkpoint 2:** Verify embedding extraction and model fitting with synthetic validation tests.

## Phase 3: Backend Server and API Endpoints
- [x] Task 3.1: Create FastAPI application in `backend/app.py` with CORS and `/api/record`, `/api/predict`, `/api/capture`.
- [x] Task 3.2: Implement training and inspection endpoints: `/api/train`, `/api/samples`, `/api/review`, `/api/metrics`.
- [x] **Checkpoint 3:** Verify all backend API endpoints via automated test requests.

## Phase 4: Developer Web Dashboard
- [x] Task 4.1: Build dashboard interface in `backend/static/index.html`, `style.css`, and `app.js` with Review Queue thumbnail grid and hotkeys (`1`/`0`).
- [x] Task 4.2: Build dynamic Decision Threshold slider, confusion matrix view, and one-click Retrain controls.
- [x] **Checkpoint 4:** Verify dashboard UI at `http://localhost:8000`.

## Phase 5: Tampermonkey Userscript
- [x] Task 5.1: Build `userscript/taste_collector.user.js` with image extraction waterfall (DOM -> Canvas -> `/api/capture`).
- [x] Task 5.2: Implement Manual mode, Supervised mode HUD overlay, Full auto mode, and arrow key simulation.
- [x] **Checkpoint 5:** End-to-end integration test on the target art library.
