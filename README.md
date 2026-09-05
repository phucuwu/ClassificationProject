# Art taste classifier

A local system that learns your subjective taste profile from artworks in an online library and automates rating actions.

The system extracts a vision embedding for each artwork, records your rating decisions, trains a binary classification model, and predicts whether you will like future artworks. All data, images, and model weights remain on your local machine.

## System components

The project consists of four main components:

* **Backend server**: A Python FastAPI service running at `http://localhost:8000`. The server handles image capture, embedding extraction, model training, prediction inference, and database queries.
* **Dataset database**: A local SQLite database at `data/dataset.db` running in write-ahead logging (WAL) mode. The database stores sample metadata, binary labels, prediction scores, and embedding vectors.
* **Developer dashboard**: A web interface served at `http://localhost:8000`. The dashboard provides dataset inspection, a review queue for automated decisions, decision threshold calibration, and model retraining controls.
* **Userscript**: A Tampermonkey JavaScript script injected into the target library website. The script captures images, displays prediction badges, dispatches keyboard actions, and communicates with the backend server.

## Operating modes

The system operates in three modes:

* **Manual mode**: You browse the library and press arrow keys to rate artworks. The userscript captures the primary image, sends the image with your manual label to the backend server, and advances the page.
* **Supervised mode**: The userscript requests a prediction score from the backend server and displays an on-screen confidence badge. You confirm or override the predicted label before the script advances the page.
* **Full auto mode**: The userscript requests a prediction score, compares the score against the active decision threshold, dispatches the rating action automatically, and saves the sample to the database as an unreviewed automated decision (`auto_decision`, `reviewed = 0`). Full auto samples remain unreviewed until a human review confirms them, and only reviewed human-confirmed samples train the model.

## Quick start

### Prerequisites

* Python 3.10 or higher.
* A modern web browser with the Tampermonkey extension installed.
* Optional: An NVIDIA GPU with CUDA support for faster vision embedding extraction.

### Setup instructions

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/phucuwu/ClassificationProject.git
   cd ClassificationProject
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux or macOS:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the backend server and developer dashboard

### 1. Start the backend server

To start the FastAPI server, run:

```bash
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

When the server starts successfully, your terminal outputs:

```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

### 2. View the developer dashboard

To open the developer dashboard, open a web browser and navigate to:

```text
http://localhost:8000
```

The browser loads the dashboard, which displays:
* The **Review Queue** tab for inspecting automated decisions.
* The **Samples** tab for browsing and filtering the dataset.
* The **Embedding Space** tab for exploring 2D feature projections (PCA and t-SNE) with hover previews.
* The **Model & Metrics** tab for tuning the decision threshold and retraining the classifier.
* The **Console** tab for monitoring live system activity logs.

To stop the server, press `Ctrl+C` in your terminal.

## Workflow

1. **Collect initial samples in manual mode**:
   Browse the target library website. Press `Right Arrow` for Like (positive class, `1`) or `Left Arrow` for Dislike (negative class, `0`). Gather at least 50 to 100 samples with both positive and negative examples.

2. **Train the classification model**:
   Open the developer dashboard at `http://localhost:8000`. Click **Retrain Model** to fit the classifier on your stored dataset.

3. **Explore embedding space**:
   Switch to the **Embedding Space** tab to view your dataset projected in 2D. Toggle between **PCA** (to inspect global linear separation) and **t-SNE** (to discover style and aesthetic clusters). Hover over points to preview artwork thumbnails.

4. **Tune the decision threshold**:
   Adjust the decision threshold slider in the dashboard. Lower thresholds (such as `0.35`) prioritize recall so you miss fewer likes in imbalanced libraries where likes represent only 5% to 10% of samples.

5. **Run in supervised mode or full auto mode**:
   * Press `S` on the library website to toggle supervised mode. Press `Y` to accept predictions or `N` to override them.
   * Press `A` on the library website to toggle full auto mode. The userscript rates artworks automatically according to the active decision threshold.

6. **Review automated decisions**:
   Open the review queue tab in the developer dashboard. Inspect unreviewed samples, correct misclassified items with hotkeys (`1` for Like, `0` for Dislike), and click **Mark All Visible as Reviewed**.

## Machine learning pipeline

* **Feature extraction**: Uses the `clip-ViT-L-14` model to extract a 768-dimensional vision embedding from each primary image. Embeddings are $L_2$-normalized to unit length.
* **Classifier**: Uses `scikit-learn` `LogisticRegression` with `class_weight='balanced'` to account for class imbalance (likes represent 5% to 10% of samples).
* **Evaluation metrics**: Evaluates model performance using Precision-Recall Area Under Curve (PR-AUC) and the $F_2$ score, which weights recall twice as heavily as precision.
* **Dimensionality reduction**:
  * **PCA (2D)**: Linear projection that preserves global variance and dataset spread, displaying explained variance ratios for PC1 and PC2.
  * **t-SNE (2D)**: Non-linear manifold learning that preserves local neighborhood similarities to reveal aesthetic sub-clusters.
* **Inference**: Computes the predicted probability $P(\text{Like})$ between `0.0` and `1.0`. If $P \ge \theta$ (decision threshold, default `0.35`), the system classifies the artwork as a Like.

## API reference

The backend server exposes the following HTTP JSON endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/record` | `POST` | Ingests a sample with a base64 image, label, and operating mode. Extracts and saves the vision embedding. Consolidates near-duplicates. |
| `/api/predict` | `POST` | Computes the vision embedding for a base64 image and returns the prediction score and binary decision. |
| `/api/capture` | `POST` | Captures an OS desktop screen region when the browser DOM cannot access raw image bytes. |
| `/api/train` | `POST` | Trains the Logistic Regression model on all labeled samples in the database and returns evaluation metrics. |
| `/api/threshold` | `POST` | Updates the active decision threshold in the model artifact and recalculates evaluation metrics. |
| `/api/samples` | `GET` | Queries stored samples for dashboard inspection and review, with outlier filtering. |
| `/api/samples/{id}` | `DELETE` | Deletes a single sample and its stored image file. |
| `/api/samples/batch-delete` | `POST` | Deletes multiple samples and their images in a single transaction. |
| `/api/embeddings/scatter` | `GET` | Computes 2D PCA or t-SNE coordinates for all sample embeddings for interactive scatter visualization. |
| `/api/review` | `POST` | Applies batch label updates and marks samples as reviewed. |
| `/api/metrics` | `GET` | Returns dataset distribution, confusion matrix data, and PR curves across candidate thresholds. |
| `/api/benchmark` | `POST` | Starts asynchronous vision backbone benchmark evaluation in the background. |
| `/api/benchmark` | `GET` | Queries live progress and evaluation metrics of the vision backbone benchmark. |
| `/api/logs` | `GET` | Retrieves real-time interaction logs for manual ratings, predictions, auto actions, and training events. |
| `/api/logs/clear` | `POST` | Clears all activity console logs. |

## Repository structure

```
ClassificationProject/
├── agent-docs/
│   ├── design-specification.md    # Complete technical design specification
│   ├── glossary.md                # Canonical terminology and definitions
│   └── system-architecture.md     # System architecture and component contracts
├── backend/
│   ├── app.py                     # FastAPI application and route handlers
│   ├── database.py                # SQLite connection and sample queries
│   ├── model.py                   # Vision embedding extractor and classifier
│   └── static/                    # Developer dashboard frontend assets
│       ├── app.js
│       ├── index.html
│       └── style.css
├── data/
│   ├── cache/                     # Cached benchmark and computation outputs
│   ├── dataset.db                 # SQLite dataset database (WAL mode)
│   ├── images/                    # Local cache of primary image files
│   └── model.json                 # Serialized model weights and evaluation metrics
├── tasks/
│   ├── benchmark_backbones.py     # Vision backbone benchmark engine
│   └── todo.md                    # Project roadmap and checklist
├── tests/                         # Automated test suites
├── userscript/
│   └── taste_collector.user.js    # Tampermonkey browser script
├── requirements.txt               # Python package dependencies
└── README.md                      # Project documentation
```

