# Project glossary

This glossary defines standard terminology used across the backend, userscript, database, and documentation.

---

## Domain terms

* **Library**: The target art gallery website where artwork is displayed and rated.
* **Artwork**: A piece of art displayed in the library. An artwork consists of either a single image or an image set.
* **Sample**: A single database record in `data/dataset.db`. A sample contains an image hash, a local file path, a binary label, a prediction score, an operating mode, a review status, and a 768-dimensional vision embedding.
* **Image set**: A collection of related images presented together in the library under a single rating decision. The system treats each image set as one sample.
* **Primary image**: The first image in an image set. The system uses only the primary image for feature extraction and classification.
* **Taste profile**: The subjective aesthetic preference of the user, learned by the binary classification model.
* **Positive class (Like, `1`)**: An artwork that matches the user taste profile. Represents 5% to 10% of total samples.
* **Negative class (Dislike, `0`)**: An artwork that does not match the user taste profile. Represents 90% to 95% of total samples.

---

## Machine learning and data terms

* **Vision embedding**: A dense 768-dimensional `float32` vector extracted from an image using the `clip-ViT-L-14` model. Embeddings are $L_2$-normalized.
* **Feature matrix ($X$)**: A 2D NumPy array with shape $(N, 768)$ containing $N$ sample embeddings.
* **Label vector ($y$)**: A 1D NumPy array with shape $(N,)$ containing binary integers (`1` or `0`).
* **Prediction score ($P$)**: The model output float between `0.0` and `1.0` representing the estimated probability that the user likes the artwork ($P(\text{Like})$).
* **Decision threshold ($\theta$)**: The cutoff probability value where $P \ge \theta$ classifies a sample as a Like. Configured below `0.5` (default `0.35`) to prioritize recall.
* **Recall**: The proportion of all true likes that the model detects: $\frac{\text{TP}}{\text{TP} + \text{FN}}$.
* **Precision**: The proportion of predicted likes that are true likes: $\frac{\text{TP}}{\text{TP} + \text{FP}}$.
* **PR-AUC**: The area under the Precision-Recall curve. The primary evaluation metric for model training on this imbalanced dataset.
* **$F_2$ score**: An evaluation metric that weights recall twice as heavily as precision: $\frac{5 \cdot \text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$.
* **Near-duplicate detection**: Pairwise cosine similarity comparison identifying when an incoming artwork embedding is visually identical or nearly identical (similarity $\ge 0.98$) to an existing sample in the dataset database.
* **Outlier detection**: Distance and score analysis flagging ratings that contradict established clusters in feature space (distance $> 2\sigma$ from class centroid or out-of-fold score inconsistent with assigned label).
* **Session drift**: Deviation in rating distribution over a rolling window of recent samples, flagging when the positive class ratio strays outside the expected 5% to 10% baseline range.
* **Backbone benchmark**: An evaluation engine that scores alternative pretrained vision models (such as SigLIP and DINOv2) against baseline CLIP on cross-validated PR-AUC and $F_2$ metrics.

---

## Operating modes

* **Manual mode**: The user browses the library and presses rating hotkeys. The userscript captures the primary image, sends the image with the manual label to the backend, and advances the page.
* **Supervised mode**: The userscript requests a prediction score from the backend, displays an on-screen confidence badge, and waits for user confirmation before logging the sample and advancing the page.
* **Full auto mode**: The userscript requests a prediction score from the backend, classifies the artwork against the active decision threshold, triggers the rating action automatically, and saves the record to the database with `reviewed = 0`.

---

## System components

* **Backend server**: The local Python FastAPI service running at `http://localhost:8000`. Handles image capture, embedding extraction, model training, prediction inference, and database queries.
* **Userscript**: The client-side Tampermonkey JavaScript injected into the library website. Captures images, renders interface badges, dispatches keyboard events, and communicates with the backend server via HTTP requests.
* **Review queue**: The dashboard table view and review interface where unreviewed automated decisions (`reviewed = 0`) are displayed for human confirmation or relabeling.
* **Developer dashboard**: The web UI served by the backend at `http://localhost:8000` for dataset inspection, batch review, decision threshold tuning, model retraining, and embedding space visualization.
* **Embedding space visualizer**: An interactive 2D scatter plot view in the developer dashboard for exploring the distribution and clustering of vision embeddings.
* **Dataset database**: The local SQLite database file at `data/dataset.db` running in WAL mode, storing sample metadata, binary labels, prediction scores, and embedding BLOBs.
* **PCA (Principal Component Analysis)**: A linear dimensionality reduction technique that projects 768-dimensional vision embeddings onto the 2 orthogonal axes of highest variance while preserving global dataset structure.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: A non-linear manifold learning technique that projects 768-dimensional vision embeddings into 2D coordinates while preserving local neighborhood similarities and revealing aesthetic sub-clusters.
