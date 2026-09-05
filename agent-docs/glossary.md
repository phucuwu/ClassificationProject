# Project glossary

This glossary defines standard terminology used across the backend, userscript, database, and documentation.

---

## Domain terms

* **Library**: The target art gallery website where artwork is displayed and rated.
* **Artwork**: A piece of art displayed in the library. An artwork consists of either a single image or an image set.
* **Sample**: A single database record in `data/dataset.db`. A sample contains an image hash, a local file path, a binary label, a prediction score, an operating mode, a review status, a label provenance, and a 768-dimensional vision embedding.
* **Label provenance**: The recorded origin of a Sample label. Every Sample carries exactly one value: `manual_rating` (a Manual-mode rating recorded as reviewed), `supervised_confirmation` (a Supervised-mode prediction confirmed by the user at record time), `review_confirmation` (any prior decision later confirmed or corrected by an explicit human review), or `auto_decision` (a Full auto mode automated decision that no human has confirmed).
* **Training-eligible Sample**: A Sample that may enter the Feature matrix and Label vector. A Sample is training-eligible only when it has a binary label, `reviewed = 1`, and label provenance `manual_rating`, `supervised_confirmation`, or `review_confirmation`. A Sample with `auto_decision` provenance is never training-eligible until an explicit human review changes its provenance to `review_confirmation`.
* **Image set**: A collection of related images presented together in the library under a single rating decision. The system treats each image set as one sample.
* **Primary image**: The first image in an image set. The system uses only the primary image for feature extraction and classification.
* **Taste profile**: The subjective aesthetic preference of the user, learned by the binary classification model.
* **Positive class (Like, `1`)**: An artwork that matches the user taste profile. Represents 5% to 10% of total samples.
* **Negative class (Dislike, `0`)**: An artwork that does not match the user taste profile. Represents 90% to 95% of total samples.

---

## Machine learning and data terms

* **Vision embedding**: A dense 768-dimensional `float32` vector extracted from an image using the `clip-ViT-L-14` model. Embeddings are $L_2$-normalized.
* **Feature matrix ($X$)**: A 2D NumPy array with shape $(N, 768)$ containing the embeddings of the $N$ training-eligible Samples.
* **Label vector ($y$)**: A 1D NumPy array with shape $(N,)$ containing binary integers (`1` or `0`) for the training-eligible Samples.
* **Temporal holdout**: A contiguous suffix of training-eligible Samples ordered by creation order (`id`/`created_at`). The newest Samples form the holdout and the earlier prefix forms the development partition. Neither partition is shuffled across time. A temporal holdout is valid only when it retains both classes and the configured minimum Positive-class count; otherwise temporal evaluation is unavailable and no random split is substituted.
* **Effectiveness warning**: The operational signal derived exclusively from the temporal holdout. The warning is active when temporal evaluation is unavailable or when the agreed recall-first target is unmet (holdout Positive-class count below 30, recall below `0.80`, or precision below `0.60`). The target is an operational warning threshold, not proof that a Taste profile is universally effective. Cross-validation metrics are development/tuning metrics and never the effectiveness report.
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
* **Full auto mode**: The userscript requests a prediction score from the backend, classifies the artwork against the active decision threshold, triggers the rating action automatically, and saves the record to the database as an `auto_decision` with `reviewed = 0`. Every Full auto Sample remains unreviewed until an explicit human review confirms or corrects it to `review_confirmation`.

---

## System components

* **Backend server**: The local Python FastAPI service running at `http://localhost:8000`. Handles image capture, embedding extraction, model training, prediction inference, and database queries.
* **Userscript**: The client-side Tampermonkey JavaScript injected into the library website. Captures images, renders interface badges, dispatches keyboard events, and communicates with the backend server via HTTP requests.
* **Review queue**: The dashboard table view and review interface where unreviewed automated decisions (`reviewed = 0`, `auto_decision` provenance) are displayed for human confirmation or relabeling. Confirming or correcting a decision writes `reviewed = 1` with `review_confirmation` provenance.
* **Developer dashboard**: The web UI served by the backend at `http://localhost:8000` for dataset inspection, batch review, decision threshold tuning, model retraining, and embedding space visualization.
* **Embedding space visualizer**: An interactive 2D scatter plot view in the developer dashboard for exploring the distribution and clustering of vision embeddings.
* **Dataset database**: The local SQLite database file at `data/dataset.db` running in WAL mode, storing sample metadata, binary labels, prediction scores, operating modes, review status, label provenance, and embedding BLOBs. Database-level `CHECK` constraints enforce binary-or-null labels, finite Prediction scores in `[0, 1]`, valid operating modes, binary review status, valid label provenance, and 3,072-byte Vision embeddings.
* **PCA (Principal Component Analysis)**: A linear dimensionality reduction technique that projects 768-dimensional vision embeddings onto the 2 orthogonal axes of highest variance while preserving global dataset structure.
* **t-SNE (t-Distributed Stochastic Neighbor Embedding)**: A non-linear manifold learning technique that projects 768-dimensional vision embeddings into 2D coordinates while preserving local neighborhood similarities and revealing aesthetic sub-clusters.
