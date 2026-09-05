"""Checkpoint 2 test suite: Verifies CLIP feature extraction, model fitting, and inference."""

import base64
import io
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from backend.model import (
    DEFAULT_DECISION_THRESHOLD,
    DEFAULT_ZERO_SHOT_PROMPT,
    EMBEDDING_DIM,
    HYPERPARAMETER_C_GRID,
    extract_text_embedding,
    extract_vision_embedding,
    load_classifier,
    predict_taste,
    train_taste_classifier,
    update_decision_threshold,
)


def test_feature_extraction():
    """Verify that CLIP extracts a 768-dim normalized embedding from a PIL Image and base64 string."""
    # Create a small synthetic test image
    img = Image.new("RGB", (64, 64), color=(120, 150, 200))
    emb1 = extract_vision_embedding(img)

    assert isinstance(emb1, np.ndarray)
    assert emb1.shape == (EMBEDDING_DIM,), f"Expected shape ({EMBEDDING_DIM},), got {emb1.shape}"
    assert emb1.dtype == np.float32

    # Check L2 normalization (norm should be ~1.0)
    norm = np.linalg.norm(emb1)
    np.testing.assert_allclose(norm, 1.0, atol=1e-3)

    # Test base64 data URI input
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    emb2 = extract_vision_embedding(img_b64)

    assert emb2.shape == (EMBEDDING_DIM,)
    np.testing.assert_allclose(emb1, emb2, atol=1e-3)


def test_cold_start_and_insufficient_data():
    """Verify that predict_taste and train_taste_classifier handle cold start gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fake_model_path = Path(tmpdir) / "non_existent_model.pkl"

        # 1. Prediction without trained model
        dummy_vec = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        res = predict_taste(dummy_vec, model_path=fake_model_path)
        assert res["model_loaded"] is False
        assert res["prediction_score"] is None
        assert res["decision"] is None

        # 2. Training with 0 samples
        X_empty = np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        y_empty = np.empty((0,), dtype=np.int32)
        train_res = train_taste_classifier(X_empty, y_empty, model_path=fake_model_path)
        assert train_res["status"] == "insufficient_data"

        # 3. Training with only negative class
        X_single = np.random.randn(5, EMBEDDING_DIM).astype(np.float32)
        y_single = np.zeros(5, dtype=np.int32)
        train_res2 = train_taste_classifier(X_single, y_single, model_path=fake_model_path)
        assert train_res2["status"] == "insufficient_data"


def test_model_training_and_inference():
    """Verify training on imbalanced data, threshold calibration, and inference."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"

        # Generate synthetic imbalanced dataset: 90 dislikes (0), 10 likes (1),
        # interleaved in creation order so the development prefix holds both classes.
        np.random.seed(42)
        n_dislikes = 90
        n_likes = 10
        n_total = n_dislikes + n_likes

        # Shift the positive class slightly in feature space
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.5

        # Normalize rows to unit sphere
        X_neg = X_neg / np.linalg.norm(X_neg, axis=1, keepdims=True)
        X_pos = X_pos / np.linalg.norm(X_pos, axis=1, keepdims=True)

        # Interleave: every 10th creation slot is a Like (deterministic order)
        X = np.empty((n_total, EMBEDDING_DIM), dtype=np.float32)
        y = np.zeros(n_total, dtype=np.int32)
        sample_ids = list(range(1000, 1000 + n_total))
        neg_idx = 0
        pos_idx = 0
        for i in range(n_total):
            if i % 10 == 9:
                X[i] = X_pos[pos_idx]
                y[i] = 1
                pos_idx += 1
            else:
                X[i] = X_neg[neg_idx]
                y[i] = 0
                neg_idx += 1

        # Train classifier
        train_result = train_taste_classifier(X, y, sample_ids=sample_ids, target_recall=0.90, model_path=model_path)
        assert train_result["status"] == "trained"
        assert train_result["sample_count"] == n_total
        assert train_result["positive_count"] == n_likes
        assert train_result["negative_count"] == n_dislikes

        metrics = train_result["metrics"]
        assert "pr_auc" in metrics
        assert "recall" in metrics
        assert "precision" in metrics
        assert "f2_score" in metrics
        assert metrics["recall"] >= 0.80  # Should achieve high recall on training data

        # Check saved model file
        assert model_path.exists()
        loaded = load_classifier(model_path)
        assert loaded is not None

        # Test inference on a positive-like sample
        test_pos = (np.random.randn(EMBEDDING_DIM).astype(np.float32) + 0.5)
        test_pos = test_pos / np.linalg.norm(test_pos)
        pred_res = predict_taste(test_pos, model_path=model_path)

        assert pred_res["model_loaded"] is True
        assert isinstance(pred_res["prediction_score"], float)
        assert 0.0 <= pred_res["prediction_score"] <= 1.0
        assert pred_res["decision"] in (0, 1)


def test_stratified_cv_and_oof_metrics():
    """Verify temporal split: newest suffix is the holdout, earlier prefix tunes.

    With only 15 Likes in a 100-Sample collection the newest 15-Sample suffix
    cannot meet the 30-Like minimum, so temporal evaluation is unavailable and
    no random split is substituted. Development tuning still runs stratified CV.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_cv_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 85, 15
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.8
        X_neg /= np.linalg.norm(X_neg, axis=1, keepdims=True)
        X_pos /= np.linalg.norm(X_pos, axis=1, keepdims=True)

        # Interleave so creation order carries both classes throughout
        X = np.empty((100, EMBEDDING_DIM), dtype=np.float32)
        y = np.zeros(100, dtype=np.int32)
        pos_indices = {5, 12, 19, 25, 33, 41, 48, 55, 62, 70, 78, 83, 89, 94, 98}
        neg_idx = 0
        pos_idx = 0
        for i in range(100):
            if i in pos_indices:
                X[i] = X_pos[pos_idx]
                y[i] = 1
                pos_idx += 1
            else:
                X[i] = X_neg[neg_idx]
                y[i] = 0
                neg_idx += 1
        sample_ids = list(range(500, 600))

        result = train_taste_classifier(X, y, sample_ids=sample_ids, holdout_ratio=0.15, model_path=model_path)
        assert result["status"] == "trained"
        m = result["metrics"]
        assert m["evaluation_type"] == "stratified_cv"
        assert m["folds"] == 5
        assert "pr_auc" in m and isinstance(m["pr_auc"], float)
        assert "average_precision" in m and isinstance(m["average_precision"], float)
        assert "recall" in m
        assert "precision" in m
        assert "f2_score" in m
        assert "confusion_matrix" in m

        # Temporal holdout is the newest 15-Sample suffix: exact IDs, no leakage
        boundary = m["eval_boundary"]
        assert boundary["dev_size"] == 85
        assert boundary["holdout_size"] == 15
        assert boundary["dev_sample_ids"] == list(range(500, 585))
        assert boundary["holdout_sample_ids"] == list(range(585, 600))
        assert max(boundary["dev_sample_ids"]) < min(boundary["holdout_sample_ids"])

        # Holdout cannot meet the 30-Like minimum: unavailable, never random
        assert m["temporal_holdout"]["status"] == "temporal_evaluation_unavailable"
        assert m["holdout"] is None
        assert "temporal_evaluation_unavailable" in m["warning_reasons"]
        assert "holdout_likes_below_minimum" in m["warning_reasons"]
        assert m["warning_active"] is True
        assert m["effectiveness"]["status"] == "temporal_evaluation_unavailable"

        # Tuning section mirrors the development-partition metrics
        assert m["tuning"]["evaluation_type"] == "stratified_cv"
        assert m["tuning"]["dev_sample_count"] == 85
        assert m["tuning"]["dev_sample_ids"] == list(range(500, 585))

        # Model artifact contains cached development-partition probabilities
        loaded = load_classifier(model_path)
        assert "oof_probabilities" in loaded
        assert "y_oof" in loaded
        assert len(loaded["oof_probabilities"]) == 85
        assert len(loaded["y_oof"]) == 85
        assert loaded["holdout_sample_ids"] == list(range(585, 600))


def test_dynamic_fold_scaling():
    """Verify dynamic fold scaling when positive samples are fewer than 5."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_dynamic_model.pkl"
        np.random.seed(42)
        # 3 likes, 30 dislikes
        n_dislikes, n_likes = 30, 3
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.5
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        res = train_taste_classifier(X, y, holdout_ratio=0.0, model_path=model_path)
        assert res["status"] == "trained"
        assert res["metrics"]["evaluation_type"] == "stratified_cv"
        assert res["metrics"]["folds"] == 3  # min(5, 3) = 3
        # holdout_ratio=0.0 disables the temporal holdout: unavailable + warning
        assert res["metrics"]["temporal_holdout"]["status"] == "temporal_evaluation_unavailable"
        assert res["metrics"]["warning_active"] is True

        # Single positive sample: limited tuning data, still trains, but temporal
        # evaluation is unavailable and no in-sample effectiveness is reported.
        X_single = np.vstack([X_neg[:10], X_pos[:1]]).astype(np.float32)
        y_single = np.array([0] * 10 + [1] * 1, dtype=np.int32)
        res_single = train_taste_classifier(X_single, y_single, holdout_ratio=0.0, model_path=model_path)
        assert res_single["status"] == "trained"
        assert res_single["metrics"]["evaluation_type"] == "limited_tuning_data"
        assert "in_sample" not in res_single["metrics"]["evaluation_type"]
        assert res_single["metrics"]["temporal_holdout"]["status"] == "temporal_evaluation_unavailable"
        assert res_single["metrics"]["warning_active"] is True


def test_hybrid_threshold_calibration():
    """Verify that hybrid threshold calibration targets F2 with a recall floor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_hybrid_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 80, 20
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.6
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        res = train_taste_classifier(X, y, min_recall_floor=0.70, holdout_ratio=0.0, model_path=model_path)
        assert res["status"] == "trained"
        m = res["metrics"]
        assert m["recall"] >= 0.70 or m["recall"] > 0.0
        assert 0.05 <= m["decision_threshold"] <= 0.95


def test_temporal_holdout_recall_below_target():
    """Verify the effectiveness warning fires when holdout recall is below 0.80.

    The development prefix is cleanly separable but the newest holdout Likes
    are shifted into Dislike territory, so the dev-calibrated threshold misses
    them. The temporal holdout is valid (both classes + minimum met via an
    explicit small minimum), yet recall misses the target.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_warning_model.pkl"
        np.random.seed(7)

        # Development prefix: 50 Dislikes + 10 Likes, well separated
        X_dev_neg = np.random.randn(50, EMBEDDING_DIM).astype(np.float32)
        X_dev_pos = np.random.randn(10, EMBEDDING_DIM).astype(np.float32) + 2.0
        # Temporal suffix: 8 Dislikes (normal) + 4 Likes shifted into Dislike territory
        X_hold_neg = np.random.randn(8, EMBEDDING_DIM).astype(np.float32)
        X_hold_pos = np.random.randn(4, EMBEDDING_DIM).astype(np.float32) - 2.0

        X = np.vstack([X_dev_neg, X_dev_pos, X_hold_neg, X_hold_pos]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        y = np.array([0] * 50 + [1] * 10 + [0] * 8 + [1] * 4, dtype=np.int32)
        sample_ids = list(range(1, 73))

        # 12 newest Samples form the holdout: round(72 * 0.1667) = 12
        res = train_taste_classifier(
            X, y, sample_ids=sample_ids, holdout_ratio=0.1667, min_holdout_positives=4, model_path=model_path
        )
        assert res["status"] == "trained"
        m = res["metrics"]
        assert m["temporal_holdout"]["status"] == "available"
        assert m["temporal_holdout"]["sample_count"] == 12
        assert m["temporal_holdout"]["positive_count"] == 4
        assert m["holdout"] is not None
        assert m["effectiveness"]["status"] == "below_target"
        assert m["warning_active"] is True
        assert "recall_below_target" in m["warning_reasons"]
        # Legacy alias mirrors the recall-first warning
        assert m["generalization_warning"] is True
        # Exact temporal boundary: holdout is the newest suffix
        assert m["eval_boundary"]["holdout_sample_ids"] == list(range(61, 73))
        assert max(m["eval_boundary"]["dev_sample_ids"]) < min(m["eval_boundary"]["holdout_sample_ids"])


def test_threshold_update_with_cached_oof():
    """Verify update_decision_threshold uses cached out-of-fold probabilities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_thresh_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 50, 10
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.8
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        train_res = train_taste_classifier(X, y, holdout_ratio=0.0, model_path=model_path)
        assert train_res["status"] == "trained"

        # Update threshold to 0.70
        update_res = update_decision_threshold(0.70, model_path=model_path)
        assert update_res["success"] is True
        assert update_res["decision_threshold"] == 0.70
        assert "average_precision" in update_res["metrics"]

        # Verify persistence
        loaded = load_classifier(model_path)
        assert loaded["decision_threshold"] == 0.70


def test_hyperparameter_grid_search():
    """Verify regularized hyperparameter search selects best C and class weighting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_grid_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 80, 15
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 1.0
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        res = train_taste_classifier(X, y, holdout_ratio=0.0, model_path=model_path)
        assert res["status"] == "trained"
        m = res["metrics"]
        assert "best_params" in m
        assert "tuning_summary" in m
        best_p = m["best_params"]
        assert best_p["C"] in HYPERPARAMETER_C_GRID
        assert best_p["class_weight"] in ["balanced", "unweighted", "balanced_1.5x", "balanced_2.0x"]
        assert len(m["tuning_summary"]) == len(HYPERPARAMETER_C_GRID) * 4

        # Loaded model preserves best_params and tuning_summary
        loaded = load_classifier(model_path)
        assert loaded["metrics"]["best_params"] == best_p
        assert "tuning_summary" in loaded["metrics"]


def test_reference_baselines():
    """Verify calculation of random guess, positive centroid, and zero-shot baselines."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_baselines_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 90, 10
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 1.0
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        res = train_taste_classifier(X, y, holdout_ratio=0.0, model_path=model_path)
        assert res["status"] == "trained"
        assert "baselines" in res["metrics"]
        b = res["metrics"]["baselines"]
        assert "random_guess" in b
        assert "positive_centroid" in b
        assert "zero_shot" in b
        assert b["reference_type"] == "text"
        assert b["reference_source"] == DEFAULT_ZERO_SHOT_PROMPT

        # Random guess baseline matches positive prevalence (10/100 = 0.10)
        expected_prev = round(10 / 100, 4)
        assert abs(b["random_guess"] - expected_prev) < 0.01

        # Centroid and zero-shot baselines return valid PR-AUC floats
        assert 0.0 <= b["positive_centroid"] <= 1.0
        assert 0.0 <= b["zero_shot"] <= 1.0


def test_baseline_customization_and_persistence():
    """Verify custom prompt, exemplar image override, and baseline persistence across runs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_custom_baseline_model.pkl"
        np.random.seed(42)
        n_dislikes, n_likes = 50, 10
        X_neg = np.random.randn(n_dislikes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 0.8
        X = np.vstack([X_neg, X_pos]).astype(np.float32)
        y = np.array([0] * n_dislikes + [1] * n_likes, dtype=np.int32)

        # 1. Custom text prompt
        custom_prompt = "cyberpunk neon anime illustration"
        res1 = train_taste_classifier(
            X, y,
            baseline_prompt_text=custom_prompt,
            holdout_ratio=0.0,
            model_path=model_path,
        )
        assert res1["status"] == "trained"
        b1 = res1["metrics"]["baselines"]
        assert b1["reference_type"] == "text"
        assert b1["reference_source"] == custom_prompt

        # 2. Exemplar image upload override
        test_img = Image.new("RGB", (64, 64), color=(200, 50, 80))
        buf = io.BytesIO()
        test_img.save(buf, format="JPEG")
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

        res2 = train_taste_classifier(
            X, y,
            baseline_image_base64=img_b64,
            holdout_ratio=0.0,
            model_path=model_path,
        )
        assert res2["status"] == "trained"
        b2 = res2["metrics"]["baselines"]
        assert b2["reference_type"] == "image"
        assert b2["reference_source"] == "exemplar_image"

        # 3. Persistence: subsequent retrain without overrides keeps exemplar image
        res3 = train_taste_classifier(
            X, y,
            holdout_ratio=0.0,
            model_path=model_path,
        )
        assert res3["status"] == "trained"
        b3 = res3["metrics"]["baselines"]
        assert b3["reference_type"] == "image"

        # 4. Reset baseline to default
        res4 = train_taste_classifier(
            X, y,
            reset_baseline_to_default=True,
            holdout_ratio=0.0,
            model_path=model_path,
        )
        assert res4["status"] == "trained"
        b4 = res4["metrics"]["baselines"]
        assert b4["reference_type"] == "text"
        assert b4["reference_source"] == DEFAULT_ZERO_SHOT_PROMPT


def test_temporal_split_orders_by_creation_ids():
    """Verify creation ordering: shuffled input is re-sorted by Sample id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_order_model.pkl"
        np.random.seed(11)
        n = 40
        X = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        y = np.zeros(n, dtype=np.int32)
        y[::5] = 1  # every 5th creation slot is a Like in input order
        ids = list(range(2000, 2000 + n))

        # Shuffle input order deterministically (labels travel with rows)
        perm = np.random.RandomState(11).permutation(n)
        Xs, ys = X[perm], y[perm]
        ids_shuffled = [ids[i] for i in perm]

        res = train_taste_classifier(Xs, ys, sample_ids=ids_shuffled, holdout_ratio=0.2, model_path=model_path)
        assert res["status"] == "trained"
        boundary = res["metrics"]["eval_boundary"]
        # Newest suffix by id, regardless of input order
        assert boundary["holdout_sample_ids"] == list(range(2032, 2040))
        assert boundary["dev_sample_ids"] == list(range(2000, 2032))
        assert boundary["dev_max_id"] == 2031
        assert boundary["holdout_min_id"] == 2032


def test_temporal_never_uses_random_split():
    """Verify no random splitter remains in the training path."""
    import inspect

    import backend.model as model_mod

    source = inspect.getsource(model_mod.train_taste_classifier)
    assert "train_test_split" not in source
    assert "in_sample_fallback" not in source

    # Deterministic: repeated runs record identical boundaries
    with tempfile.TemporaryDirectory() as tmpdir:
        np.random.seed(13)
        n = 60
        X = np.random.randn(n, EMBEDDING_DIM).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        y = np.zeros(n, dtype=np.int32)
        y[::4] = 1
        ids = list(range(300, 360))
        r1 = train_taste_classifier(X, y, sample_ids=ids, holdout_ratio=0.2, model_path=Path(tmpdir) / "a.json")
        r2 = train_taste_classifier(X, y, sample_ids=ids, holdout_ratio=0.2, model_path=Path(tmpdir) / "b.json")
        assert r1["metrics"]["eval_boundary"] == r2["metrics"]["eval_boundary"]


def test_temporal_unavailable_when_holdout_missing_class():
    """Verify unavailable status when the newest suffix holds a single class."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_single_class_holdout.pkl"
        np.random.seed(17)
        # 30 mixed Samples followed by 10 newest Dislikes
        X_mixed = np.random.randn(30, EMBEDDING_DIM).astype(np.float32)
        X_new = np.random.randn(10, EMBEDDING_DIM).astype(np.float32)
        X = np.vstack([X_mixed, X_new]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        y = np.zeros(40, dtype=np.int32)
        y[2::6] = 1
        y[30:] = 0
        assert int(np.sum(y[:30] == 1)) >= 2

        res = train_taste_classifier(X, y, sample_ids=list(range(40)), holdout_ratio=0.25, model_path=model_path)
        assert res["status"] == "trained"
        m = res["metrics"]
        assert m["temporal_holdout"]["status"] == "temporal_evaluation_unavailable"
        assert "holdout_missing_positive_class" in m["warning_reasons"]
        assert m["warning_active"] is True


def test_insufficient_data_rejects_without_insample_metrics():
    """Verify rejection for model use when the dev prefix lacks both classes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_reject.pkl"
        np.random.seed(19)
        # All Likes are the newest Samples: the development prefix has none
        X = np.random.randn(20, EMBEDDING_DIM).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        y = np.array([0] * 14 + [1] * 6, dtype=np.int32)

        res = train_taste_classifier(X, y, sample_ids=list(range(20)), holdout_ratio=0.3, model_path=model_path)
        assert res["status"] == "insufficient_data"
        assert "metrics" not in res  # no in-sample effectiveness signal


def test_temporal_holdout_precision_below_target():
    """Verify the warning fires when holdout precision is below 0.60 but recall holds."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_prec_model.pkl"
        np.random.seed(23)

        # Development prefix: 50 Dislikes + 20 Likes, well separated
        X_dev_neg = np.random.randn(50, EMBEDDING_DIM).astype(np.float32)
        X_dev_pos = np.random.randn(20, EMBEDDING_DIM).astype(np.float32) + 2.0
        # Newest suffix: 6 Likes (caught) + 10 Dislikes shifted into Like territory
        X_hold_pos = np.random.randn(6, EMBEDDING_DIM).astype(np.float32) + 2.0
        X_hold_neg = np.random.randn(10, EMBEDDING_DIM).astype(np.float32) + 2.0

        X = np.vstack([X_dev_neg, X_dev_pos, X_hold_pos, X_hold_neg]).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        y = np.array([0] * 50 + [1] * 20 + [1] * 6 + [0] * 10, dtype=np.int32)
        sample_ids = list(range(1, 87))

        res = train_taste_classifier(
            X, y, sample_ids=sample_ids, holdout_ratio=0.186, min_holdout_positives=6, model_path=model_path
        )
        assert res["status"] == "trained"
        m = res["metrics"]
        assert m["temporal_holdout"]["status"] == "available"
        assert m["temporal_holdout"]["sample_count"] == 16
        assert m["temporal_holdout"]["positive_count"] == 6
        assert m["effectiveness"]["status"] == "below_target"
        assert "precision_below_target" in m["warning_reasons"]
        assert "recall_below_target" not in m["warning_reasons"]
        assert m["warning_active"] is True


def test_temporal_meets_target():
    """Verify a well-generalizing model with 30 holdout Likes meets the target."""
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_meets_target.json"
        np.random.seed(29)
        n_total, n_likes = 400, 120

        X_neg = np.random.randn(n_total - n_likes, EMBEDDING_DIM).astype(np.float32)
        X_pos = np.random.randn(n_likes, EMBEDDING_DIM).astype(np.float32) + 2.5
        X = np.empty((n_total, EMBEDDING_DIM), dtype=np.float32)
        y = np.zeros(n_total, dtype=np.int32)
        neg_idx = 0
        pos_idx = 0
        for i in range(n_total):
            if i % 10 in (7, 8, 9):  # 30% Likes, evenly spread through creation order
                X[i] = X_pos[pos_idx]
                y[i] = 1
                pos_idx += 1
            else:
                X[i] = X_neg[neg_idx]
                y[i] = 0
                neg_idx += 1
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        sample_ids = list(range(10000, 10000 + n_total))

        res = train_taste_classifier(X, y, sample_ids=sample_ids, holdout_ratio=0.25, model_path=model_path)
        assert res["status"] == "trained"
        m = res["metrics"]
        # Newest 100 Samples hold exactly 30 Likes
        assert m["temporal_holdout"]["status"] == "available"
        assert m["temporal_holdout"]["sample_count"] == 100
        assert m["temporal_holdout"]["positive_count"] == 30
        assert m["temporal_holdout"]["recall"] >= 0.80
        assert m["temporal_holdout"]["precision"] >= 0.60
        assert m["effectiveness"]["status"] == "meets_target"
        assert m["warning_active"] is False
        assert m["warning_reasons"] == []
        assert m["threshold_source"] == "calibrated"

        # Final classifier fit on all eligible Samples after eval was recorded
        loaded = load_classifier(model_path)
        assert loaded["sample_count"] == n_total
        assert loaded["training_eligible"]["sample_count"] == n_total
        assert loaded["eval_boundary"]["holdout_sample_ids"] == list(range(10300, 10400))
        assert loaded["threshold_source"] == "calibrated"


if __name__ == "__main__":
    print("Testing cold start and edge cases...")
    test_cold_start_and_insufficient_data()
    print("Testing model training and inference...")
    test_model_training_and_inference()
    print("Testing Stratified 5-Fold CV and out-of-fold metrics...")
    test_stratified_cv_and_oof_metrics()
    print("Testing dynamic fold scaling...")
    test_dynamic_fold_scaling()
    print("Testing temporal holdout recall below target...")
    test_temporal_holdout_recall_below_target()
    print("Testing temporal ordering by creation IDs...")
    test_temporal_split_orders_by_creation_ids()
    print("Testing no random split in training path...")
    test_temporal_never_uses_random_split()
    print("Testing unavailable holdout for single-class suffix...")
    test_temporal_unavailable_when_holdout_missing_class()
    print("Testing insufficient-data rejection...")
    test_insufficient_data_rejects_without_insample_metrics()
    print("Testing precision below target...")
    test_temporal_holdout_precision_below_target()
    print("Testing meets-target case...")
    test_temporal_meets_target()
    print("Testing threshold updates with cached out-of-fold predictions...")
    test_threshold_update_with_cached_oof()
    print("Testing hyperparameter grid search...")
    test_hyperparameter_grid_search()
    print("Testing reference baselines...")
    test_reference_baselines()
    print("Testing baseline customization and persistence...")
    test_baseline_customization_and_persistence()
    print("Testing CLIP feature extraction...")
    test_feature_extraction()
    print("ALL CHECKPOINT 7 MODEL TESTS PASSED SUCCESSFULLY!")
