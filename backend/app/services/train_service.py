import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, Any

from utils.configs.configs import RunConfig
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.split_factory import make_splitter
from utils.factories.pipeline_factory import make_pipeline
from utils.factories.eval_factory import make_evaluator
from utils.permutations.rng import RngManager
from utils.factories.baseline_factory import make_baseline
from utils.persistence.modelArtifact import ArtifactBuilderInput, build_model_artifact_meta
from utils.persistence.artifact_cache import artifact_cache

from ..adapters.io_adapter import LoadError
from ..progress.registry import PROGRESS


def train(cfg: RunConfig) -> Dict[str, Any]:
    # --- Load data -----------------------------------------------------------
    try:
        loader = make_data_loader(cfg.data)
        X, y = loader.load()
    except Exception as e:
        raise LoadError(str(e))

    # --- Checks --------------------------------------------------------------
    make_sanity_checker().check(X, y)

    # --- RNG -----------------------------------------------------------------
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # --- Split (hold-out or k-fold) -----------------------------------------
    split_seed = rngm.child_seed("train/split")
    splitter = make_splitter(cfg.split, seed=split_seed)

    # --- Fit / evaluate over splits (handles both holdout & kfold) ----------
    try:
        mode = getattr(cfg.split, "mode")
    except AttributeError:
        raise ValueError("RunConfig.split is missing required 'mode' attribute")

    if mode not in ("holdout", "kfold"):
        raise ValueError(f"Unsupported split mode in train service: {mode!r}")

    evaluator = make_evaluator(cfg.eval, kind="classification")

    fold_scores = []
    y_true_all, y_pred_all = [], []
    n_train_sizes, n_test_sizes = [], []

    for fold_id, (Xtr, Xte, ytr, yte) in enumerate(splitter.split(X, y), start=1):
        pipeline = make_pipeline(cfg, rngm, stream=f"{mode}/fold{fold_id}")
        pipeline.fit(Xtr, ytr)
        y_pred = pipeline.predict(Xte)
        last_pipeline = pipeline

        fold_scores.append(float(evaluator.score(yte, y_pred)))
        y_true_all.append(yte)
        y_pred_all.append(y_pred)
        n_train_sizes.append(int(Xtr.shape[0]))
        n_test_sizes.append(int(Xte.shape[0]))

    refit_pipeline = None
    if mode == "kfold":
        refit_pipeline = make_pipeline(cfg, rngm, stream="kfold/refit")
        refit_pipeline.fit(X, y)

    effective_pipeline = refit_pipeline if refit_pipeline is not None else last_pipeline

    # --- Aggregate scores and confusion matrix ------------------------------
    if not fold_scores:
        fold_scores = [float("nan")]

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))  # parity with CV

    y_true_all = np.concatenate(y_true_all) if y_true_all else np.array([])
    y_pred_all = np.concatenate(y_pred_all) if y_pred_all else np.array([])

    if y_true_all.size and y_pred_all.size:
        labels_arr = np.unique(np.concatenate([y_true_all, y_pred_all]))
        cm_mat = confusion_matrix(y_true_all, y_pred_all, labels=labels_arr).astype(int).tolist()
        labels_out = [str(l) if not isinstance(l, (int, float)) else l for l in labels_arr]
    else:
        cm_mat, labels_out = [], []

    n_train_avg = int(np.round(np.mean(n_train_sizes))) if n_train_sizes else 0
    n_test_avg = int(np.round(np.mean(n_test_sizes))) if n_test_sizes else 0

    if mode == "holdout":
        fold_scores_out = None
        n_splits_out = 1
        n_train_out = n_train_sizes[0] if n_train_sizes else n_train_avg
        n_test_out = n_test_sizes[0] if n_test_sizes else n_test_avg
    else:
        fold_scores_out = fold_scores
        n_splits_out = len(fold_scores)
        n_train_out = n_train_avg
        n_test_out = n_test_avg

    result: Dict[str, Any] = {
        "metric_name": cfg.eval.metric,
        "fold_scores": fold_scores_out,
        "mean_score": mean_score,
        "std_score": std_score,
        "n_splits": n_splits_out,
        "metric_value": mean_score,  # TrainResponse uses a single number; keep parity
        "confusion": {
            "labels": labels_out,
            "matrix": cm_mat,
        },
        "n_train": n_train_out,
        "n_test": n_test_out,
        "notes": [],
    }

    # --- Feature-related notes ----------------------------------------------
    if cfg.features.method == "pca":
        result["notes"].append("Model trained on PCA-transformed features.")
    if cfg.features.method == "lda":
        result["notes"].append("LDA was fitted on the training labels.")
    if cfg.features.method == "sfs":
        result["notes"].append("SFS performed wrapper-based feature selection on training data.")

    n_features_in = int(X.shape[1]) if hasattr(X, "shape") and len(X.shape) > 1 else None
    classes_out = result["confusion"]["labels"] if result.get("confusion") else None
    artifact_input = ArtifactBuilderInput(
        cfg=cfg,
        pipeline=effective_pipeline,
        n_train=n_train_out,
        n_test=n_test_out,
        n_features=n_features_in,
        classes=classes_out,
        summary={
            "metric_name": result["metric_name"],
            "metric_value": result["metric_value"],
            "mean_score": result["mean_score"],
            "std_score": result["std_score"],
            "n_splits": result["n_splits"],
            "notes": result.get("notes", []),
        },
    )
    result["artifact"] = build_model_artifact_meta(artifact_input)
    try:
        uid = result["artifact"]["uid"]
        artifact_cache.put(uid, effective_pipeline)
    except Exception:
        # Cache errors must not break training response.
        pass

    # --- Shuffle baseline ----------------------------------------------------
    n_shuffles = int(getattr(cfg.eval, "n_shuffles", 0) or 0)
    progress_id = getattr(cfg.eval, "progress_id", None)

    if n_shuffles > 0 and progress_id:
        # PRE-INIT progress so the first poll doesn't 404
        PROGRESS.init(progress_id, total=n_shuffles, label=f"Shuffling 0/{n_shuffles}…")

        baseline = make_baseline(cfg, rngm)
        # Inject progress registry + parameters for the runner
        setattr(baseline, "progress_id", progress_id)
        setattr(baseline, "_progress_total", n_shuffles)
        setattr(baseline, "_progress", PROGRESS)

        scores = np.asarray(baseline.run(X, y), dtype=float).ravel()
        ge = int(np.sum(scores >= mean_score))
        p_val = (ge + 1.0) / (scores.size + 1.0)

        result["shuffled_scores"] = [float(v) for v in scores.tolist()]
        result["p_value"] = float(p_val)
        result["notes"].append(
            f"Shuffle baseline: mean={float(np.mean(scores)):.4f} ± {float(np.std(scores)):.4f}, p≈{float(p_val):.4f}"
        )

    return result
