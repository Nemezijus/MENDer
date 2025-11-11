import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, Any

from utils.configs.configs import (
    RunConfig, DataConfig, SplitConfig,
    ScaleConfig, FeatureConfig, ModelConfig, EvalConfig,
)
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.split_factory import make_splitter
from utils.factories.pipeline_factory import make_pipeline
from utils.factories.eval_factory import make_evaluator
from utils.permutations.rng import RngManager
from utils.factories.baseline_factory import make_baseline

from ..adapters.io_adapter import LoadError
from ..progress.registry import PROGRESS  # inject this into the baseline


def train_once(payload) -> Dict[str, Any]:
    # --- Build config from payload (unchanged names) -------------------------
    data_cfg = DataConfig(
        npz_path=payload.data.npz_path,
        x_path=payload.data.x_path,
        y_path=payload.data.y_path,
        x_key=payload.data.x_key,
        y_key=payload.data.y_key,
    )
    split_cfg = SplitConfig(
        mode=payload.split.mode,
        train_frac=getattr(payload.split, "train_frac", None),
        n_splits=getattr(payload.split, "n_splits", None),
        stratified=payload.split.stratified,
        shuffle=payload.split.shuffle,
    )
    scale_cfg = ScaleConfig(**payload.scale.model_dump())
    feature_cfg = FeatureConfig(**payload.features.model_dump())
    model_cfg = ModelConfig(**payload.model.model_dump())

    # IMPORTANT: exclude progress_id from EvalConfig (not part of schema)
    eval_cfg = EvalConfig(**payload.eval.model_dump(exclude={"progress_id"}))

    cfg = RunConfig(
        data=data_cfg,
        split=split_cfg,
        scale=scale_cfg,
        features=feature_cfg,
        model=model_cfg,
        eval=eval_cfg,
    )

    # --- Load data -----------------------------------------------------------
    try:
        loader = make_data_loader(cfg.data)
        X, y = loader.load()
    except Exception as e:
        raise LoadError(str(e))

    # --- Checks --------------------------------------------------------------
    make_sanity_checker().check(X, y)

    # --- RNG ---------------------------------------------------------------
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # --- Split (hold-out) ----------------------------------------------------
    split_seed = rngm.child_seed("train/split")
    splitter = make_splitter(cfg.split, seed=split_seed)
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    # --- Fit / Predict / Score ----------------------------------------------
    pipeline = make_pipeline(cfg, rngm, stream="real")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    evaluator = make_evaluator(cfg.eval)
    real_score = float(evaluator.score(y_test, y_pred))

    labels = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    result: Dict[str, Any] = {
        "metric_name": cfg.eval.metric,
        "metric_value": real_score,
        "confusion": {
            "labels": [str(l) if not isinstance(l, (int, float)) else l for l in labels],
            "matrix": cm.astype(int).tolist(),
        },
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "notes": [],
    }

    if cfg.features.method == "pca":
        result["notes"].append("Model trained on PCA-transformed features.")
    if cfg.features.method == "lda":
        result["notes"].append("LDA was fitted on the training labels.")
    if cfg.features.method == "sfs":
        result["notes"].append("SFS performed wrapper-based feature selection on training data.")

    # --- Shuffle baseline ----------------------------------------------------
    n_shuffles = int(getattr(cfg.eval, "n_shuffles", 0) or 0)
    progress_id = getattr(payload.eval, "progress_id", None)

    if n_shuffles > 0 and progress_id:
        # PRE-INIT progress so the first poll doesn't 404
        PROGRESS.init(progress_id, total=n_shuffles, label=f"Shuffling 0/{n_shuffles}…")

        baseline = make_baseline(cfg, rngm)
        # Inject progress registry + parameters for the runner
        setattr(baseline, "progress_id", progress_id)
        setattr(baseline, "_progress_total", n_shuffles)
        setattr(baseline, "_progress", PROGRESS)

        scores = np.asarray(baseline.run(X, y), dtype=float).ravel()
        ge = int(np.sum(scores >= real_score))
        p_val = (ge + 1.0) / (scores.size + 1.0)

        result["shuffled_scores"] = [float(v) for v in scores.tolist()]
        result["p_value"] = float(p_val)
        result["notes"].append(
            f"Shuffle baseline: mean={float(np.mean(scores)):.4f} ± {float(np.std(scores)):.4f}, p≈{float(p_val):.4f}"
        )

    return result
