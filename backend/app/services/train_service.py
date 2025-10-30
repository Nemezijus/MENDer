import numpy as np
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, List

# === bring in your configs ===
from utils.configs.configs import (
    RunConfig, DataConfig, SplitConfig,
    ScaleConfig, FeatureConfig, ModelConfig, EvalConfig,
)

# === bring in your factories/strategies ===
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.split_factory import make_splitter
from utils.factories.pipeline_factory import make_pipeline
from utils.factories.eval_factory import make_evaluator
from utils.permutations.rng import RngManager

# we still reuse LoadError from io_adapter to normalize loader-related failures
from ..adapters.io_adapter import LoadError

def train_once(payload) -> Dict[str, Any]:
    """
    Reproduce (subset of) run_logreg_decoding:
    1. Build RunConfig from the request.
    2. Load + sanity check.
    3. RNG manager.
    4. Split using make_splitter().
    5. Build pipeline via make_pipeline(), fit, predict.
    6. Score using make_evaluator().
    7. Build confusion matrix + summary.
    """

    # 1. Build RunConfig (we mirror how instances/logreg_classify_with_shuffle.py expects cfg)
    data_cfg = DataConfig(
        npz_path=payload.data.npz_path,
        x_path=payload.data.x_path,
        y_path=payload.data.y_path,
        x_key=payload.data.x_key,
        y_key=payload.data.y_key,
    )

    split_cfg = SplitConfig(
        train_frac=payload.split.train_frac,
        stratified=payload.split.stratified,
    )

    scale_cfg = ScaleConfig(**payload.scale.model_dump())
    feature_cfg = FeatureConfig(**payload.features.model_dump())
    model_cfg = ModelConfig(**payload.model.model_dump())
    eval_cfg = EvalConfig(**payload.eval.model_dump())

    cfg = RunConfig(
        data=data_cfg,
        split=split_cfg,
        scale=scale_cfg,
        features=feature_cfg,
        model=model_cfg,
        eval=eval_cfg,
    )

    # 2. Load data using your loader factory
    #    NOTE: this bypasses io_adapter.load_X_y now and talks to your loader directly.
    try:
        loader = make_data_loader(cfg.data)
        X, y = loader.load()
    except Exception as e:
        # normalize to LoadError so router can map it to HTTP 400
        raise LoadError(str(e))

    # 2.1 sanity check (classification)
    sanity = make_sanity_checker()
    sanity.check(X, y)

    # 3. Central RNG
    rngm = RngManager(None if cfg.eval.seed is None else int(cfg.eval.seed))

    # 4. Split
    split_seed = rngm.child_seed("real/split")
    splitter = make_splitter(cfg.split, seed=split_seed)
    X_train, X_test, y_train, y_test = splitter.split(X, y)

    # 5. Build + fit pipeline
    pipeline = make_pipeline(cfg, rngm, stream="real")
    pipeline.fit(X_train, y_train)

    # 6. Predict + score using your evaluator strategy
    evaluator = make_evaluator(cfg.eval, kind="classification")
    y_pred = pipeline.predict(X_test)
    real_score = evaluator.score(y_test, y_pred)

    # 7. Confusion matrix
    # We'll mirror what a confusion matrix gives us in sklearn,
    # and package labels + matrix for the frontend.
    labels_unique = np.unique(np.concatenate([y_test, y_pred]))
    cm = confusion_matrix(y_test, y_pred, labels=labels_unique)
    cm_list = cm.astype(int).tolist()
    labels_list = [c.item() if hasattr(c, "item") else c for c in labels_unique]

    result = {
        "metric_name": cfg.eval.metric,
        "metric_value": float(real_score),
        "confusion": {
            "labels": labels_list,
            "matrix": cm_list,
        },
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "notes": [],
    }

    # Optional interpretive hints, like before
    if cfg.features.method == "pca":
        result["notes"].append("Model trained on PCA-transformed features.")
    if cfg.features.method == "lda":
        result["notes"].append("LDA was fitted on the training labels.")
    if cfg.features.method == "sfs":
        result["notes"].append("SFS performed wrapper-based feature selection on training data.")

    return result
