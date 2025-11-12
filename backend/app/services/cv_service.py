from ..adapters.io_adapter import load_X_y
from utils.configs.configs import RunConfig
from utils.factories.sanity_factory import make_sanity_checker
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.split_factory import make_splitter
from utils.factories.pipeline_factory import make_pipeline
from utils.factories.eval_factory import make_evaluator
from utils.permutations.rng import RngManager
import numpy as np

from utils.factories.baseline_factory import make_baseline
from ..adapters.io_adapter import LoadError
from ..progress.registry import PROGRESS


def run_kfold_cv(cfg: RunConfig):
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

    # --- Split (K-fold) ----------------------------------------------------
    split_seed = rngm.child_seed("train/split")
    splitter = make_splitter(cfg.split, seed=split_seed)
    
    evaluator = make_evaluator(cfg.eval, kind="classification")

    fold_scores = []
    for fold_id, (Xtr, Xte, ytr, yte) in enumerate(splitter.split(X, y), start=1):
        pipeline = make_pipeline(cfg, rngm, stream=f"cv/fold{fold_id}")
        pipeline.fit(Xtr, ytr)
        y_pred = pipeline.predict(Xte)
        fold_scores.append(evaluator.score(yte, y_pred))

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    result = {
        "metric_name": cfg.eval.metric,
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "n_splits": cfg.split.n_splits,
    }

    # --- Shuffle baseline (CV mean vs null) ---------------------------------
    n_shuffles = int(getattr(cfg.eval, "n_shuffles", 0) or 0)
    progress_id = getattr(cfg.eval, "progress_id", None)

    if n_shuffles > 0 and progress_id:
        # PRE-INIT to avoid 404 on first poll
        PROGRESS.init(progress_id, total=n_shuffles, label=f"Shuffling 0/{n_shuffles}…")

        baseline = make_baseline(cfg, rngm)
        setattr(baseline, "progress_id", progress_id)
        setattr(baseline, "_progress_total", n_shuffles)
        setattr(baseline, "_progress", PROGRESS)

        scores = np.asarray(baseline.run(X, y), dtype=float).ravel()
        ge = int(np.sum(scores >= mean_score))
        p_val = (ge + 1.0) / (scores.size + 1.0)

        result["shuffled_scores"] = [float(v) for v in scores.tolist()]
        result["p_value"] = float(p_val)

    return result
