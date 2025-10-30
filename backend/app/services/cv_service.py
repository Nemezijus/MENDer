from ..adapters.io_adapter import load_X_y
from utils.configs.configs import RunConfig
from utils.factories.data_loading_factory import make_data_loader
from utils.factories.split_factory import make_splitter
from utils.factories.pipeline_factory import make_pipeline
from utils.factories.eval_factory import make_evaluator
from utils.permutations.rng import RngManager
import numpy as np


def run_kfold_cv(cfg: RunConfig):
    loader = make_data_loader(cfg.data)
    X, y = loader.load()

    rngm = RngManager(cfg.eval.seed)
    split_seed = rngm.child_seed("cv/split")

    splitter = make_splitter(cfg.split, seed=split_seed)
    evaluator = make_evaluator(cfg.eval, kind="classification")

    fold_scores = []
    fold_indices = []
    for fold_id, (Xtr, Xte, ytr, yte) in enumerate(splitter.split(X, y), start=1):
        pipe_seed = rngm.child_seed(f"cv/fold{fold_id}")
        pipeline = make_pipeline(cfg, rngm, stream=f"cv/fold{fold_id}")
        pipeline.fit(Xtr, ytr)
        y_pred = pipeline.predict(Xte)
        score = evaluator.score(yte, y_pred)
        fold_scores.append(score)
        fold_indices.append(fold_id)

    mean_score = float(np.mean(fold_scores))
    std_score = float(np.std(fold_scores))

    return {
        "metric_name": cfg.eval.metric,
        "fold_scores": fold_scores,
        "mean_score": mean_score,
        "std_score": std_score,
        "n_splits": cfg.split.n_splits,
    }
