# utils/strategies/baselines.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from sklearn.base import clone

from utils.configs.configs import RunConfig
from utils.strategies.interfaces import BaselineRunner
from utils.permutations.rng import RngManager
from utils.permutations.shuffle import shuffle_simple_vector

# factories we already have
from utils.factories.split_factory import make_splitter
from utils.factories.scale_factory import make_scaler
from utils.factories.feature_factory import make_features
from utils.factories.model_factory import make_model
from utils.factories.training_factory import make_trainer
from utils.factories.predict_factory import make_predictor
from utils.factories.eval_factory import make_evaluator


@dataclass
class LabelShuffleBaseline(BaselineRunner):
    """
    Shuffle-label baseline that reuses the same pipeline as the real run:
    split -> scale -> features -> fit -> predict -> score.

    RNG policy (matches your previous behavior):
      - For iteration i:
          lbl_gen   = rngm.child_generator(f"shuffle_{i}/labels")  # to permute y
          pipe_seed = rngm.child_seed(f"shuffle_{i}/pipeline")     # used as the *single* int seed
        We pass `pipe_seed` to both the splitter and the feature extractor,
        mirroring how `classification.py` used a single seed for the whole pipeline.
    """
    cfg: RunConfig
    rngm: RngManager

    def run(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_shuffles = int(self.cfg.eval.n_shuffles)
        if n_shuffles < 1:
            raise ValueError("n_shuffles must be >= 1")

        scores = np.empty(n_shuffles, dtype=float)

        for i in range(n_shuffles):
            # Per-iteration RNG: labels + pipeline
            lbl_gen = self.rngm.child_generator(f"shuffle_{i}/labels")
            pipe_seed = self.rngm.child_seed(f"shuffle_{i}/pipeline")

            # Shuffle labels
            y_shuf = shuffle_simple_vector(y, rng=lbl_gen)

            # Fresh pipeline objects for this iteration
            splitter   = make_splitter(self.cfg.split, seed=pipe_seed)
            scaler     = make_scaler(self.cfg.scale)
            features   = make_features(self.cfg.features, seed=pipe_seed, model_cfg=self.cfg.model, eval_cfg=self.cfg.eval)
            model_bld  = make_model(self.cfg.model)
            trainer    = make_trainer()
            predictor  = make_predictor()
            evaluator  = make_evaluator(self.cfg.eval, kind="classification")

            # Data flow
            Xtr, Xte, ytr, yte = splitter.split(X, y_shuf)
            Xtr, Xte = scaler.fit_transform(Xtr, Xte)
            _, Xtr_fx, Xte_fx = features.fit_transform_train_test(Xtr, Xte, ytr)

            # Model: build fresh per iteration (equivalent to clone(model))
            model = model_bld.build()
            model = trainer.fit(model, Xtr_fx, ytr)
            y_pred = predictor.predict(model, Xte_fx)
            scores[i] = evaluator.score(yte, y_pred)

        return scores
