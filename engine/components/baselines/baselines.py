from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

from engine.contracts.run_config import RunConfig
from engine.contracts.model_configs import get_model_task
from engine.components.interfaces import BaselineRunner
from engine.runtime.random.rng import RngManager
from engine.runtime.random.shuffle import shuffle_simple_vector

from engine.factories.split_factory import make_splitter
from engine.factories.scale_factory import make_scaler
from engine.factories.feature_factory import make_features
from engine.factories.model_factory import make_model
from engine.factories.training_factory import make_trainer
from engine.factories.predict_factory import make_predictor
from engine.factories.eval_factory import make_evaluator
from engine.components.splitters.types import Split


@dataclass
class LabelShuffleBaseline(BaselineRunner):
    cfg: RunConfig
    rngm: RngManager

    def _get_eval_kind(self) -> str:
        """
        Decide whether to use classification or regression scoring
        based on the model config's task metadata.
        """
        task = get_model_task(self.cfg.model)
        return "regression" if task == "regression" else "classification"

    def _score_once_holdout(self, X: np.ndarray, y: np.ndarray, seed: int) -> float:
        splitter   = make_splitter(self.cfg.split, seed=seed)
        scaler     = make_scaler(self.cfg.scale)
        features   = make_features(self.cfg.features, seed=seed, model_cfg=self.cfg.model, eval_cfg=self.cfg.eval)
        model_bld  = make_model(self.cfg.model)
        trainer    = make_trainer()
        predictor  = make_predictor()

        eval_kind  = self._get_eval_kind()
        evaluator  = make_evaluator(self.cfg.eval, kind=eval_kind)

        split: Split = next(splitter.split(X, y))
        Xtr, Xte, ytr, yte = split.Xtr, split.Xte, split.ytr, split.yte
        Xtr, Xte = scaler.fit_transform(Xtr, Xte)
        _, Xtr_fx, Xte_fx = features.fit_transform_train_test(Xtr, Xte, ytr)

        model = model_bld.make_estimator()
        model = trainer.fit(model, Xtr_fx, ytr)
        y_pred = predictor.predict(model, Xte_fx)

        return float(evaluator.score(yte, y_pred))

    def _score_once_kfold(self, X: np.ndarray, y: np.ndarray, seed_base: int) -> float:
        """
        For CV, compute a single scalar per shuffle: the mean of fold scores.
        Each fold uses a deterministic child seed derived from the shuffle index.
        """
        eval_kind = self._get_eval_kind()
        evaluator = make_evaluator(self.cfg.eval, kind=eval_kind)
        fold_scores: list[float] = []

        # Recreate a fresh splitter for this shuffle (generators are one-shot)
        splitter_master = make_splitter(self.cfg.split, seed=seed_base)

        # Iterate folds
        for fold_id, split in enumerate(splitter_master.split(X, y), start=1):
            Xtr, Xte, ytr, yte = split.Xtr, split.Xte, split.ytr, split.yte
            # Derive a fold-specific seed so feature/model randomness is stable
            fold_seed = self.rngm.child_seed(f"cv/fold{fold_id}@{seed_base}")

            scaler     = make_scaler(self.cfg.scale)
            features   = make_features(self.cfg.features, seed=fold_seed, model_cfg=self.cfg.model, eval_cfg=self.cfg.eval)
            model_bld  = make_model(self.cfg.model)
            trainer    = make_trainer()
            predictor  = make_predictor()

            Xtr_s, Xte_s = scaler.fit_transform(Xtr, Xte)
            _, Xtr_fx, Xte_fx = features.fit_transform_train_test(Xtr_s, Xte_s, ytr)

            model = model_bld.make_estimator()
            model = trainer.fit(model, Xtr_fx, ytr)
            y_pred = predictor.predict(model, Xte_fx)
            fold_scores.append(float(evaluator.score(yte, y_pred)))

        # Mean score across folds for this shuffle
        return float(np.mean(fold_scores)) if len(fold_scores) else float("nan")

    def run(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Runs n_shuffles label permutations and returns an array of length n_shuffles.
        - Hold-out: each element is the hold-out score for that shuffle.
        - K-fold:   each element is the mean CV score across folds for that shuffle.
        Progress registry is injected via self._progress and self.progress_id.
        """
        n_shuffles = int(getattr(self, "_progress_total", 0) or 0)
        progress_id: Optional[str] = getattr(self, "progress_id", None)
        PROGRESS = getattr(self, "_progress", None)  # injected by service

        if n_shuffles < 1:
            raise ValueError("n_shuffles must be >= 1")

        use_kfold = getattr(self.cfg.split, "mode", "").lower() == "kfold"

        if PROGRESS and progress_id:
            PROGRESS.init(progress_id, total=n_shuffles, label=f"Shuffling 0/{n_shuffles}…")

        scores = np.empty(n_shuffles, dtype=float)
        try:
            for i in range(n_shuffles):
                # Per-shuffle RNG derivations
                lbl_gen   = self.rngm.child_generator(f"shuffle_{i}/labels")
                pipe_seed = self.rngm.child_seed(f"shuffle_{i}/pipeline")

                # Shuffle labels
                y_shuf = shuffle_simple_vector(y, rng=lbl_gen)

                # Compute score depending on split mode
                if use_kfold:
                    score_i = self._score_once_kfold(X, y_shuf, seed_base=pipe_seed)
                else:
                    score_i = self._score_once_holdout(X, y_shuf, seed=pipe_seed)

                scores[i] = score_i

                if PROGRESS and progress_id:
                    PROGRESS.update(progress_id, current=i + 1, label=f"Shuffling {i+1}/{n_shuffles}…")

            return scores
        finally:
            if PROGRESS and progress_id:
                PROGRESS.finalize(progress_id, label="Finalizing…")
