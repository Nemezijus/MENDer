from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from engine.components.interfaces import BaselineRunner
from engine.core.progress import ProgressCallback
from engine.core.random.rng import RngManager
from engine.core.random.shuffle import shuffle_simple_vector


ScoreOnceFn = Callable[[np.ndarray, np.ndarray, int], float]


@dataclass
class LabelShuffleBaseline(BaselineRunner):
    """Label-shuffle baseline runner (component).

    This component is intentionally narrow in scope:
    - It implements the *baseline-specific computation* (shuffle labels N times).
    - It calls an injected ``score_once`` function for each shuffle.

    The injected scorer is responsible for orchestration such as:
    - building splitters / pipelines
    - fitting / predicting
    - computing the scalar evaluation metric

    That orchestration lives in :mod:`engine.use_cases.baselines.label_shuffle`.
    """

    rngm: RngManager
    score_once: ScoreOnceFn
    n_shuffles_default: int = 0

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        n_shuffles: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> np.ndarray:
        """Run the label-shuffle baseline.

        Returns
        -------
        np.ndarray
            Array of length ``n_shuffles`` (one score per shuffle).
        """

        n_shuffles_i = int(n_shuffles if n_shuffles is not None else int(self.n_shuffles_default or 0))
        if n_shuffles_i < 1:
            raise ValueError("n_shuffles must be >= 1")

        if progress:
            progress.init(total=n_shuffles_i, label=f"Shuffling 0/{n_shuffles_i}…")

        scores = np.empty(n_shuffles_i, dtype=float)
        try:
            for i in range(n_shuffles_i):
                # Deterministic substreams per shuffle
                lbl_gen = self.rngm.child_generator(f"shuffle_{i}/labels")
                pipe_seed = int(self.rngm.child_seed(f"shuffle_{i}/pipeline"))

                y_shuf = shuffle_simple_vector(y, rng=lbl_gen)
                scores[i] = float(self.score_once(X, y_shuf, pipe_seed))

                if progress:
                    progress.update(current=i + 1, label=f"Shuffling {i+1}/{n_shuffles_i}…")

            return scores
        finally:
            if progress:
                progress.finalize(label="Finalizing…")
