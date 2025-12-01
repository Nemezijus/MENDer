from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Any
import numpy as np

from sklearn.preprocessing import (
    StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer
)

from shared_schemas.scale_configs import ScaleModel
from utils.strategies.interfaces import Scaler
from utils.preprocessing.general.feature_scaling import scale_train_test

@dataclass
class PairScaler(Scaler):
    """
    Strategy that (a) exposes an sklearn transformer via `make_transformer()`,
    and (b) preserves your existing `fit_transform(Xtr, Xte)` convenience.

    Use `make_transformer()` when assembling a scikit-learn Pipeline.
    Use `fit_transform()` in the current orchestrator (no pipeline).
    """
    cfg: ScaleModel

    # ---------- sklearn step accessor ----------
    def make_transformer(self) -> Any:
        method = (self.cfg.method or "none").lower()

        if method == "standard":
            return StandardScaler(with_mean=True, with_std=True, copy=True)

        if method == "robust":
            # Tukey's IQR by default (25–75)
            return RobustScaler(with_centering=True, with_scaling=True,
                                quantile_range=(25.0, 75.0), copy=True)

        if method == "minmax":
            return MinMaxScaler(copy=True)

        if method == "maxabs":
            return MaxAbsScaler(copy=True)

        if method == "quantile":
            output_dist = getattr(self.cfg, "quantile_output_distribution", "normal")
            n_quantiles = int(getattr(self.cfg, "quantile_n", 1000))
            subsample   = int(getattr(self.cfg, "quantile_subsample", 1e5))
            return QuantileTransformer(
                output_distribution=output_dist,
                n_quantiles=n_quantiles,
                subsample=subsample,
                copy=True,
                random_state=0,  # deterministic tie-breaking
            )

        if method == "none":
            # Pipelines accept the literal string 'passthrough'
            return "passthrough"

        raise ValueError(f"Unknown scaler method: {self.cfg.method!r}")

    # ---------- Existing API (kept for backwards compatibility) ----------
    def fit_transform(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Keep using your proven helper for the non-pipeline path.
        Alternatively, we could call `make_transformer()` and .fit/.transform here.
        """
        return scale_train_test(
            X_train, X_test,
            method=self.cfg.method,
            # If you expose quantile options in ScaleModel, pass them through here too.
            # e.g., quantile_output_distribution=self.cfg.quantile_output_distribution, ...
        )
