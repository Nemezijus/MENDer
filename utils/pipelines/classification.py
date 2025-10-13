# utils/pipelines/classification.py
from __future__ import annotations

from typing import Any, Dict, Tuple, Union, Optional, Literal
import numpy as np
from sklearn.metrics import classification_report

from utils.preprocessing.general.trial_split import split
from utils.preprocessing.general.feature_scaling import scale_train_test
from utils.processing.fitting import fit_model
from utils.postprocessing.predicting import predict_labels
from utils.postprocessing.scoring import score
from utils.preprocessing.general.feature_extraction.pca import pca_fit_transform_train_test

FeatureScaling = Literal["none", "pca"]  # extend later: "lda", etc.


def train_and_score_classifier(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    train_frac: float = 0.8,
    scale: Optional[str] = "standard",          # 'standard'|'robust'|'minmax'|'maxabs'|'quantile'|'none'|None
    feature_scaling: FeatureScaling = "none",   # 'none'|'pca'
    # PCA options (used only if feature_scaling == 'pca')
    pca_n_components: Optional[int] = None,     # None => auto by variance threshold
    pca_variance_threshold: float = 0.95,
    pca_whiten: bool = False,
    rng: Union[None, int, np.random.Generator] = None,
    metric: str = "accuracy",
    debug: bool = True,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, Any]]]:
    """
    Generic classification pipeline:
      1) stratified split
      2) optional scaling (fit on train, apply to test)
      3) optional feature extraction (e.g., PCA) on train, apply to test
      4) fit provided `model`
      5) predict labels
      6) compute a metric (default: accuracy)

    RNG policy:
      - We DO NOT draw any extra numbers here.
      - The provided `rng` (int | Generator | None) is passed directly to the splitter.
      - If PCA is enabled and `rng` is an int, we pass it to sklearn as `random_state`.
        If `rng` is a Generator, we use `random_state=None` (sklearn will be deterministic
        for PCA when `svd_solver='full'`; if you later switch to a randomized solver,
        send an int instead).
    """

    # 1) Split (stratified) — pass rng straight through (no extra draws)
    X_train, X_test, y_train, y_test = split(
        X, y, train_frac=train_frac, custom=False, rng=rng
    )

    # 2) Optional scaling
    if scale and str(scale).lower() != "none":
        X_train, X_test = scale_train_test(X_train, X_test, method=str(scale).lower())

    # 3) Optional feature extraction (currently PCA)
    pca = None
    if feature_scaling == "pca":
        # Deterministic when rng is an int; if it's a Generator, keep None to avoid consuming it
        pca_rs = None if isinstance(rng, np.random.Generator) else rng
        pca, X_train, X_test = pca_fit_transform_train_test(
            X_train, X_test,
            n_components=pca_n_components,
            variance_threshold=pca_variance_threshold,
            whiten=pca_whiten,
            random_state=pca_rs,
        )
    elif feature_scaling != "none":
        raise ValueError(
            f"Unknown feature_scaling option '{feature_scaling}'. Use 'none' or 'pca'."
        )

    # 4) Fit
    model = fit_model(model, X_train, y_train)

    # 5) Predict
    y_pred = predict_labels(model, X_test)

    # 6) Score
    score_val = score(
        y_test, y_pred,
        kind="classification",
        metric=metric,
    )

    if debug:
        print(
            f"[INFO] {metric} = {score_val:.3f}  "
            f"(train_frac={train_frac}, scale={scale}, features={feature_scaling}"
            + (f", pca_n={pca.n_components_}" if pca is not None else "")
            + ")"
        )
        print(classification_report(y_test, y_pred, digits=3))

    if return_details:
        details = {
            "model": model,
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "y_pred": y_pred,
            "scale": scale,
            "feature_scaling": feature_scaling,
            "pca": pca,
        }
        return score_val, details

    return score_val
