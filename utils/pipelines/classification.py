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

    Parameters
    ----------
    model : sklearn-like estimator
        Must expose `.fit(X, y)` and `.predict(X)`.
    X, y : arrays
        Data and labels. X shape (n_samples, n_features).
    train_frac : float
        Fraction for the train split (stratified).
    scale : str or None
        Scaling method (see utils.preprocessing.feature_scaling). None/'none' disables.
    feature_scaling : {'none','pca'}
        Feature extraction step after scaling.
    pca_n_components : int or None
        If None, auto-pick by `pca_variance_threshold` (train-only).
    pca_variance_threshold : float
        Cumulative variance to retain when auto-selecting PCA components.
    pca_whiten : bool
        Whether to whiten PCs.
    rng : int | numpy.random.Generator | None
        Seed or generator for split reproducibility.
    metric : str
        Passed to utils.postprocessing.scoring.score (e.g., 'accuracy').
    debug : bool
        Print summary + sklearn classification_report.
    return_details : bool
        If True, also return dict with intermediate artifacts.

    Returns
    -------
    float
        The requested metric.
    or (float, dict)
        If return_details=True, includes model, splits, transforms, etc.
    """

    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    child_for_split = int(gen.integers(1 << 32))
    # 1) Split (stratified)
    X_train, X_test, y_train, y_test = split(X, y, train_frac=train_frac, custom=False, rng=child_for_split)

    # 2) Optional scaling
    if scale and str(scale).lower() != "none":
        X_train, X_test = scale_train_test(X_train, X_test, method=str(scale).lower())

    # 3) Optional feature extraction (currently PCA)
    pca = None
    if feature_scaling == "pca":
        pca, X_train, X_test = pca_fit_transform_train_test(
            X_train, X_test,
            n_components=pca_n_components,
            variance_threshold=pca_variance_threshold,
            whiten=pca_whiten,
            random_state=None if isinstance(rng, np.random.Generator) else rng,
        )
    elif feature_scaling != "none":
        raise ValueError(f"Unknown feature_scaling option '{feature_scaling}'. Use 'none' or 'pca'.")

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