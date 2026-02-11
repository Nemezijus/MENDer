from __future__ import annotations

"""Ensemble training orchestration.

Refactors the former monolithic ``engine.use_cases.ensembles`` module into a
package with SRP-oriented submodules:

- folds: fit + evaluate per split
- aggregate: pool/reorder outputs + compute confusion/roc/regression payloads
- decoder: decoder state + decoder payload building
- reports: ensemble report accumulator setup + finalization
- artifacts: artifact meta + caching + persistence
"""

from typing import Any, Dict, Optional

import numpy as np

from engine.contracts.ensemble_run_config import EnsembleRunConfig
from engine.contracts.results.ensemble import EnsembleResult
from engine.io.artifacts.store import ArtifactStore
from engine.factories.data_loading_factory import make_data_loader
from engine.factories.sanity_factory import make_sanity_checker
from engine.factories.split_factory import make_splitter
from engine.factories.eval_factory import make_evaluator
from engine.factories.metrics_factory import make_metrics_computer
from engine.factories.ensemble_factory import make_ensemble_strategy
from engine.runtime.random.rng import RngManager
from engine.use_cases._deps import resolve_seed

from .aggregate import pool_eval_outputs, compute_eval_payloads
from .artifacts import attach_artifact_and_persist
from .decoder import build_decoder_payload, init_decoder_state
from .folds import run_ensemble_folds
from .reports import finalize_ensemble_report, init_report_state
from .types import FoldState


def train_ensemble(
    cfg: EnsembleRunConfig,
    *,
    store: Optional[ArtifactStore] = None,
    rng: Optional[int] = None,
) -> EnsembleResult:
    # --- Load data -----------------------------------------------------------
    try:
        loader = make_data_loader(cfg.data)
        X, y = loader.load()
    except Exception as e:
        raise ValueError(str(e))

    # --- Checks --------------------------------------------------------------
    make_sanity_checker().check(X, y)

    # --- RNG -----------------------------------------------------------------
    seed = resolve_seed(getattr(cfg.eval, "seed", None), fallback=0) if rng is None else int(rng)
    rngm = RngManager(seed)

    # --- Split mode ----------------------------------------------------------
    try:
        mode = getattr(cfg.split, "mode")
    except AttributeError:
        raise ValueError("EnsembleRunConfig.split is missing required 'mode' attribute")

    if mode not in ("holdout", "kfold"):
        raise ValueError(f"Unsupported split mode in ensemble train service: {mode!r}")

    # Decide evaluation kind based on ensemble strategy (authoritative)
    ensemble_strategy = make_ensemble_strategy(cfg)
    eval_kind = ensemble_strategy.expected_kind()

    metrics_computer = make_metrics_computer(kind=eval_kind)

    # For regression, stratification does not make sense -> force it off defensively
    if eval_kind == "regression" and hasattr(cfg.split, "stratified"):
        cfg.split.stratified = False

    split_seed = rngm.child_seed("ensemble/train/split")
    splitter = make_splitter(cfg.split, seed=split_seed)

    evaluator = make_evaluator(cfg.eval, kind=eval_kind)

    fold_state = FoldState()
    report_state = init_report_state(cfg)
    decoder_state, decoder_cfg = init_decoder_state(cfg, eval_kind=eval_kind)

    # --- Fit/eval over splits ------------------------------------------------
    last_model = run_ensemble_folds(
        cfg=cfg,
        X=X,
        y=y,
        splitter=splitter,
        ensemble_strategy=ensemble_strategy,
        evaluator=evaluator,
        eval_kind=eval_kind,
        mode=mode,
        rngm=rngm,
        fold_state=fold_state,
        decoder_state=decoder_state,
        report_state=report_state,
    )

    # If kfold, refit on full data for persisted artifact.
    if mode == "kfold":
        last_model = ensemble_strategy.fit(X, y, rngm=rngm, stream=f"{mode}/refit")

    pooled = pool_eval_outputs(
        y_true_parts=fold_state.y_true_parts,
        y_pred_parts=fold_state.y_pred_parts,
        y_proba_parts=fold_state.y_proba_parts,
        y_score_parts=fold_state.y_score_parts,
        test_indices_parts=fold_state.test_indices_parts,
        eval_fold_ids_parts=fold_state.eval_fold_ids_parts,
    )

    fold_scores = fold_state.fold_scores
    mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
    std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0

    confusion_out, roc_payload, regression_payload = compute_eval_payloads(
        eval_kind=eval_kind,
        pooled=pooled,
        metrics_computer=metrics_computer,
        cfg=cfg,
    )

    # Decoder outputs (optional)
    decoder_payload = build_decoder_payload(
        decoder_state=decoder_state,
        decoder_cfg=decoder_cfg,
        eval_kind=eval_kind,
        mode=mode,
        y_true_all=pooled.y_true,
        y_pred_all=pooled.y_pred,
        row_indices=pooled.row_indices,
        order=pooled.order,
        fold_ids_all=pooled.fold_ids,
    )

    # Regression decoder payload is handled in decoder module.

    # Ensemble report
    ensemble_report = finalize_ensemble_report(report_state)

    n_train_avg = int(np.round(np.mean(fold_state.n_train_sizes))) if fold_state.n_train_sizes else 0
    n_test_avg = int(np.round(np.mean(fold_state.n_test_sizes))) if fold_state.n_test_sizes else 0

    if mode == "holdout":
        fold_scores_out = None
        n_splits_out = 1
        n_train_out = fold_state.n_train_sizes[0] if fold_state.n_train_sizes else n_train_avg
        n_test_out = fold_state.n_test_sizes[0] if fold_state.n_test_sizes else n_test_avg
    else:
        fold_scores_out = fold_scores
        n_splits_out = len(fold_scores)
        n_train_out = n_train_avg
        n_test_out = n_test_avg

    result: Dict[str, Any] = {
        "metric_name": cfg.eval.metric,
        "fold_scores": fold_scores_out,
        "mean_score": mean_score,
        "std_score": std_score,
        "n_splits": n_splits_out,
        "metric_value": mean_score,
        "confusion": confusion_out,
        "roc": roc_payload,
        "regression": regression_payload,
        "n_train": int(n_train_out),
        "n_test": int(n_test_out),
        "notes": [],
    }

    if decoder_payload is not None:
        result["decoder_outputs"] = decoder_payload

    if ensemble_report is not None:
        result["ensemble_report"] = ensemble_report

    # Persist artifact + cache
    attach_artifact_and_persist(
        cfg=cfg,
        model=last_model,
        X=X,
        eval_kind=eval_kind,
        n_train=int(n_train_out),
        n_test=int(n_test_out),
        result=result,
        y_true=pooled.y_true,
        y_pred=pooled.y_pred,
        row_indices=pooled.row_indices,
        fold_ids=pooled.fold_ids,
        ensemble_report=ensemble_report if isinstance(ensemble_report, dict) else None,
        store=store,
    )

    return EnsembleResult.model_validate(result)
