from __future__ import annotations

"""Supervised training orchestration (Engine use-case).

This package refactors the former monolithic ``engine.use_cases.supervised_training``
module into small units:

- folds: fit/evaluate per split + out-of-fold decoder extraction
- aggregate: pooling + confusion/ROC/regression diagnostics
- decoder: building decoder payloads for the result contract
- artifacts: artifact meta + persistence + in-process caches

Correctness requirement
-----------------------
For k-fold CV, decoder outputs must be computed **out-of-fold** (OOF): each fold
model predicts on that fold's test split, and outputs are pooled. We must **not**
compute decoder outputs from a final refit model trained on all data.
"""

from typing import Any, Dict, Optional

import numpy as np

from engine.contracts.results.training import TrainResult
from engine.contracts.run_config import RunConfig

from engine.factories.baseline_factory import make_baseline
from engine.factories.data_loading_factory import make_data_loader
from engine.factories.eval_factory import make_evaluator
from engine.factories.metrics_factory import make_metrics_computer
from engine.factories.pipeline_factory import make_pipeline
from engine.factories.sanity_factory import make_sanity_checker
from engine.factories.split_factory import make_splitter
from engine.runtime.random.rng import RngManager
from engine.core.progress import ProgressCallback
from engine.use_cases._deps import resolve_seed

from .aggregate import pool_eval_outputs, compute_eval_payloads
from .artifacts import attach_artifact_and_persist
from .decoder import build_decoder_outputs
from .folds import run_supervised_folds


def train_supervised(
    run_config: RunConfig,
    *,
    store: Any = None,
    rng: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> TrainResult:
    cfg = run_config

    # --- Load data ---------------------------------------------------------
    loader = make_data_loader(cfg.data)
    X, y = loader.load()

    # --- Checks ------------------------------------------------------------
    make_sanity_checker().check(X, y)

    # --- RNG ----------------------------------------------------------------
    seed = resolve_seed(int(rng) if rng is not None else getattr(cfg.eval, "seed", None), fallback=0)
    rngm = RngManager(seed)

    # --- Split -------------------------------------------------------------
    split_seed = rngm.child_seed("train/split")
    mode = getattr(cfg.split, "mode")
    if mode not in ("holdout", "kfold"):
        raise ValueError(f"Unsupported split mode: {mode!r}")

    splitter = make_splitter(cfg.split, seed=split_seed)

    # --- Eval kind ---------------------------------------------------------
    model_task = getattr(cfg.model, "task", None)
    eval_kind = "regression" if model_task == "regression" else "classification"

    if eval_kind == "regression" and hasattr(cfg.split, "stratified"):
        # historical behavior
        cfg.split.stratified = False

    evaluator = make_evaluator(cfg.eval, kind=eval_kind)
    metrics_computer = make_metrics_computer(kind=eval_kind)

    # --- Decoder config ----------------------------------------------------
    decoder_cfg = getattr(cfg.eval, "decoder", None)
    decoder_enabled = bool(getattr(decoder_cfg, "enabled", False)) if decoder_cfg is not None else False
    decoder_positive_label = getattr(decoder_cfg, "positive_class_label", None) if decoder_cfg is not None else None
    decoder_include_scores = bool(getattr(decoder_cfg, "include_decision_scores", True)) if decoder_cfg is not None else True
    decoder_include_probabilities = bool(getattr(decoder_cfg, "include_probabilities", True)) if decoder_cfg is not None else True
    decoder_calibrate_probabilities = bool(getattr(decoder_cfg, "calibrate_probabilities", False)) if decoder_cfg is not None else False

    # --- Fit/evaluate per split --------------------------------------------
    fold_out = run_supervised_folds(
        cfg=cfg,
        X=X,
        y=y,
        splitter=splitter,
        evaluator=evaluator,
        eval_kind=eval_kind,
        mode=mode,
        rngm=rngm,
        decoder_cfg=decoder_cfg,
        decoder_enabled=decoder_enabled,
        decoder_positive_label=decoder_positive_label,
        decoder_include_scores=decoder_include_scores,
        decoder_include_probabilities=decoder_include_probabilities,
        decoder_calibrate_probabilities=decoder_calibrate_probabilities,
    )

    if fold_out.last_pipeline is None:
        raise ValueError("No folds produced by splitter; cannot train model.")

    # --- Optional refit for kfold ------------------------------------------
    if mode == "kfold":
        refit_pipeline = make_pipeline(cfg, rngm, stream="kfold/refit")
        refit_pipeline.fit(X, y)
        effective_pipeline = refit_pipeline
    else:
        effective_pipeline = fold_out.last_pipeline

    # --- Summary scores ----------------------------------------------------
    if not fold_out.fold_scores:
        fold_out.fold_scores = [float("nan")]

    mean_score = float(np.mean(fold_out.fold_scores))
    std_score = float(np.std(fold_out.fold_scores))

    # --- Pool outputs + compute payloads -----------------------------------
    pooled = pool_eval_outputs(
        y_true_parts=fold_out.y_true_parts,
        y_pred_parts=fold_out.y_pred_parts,
        y_proba_parts=fold_out.y_proba_parts,
        y_score_parts=fold_out.y_score_parts,
        test_indices_parts=fold_out.test_indices_parts,
        eval_fold_ids_parts=fold_out.eval_fold_ids_parts,
    )

    confusion_out, roc_payload, regression_payload = compute_eval_payloads(
        eval_kind=eval_kind,
        pooled=pooled,
        metrics_computer=metrics_computer,
        cfg=cfg,
    )

    # --- Decoder payload ----------------------------------------------------
    decoder_payload = build_decoder_outputs(
        decoder_cfg=decoder_cfg,
        mode=mode,
        eval_kind=eval_kind,
        y_true_all=pooled.y_true,
        y_pred_all=pooled.y_pred,
        row_indices=pooled.row_indices,
        order=pooled.order,
        fold_ids_all=pooled.fold_ids,
        decoder_parts=fold_out.decoder,
        positive_class_label=decoder_positive_label,
    )

    # --- Fold stats ---------------------------------------------------------
    n_train_avg = int(np.round(np.mean(fold_out.n_train_sizes))) if fold_out.n_train_sizes else 0
    n_test_avg = int(np.round(np.mean(fold_out.n_test_sizes))) if fold_out.n_test_sizes else 0

    if mode == "holdout":
        fold_scores_out = None
        n_splits_out = 1
        n_train_out = fold_out.n_train_sizes[0] if fold_out.n_train_sizes else n_train_avg
        n_test_out = fold_out.n_test_sizes[0] if fold_out.n_test_sizes else n_test_avg
    else:
        fold_scores_out = fold_out.fold_scores
        n_splits_out = len(fold_out.fold_scores)
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
        "n_train": n_train_out,
        "n_test": n_test_out,
        "notes": [],
    }

    if decoder_payload is not None:
        result["decoder_outputs"] = decoder_payload
    # --- Shuffle baseline --------------------------------------------------
    # Some callers may choose to compute the shuffle baseline in a separate step
    # (e.g., to stream progress updates). When ``progress_id`` is present, we treat
    # that as a signal that baseline execution/progress may be handled externally
    # and therefore skip running it here to avoid duplicate work/notes.
    n_shuffles = int(getattr(cfg.eval, "n_shuffles", 0) or 0)
    progress_id = getattr(cfg.eval, "progress_id", None)

    if n_shuffles > 0 and not progress_id:
        baseline = make_baseline(cfg, rngm=rngm)
        try:
            scores = np.asarray(
                baseline.run(X, y, n_shuffles=n_shuffles, progress=progress),
                dtype=float,
            ).ravel()

            # Compare baseline against the trained model's mean score.
            ref_score_f = float(mean_score)
            ge = int(np.sum(scores >= ref_score_f))
            p_val = (ge + 1.0) / (scores.size + 1.0) if scores.size else float("nan")

            result["shuffled_scores"] = [float(v) for v in scores.tolist()]
            result["p_value"] = float(p_val)
            result["notes"].append(
                f"Shuffle baseline: mean={float(np.mean(scores)):.4f} ± {float(np.std(scores)):.4f}, p≈{float(p_val):.4f}"
            )
        except Exception as e:
            # Baseline errors must not break training.
            result["notes"].append(f"Shuffle baseline failed: {type(e).__name__}: {e}")

    # --- Artifact meta + persistence/caching -------------------------------
    attach_artifact_and_persist(
        cfg=cfg,
        pipeline=effective_pipeline,
        X=X,
        eval_kind=eval_kind,
        n_train=n_train_out,
        n_test=n_test_out,
        result=result,
        y_true=pooled.y_true,
        y_pred=pooled.y_pred,
        row_indices=pooled.row_indices,
        fold_ids=pooled.fold_ids,
        store=store,
    )

    return TrainResult.model_validate(result)
