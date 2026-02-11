from __future__ import annotations

from typing import Optional

import numpy as np

from engine.factories.pipeline_factory import make_pipeline
from engine.reporting.decoder.decoder_outputs import compute_decoder_outputs
from engine.use_cases._splits import unpack_split

from .types import FoldRunOutputs


def run_supervised_folds(
    *,
    cfg,
    X,
    y,
    splitter,
    evaluator,
    eval_kind: str,
    mode: str,
    rngm,
    decoder_cfg,
    decoder_enabled: bool,
    decoder_positive_label,
    decoder_include_scores: bool,
    decoder_include_probabilities: bool,
    decoder_calibrate_probabilities: bool,
) -> FoldRunOutputs:
    """Fit/evaluate pipelines per split and collect fold-wise artifacts."""

    out = FoldRunOutputs()

    for fold_id, split in enumerate(splitter.split(X, y), start=1):
        ns = unpack_split(split)
        Xtr, Xte, ytr, yte = ns.Xtr, ns.Xte, ns.ytr, ns.yte
        idx_te = ns.idx_te

        pipeline = make_pipeline(cfg, rngm, stream=f"{mode}/fold{fold_id}")
        pipeline.fit(Xtr, ytr)
        y_pred = pipeline.predict(Xte)
        out.last_pipeline = pipeline

        # indices for re-ordering if available
        if idx_te is not None:
            out.test_indices_parts.append(np.asarray(idx_te, dtype=int).ravel())
        else:
            out.test_indices_parts.append(None)

        # fold ids per row
        try:
            n_fold_rows = int(np.asarray(y_pred).shape[0])
        except Exception:
            n_fold_rows = int(np.asarray(Xte).shape[0])
        out.eval_fold_ids_parts.append(np.full((n_fold_rows,), fold_id, dtype=int))

        metric_name = cfg.eval.metric
        y_proba = None
        y_score = None

        if eval_kind == "classification":
            if hasattr(pipeline, "predict_proba"):
                y_proba = pipeline.predict_proba(Xte)
            if hasattr(pipeline, "decision_function"):
                y_score = pipeline.decision_function(Xte)

            from engine.components.evaluation.scoring import PROBA_METRICS

            if metric_name in PROBA_METRICS and y_proba is None and y_score is None:
                raise ValueError(
                    f"Metric '{metric_name}' requires predict_proba or decision_function, "
                    f"but estimator {type(pipeline).__name__} has neither."
                )

        score_val = evaluator.score(yte, y_pred=y_pred, y_proba=y_proba, y_score=y_score)
        out.fold_scores.append(float(score_val))

        out.y_true_parts.append(np.asarray(yte))
        out.y_pred_parts.append(np.asarray(y_pred))
        if y_proba is not None:
            out.y_proba_parts.append(np.asarray(y_proba))
        if y_score is not None:
            out.y_score_parts.append(np.asarray(y_score))

        # --- decoder outputs (classification only) --------------------------
        if decoder_enabled and eval_kind == "classification":
            out.decoder.fold_ids_parts.append(np.full((n_fold_rows,), fold_id, dtype=int))
            try:
                dec = compute_decoder_outputs(
                    pipeline,
                    Xte,
                    positive_class_label=decoder_positive_label,
                    include_decision_scores=decoder_include_scores,
                    include_probabilities=decoder_include_probabilities,
                    calibrate_probabilities=decoder_calibrate_probabilities,
                )
                if out.decoder.classes is None and dec.classes is not None:
                    out.decoder.classes = np.asarray(dec.classes)
                if out.decoder.positive_index is None and dec.positive_class_index is not None:
                    out.decoder.positive_index = int(dec.positive_class_index)

                out.decoder.scores_parts.append(
                    np.asarray(dec.decision_scores) if dec.decision_scores is not None else None
                )
                out.decoder.proba_parts.append(
                    np.asarray(dec.proba) if dec.proba is not None else None
                )
                out.decoder.margin_parts.append(
                    np.asarray(dec.margin) if dec.margin is not None else None
                )
                if dec.notes:
                    out.decoder.notes.extend([str(x) for x in dec.notes])
            except Exception as e:
                out.decoder.scores_parts.append(None)
                out.decoder.proba_parts.append(None)
                out.decoder.margin_parts.append(None)
                out.decoder.notes.append(
                    f"decoder outputs failed on fold {fold_id}: {type(e).__name__}: {e}"
                )

        out.n_train_sizes.append(int(np.asarray(Xtr).shape[0]))
        out.n_test_sizes.append(int(np.asarray(Xte).shape[0]))

    return out
