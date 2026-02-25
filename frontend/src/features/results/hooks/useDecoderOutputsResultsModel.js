import { useMemo, useState } from 'react';

import { exportDecoderOutputs } from '../../modelArtifacts/api/modelsApi.js';
import { downloadBlob } from '../../../shared/utils/download.js';
import { toErrorText } from '../../../shared/utils/errors.js';

import { pickColumns, buildCsv } from '../utils/tables.js';
import { parseNumber, rangeText } from '../utils/formatNumbers.js';
import { mean, median } from '../utils/stats.js';
import { detectSplitType, firstKey, getSummary } from '../utils/decoderOutputs.js';

export function useDecoderOutputsResultsModel(trainResult) {
  const decoder = trainResult?.decoder_outputs || trainResult?.decoderOutputs || null;
  const preview = decoder?.preview_rows || decoder?.previewRows || [];

  const ready = Boolean(decoder && Array.isArray(preview) && preview.length > 0);

  const evalKind = trainResult?.artifact?.kind || (trainResult?.regression ? 'regression' : 'classification');
  const isRegression = evalKind === 'regression';

  const classes = decoder?.classes || [];
  const hasDecisionScores = Boolean(decoder?.has_decision_scores ?? decoder?.hasDecisionScores);
  const hasProbabilities = Boolean(
    decoder?.has_proba ?? decoder?.hasProbabilities ?? decoder?.has_probabilities ?? decoder?.hasProba,
  );

  const probaSource = decoder?.proba_source ?? decoder?.probaSource ?? null;
  const isVoteShare = String(probaSource || '').toLowerCase() === 'vote_share';

  const summary = useMemo(() => (decoder ? getSummary(decoder) : null), [decoder]);

  const splitType = useMemo(() => detectSplitType(trainResult), [trainResult]);
  const isKfold = splitType === 'kfold';
  const isHoldout = splitType === 'holdout';
  const nSplits =
    (typeof trainResult?.n_splits === 'number' ? trainResult.n_splits : null) ??
    (typeof trainResult?.artifact?.n_splits === 'number' ? trainResult.artifact.n_splits : null) ??
    null;

  const decoderNotes = useMemo(() => {
    const dnRaw = Array.isArray(decoder?.notes) ? decoder.notes : [];
    const tn = Array.isArray(trainResult?.notes) ? trainResult.notes : [];

    const dn = dnRaw.filter((n) => !tn.includes(n) && !tn.includes(`Decoder outputs: ${n}`));

    const hasStructuredSummary = summary && typeof summary === 'object';
    const filtered = hasStructuredSummary
      ? dn.filter((n) => !String(n).toLowerCase().startsWith('decoder summary:'))
      : dn;

    return filtered.filter((n) => String(n).trim().length > 0);
  }, [decoder?.notes, trainResult?.notes, summary]);

  const classOptions = useMemo(() => {
    if (!Array.isArray(classes)) return [];
    return classes.map((c) => ({ value: String(c), label: String(c) }));
  }, [classes]);

  const defaultSelected =
    decoder?.positive_class_label != null
      ? String(decoder.positive_class_label)
      : classOptions.length
        ? classOptions[0].value
        : null;

  const [selectedClass, setSelectedClass] = useState(defaultSelected);
  const selectedProbKey = selectedClass ? `p_${selectedClass}` : null;

  const probStats = useMemo(() => {
    if (!hasProbabilities || !selectedProbKey) return { mean: null, median: null, n: 0 };
    const vals = preview
      .map((r) => parseNumber(r?.[selectedProbKey]))
      .filter((v) => v !== null);
    return { mean: mean(vals), median: median(vals), n: vals.length };
  }, [hasProbabilities, preview, selectedProbKey]);

  const columns = useMemo(() => pickColumns(preview), [preview]);

  const handleExportPreview = () => {
    const csv = buildCsv(preview, columns);
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
    downloadBlob(blob, 'decoder_outputs_preview.csv');
  };

  const [isExportingFull, setIsExportingFull] = useState(false);

  const handleExportFull = async () => {
    const artifactUid = trainResult?.artifact?.uid;
    if (!artifactUid) return;

    try {
      setIsExportingFull(true);
      const { blob, filename } = await exportDecoderOutputs({
        artifactUid,
        filename: 'decoder_outputs.csv',
      });
      downloadBlob(blob, filename || 'decoder_outputs.csv');
    } catch (e) {
      window.alert(`Export failed: ${toErrorText(e) || 'Could not export decoder outputs.'}`);
    } finally {
      setIsExportingFull(false);
    }
  };

  const previewN = Array.isArray(preview) ? preview.length : 0;
  const totalN =
    typeof decoder?.n_rows_total === 'number' && Number.isFinite(decoder.n_rows_total)
      ? decoder.n_rows_total
      : null;

  const marginFracKey = summary ? firstKey(summary, ['margin_frac_lt_0_1', 'margin_frac_lt_0_05']) : null;
  const maxProbaFracKey = summary ? firstKey(summary, ['max_proba_frac_ge_0_9', 'max_proba_frac_ge_0_8']) : null;

  const eceBins = summary && Array.isArray(summary.reliability_bins) ? summary.reliability_bins : null;
  const nonEmptyBins = useMemo(() => {
    if (!eceBins) return [];
    return eceBins.filter((b) => b && typeof b.count === 'number' && b.count > 0);
  }, [eceBins]);

  const showCalibrationBins = !isRegression && nonEmptyBins.length > 0;
  const showClassSummary = !isRegression && hasProbabilities && classOptions.length > 0;

  const evalSamplesTooltip = isKfold
    ? `Number of evaluation samples pooled (concatenated) across ${nSplits || 'multiple'} folds using out-of-fold (OOF) predictions.`
    : isHoldout
      ? 'Number of samples in the held-out test split.'
      : 'Number of evaluation samples for this run.';

  const datasetTitleTip =
    'Quick context about the evaluation set used for decoder outputs. In k-fold CV this is typically pooled out-of-fold (OOF) predictions; in hold-out it is the held-out test split.';
  const regTitleTip = 'Regression summary computed from the full evaluation set (not just the preview).';
  const lossTitleTip =
    'Loss and calibration describe probability quality. Log loss and Brier measure probability error (smaller is better). ECE/MCE describe how well confidence matches accuracy (smaller is better; 0 is ideal).';
  const confTitleTip =
    'Confidence describes how decisive predictions are. Margin reflects top-1 vs runner-up separation (larger is typically more confident). Max probability is a confidence proxy (higher means more confident, not necessarily more accurate).';

  const dataParamsItems = [
    {
      key: 'Evaluation samples',
      value: summary?.n_samples ?? totalN ?? null,
      tooltip: evalSamplesTooltip,
    },
    {
      key: 'Number of classes',
      value: summary?.n_classes ?? (Array.isArray(classes) ? classes.length : null),
      tooltip: 'Number of unique class labels.',
    },
    {
      key: 'Calibration bins',
      value: summary?.ece_n_bins ?? null,
      tooltip:
        'Number of confidence bins used to compute calibration (ECE/MCE). More bins = finer detail, but fewer samples per bin.',
    },
  ];

  const lossCalItems = [
    {
      key: 'Log loss',
      value: summary?.log_loss ?? null,
      tooltip: 'Cross-entropy loss. Smaller is better. Minimum is 0; there is no fixed upper bound.',
    },
    {
      key: 'Brier score',
      value: summary?.brier ?? null,
      tooltip: 'Probability error (mean squared error vs one-hot truth). Smaller is better. Typical range is 0 to 1.',
    },
    {
      key: 'Expected calibration error (ECE)',
      value: summary?.ece ?? null,
      tooltip:
        'Calibration error using top-1 confidence. Smaller is better. Typical range is 0 to 1 (0 means perfectly calibrated).',
    },
    {
      key: 'Maximum calibration error (MCE)',
      value: summary?.mce ?? null,
      tooltip: 'Worst-bin calibration gap |accuracy − confidence|. Smaller is better. Typical range is 0 to 1.',
    },
  ];

  const confidenceItems = [
    {
      key: 'Margin (mean)',
      value: summary?.margin_mean ?? null,
      tooltip: 'Average top1−top2 separation (score or probability). Larger usually means more confident predictions.',
    },
    {
      key: 'Margin (median)',
      value: summary?.margin_median ?? null,
      tooltip: 'Median top1−top2 separation (score or probability). Larger usually means more confident predictions.',
    },
    marginFracKey
      ? {
          key: 'Low margin (fraction)',
          value: summary?.[marginFracKey] ?? null,
          format: 'pct',
          tooltip: 'Fraction of samples with small top1−top2 separation. Smaller is better (fewer uncertain predictions).',
        }
      : null,
    {
      key: 'Max probability (mean)',
      value: summary?.max_proba_mean ?? null,
      tooltip:
        'Average of max predicted probability per sample. Higher means the model is more confident (not necessarily more accurate). Range 0–1.',
    },
    maxProbaFracKey
      ? {
          key: 'High confidence (fraction)',
          value: summary?.[maxProbaFracKey] ?? null,
          format: 'pct',
          tooltip:
            'Fraction of samples with high max probability. Higher means more confident predictions (not necessarily more accurate).',
        }
      : null,
  ].filter(Boolean);

  const regPerfItems = [
    { key: 'RMSE', value: summary?.rmse ?? null, tooltip: 'Root mean squared error. Smaller is better.' },
    { key: 'MAE', value: summary?.mae ?? null, tooltip: 'Mean absolute error. Smaller is better.' },
    {
      key: 'Median AE',
      value: summary?.median_ae ?? summary?.median_abs_error ?? null,
      tooltip: 'Median absolute error (robust). Smaller is better.',
    },
    { key: 'R²', value: summary?.r2 ?? null, tooltip: 'Coefficient of determination. Larger is better (1 is perfect).' },
    { key: 'Bias', value: summary?.bias ?? null, tooltip: 'Mean(pred − true). >0 indicates overestimation on average.' },
    {
      key: 'Residual std',
      value: summary?.residual_std ?? null,
      tooltip: 'Standard deviation of residuals (pred − true). Smaller is better.',
    },
    { key: 'Pearson r', value: summary?.pearson_r ?? null, tooltip: 'Linear correlation between predictions and targets.' },
    { key: 'Spearman ρ', value: summary?.spearman_r ?? null, tooltip: 'Rank correlation between predictions and targets.' },
    {
      key: 'NRMSE',
      value: summary?.nrmse ?? null,
      tooltip: 'Normalized RMSE (e.g., divided by std(y_true)). Smaller is better.',
    },
  ];

  const regDataItems = [
    { key: 'Evaluation samples', value: summary?.n ?? totalN ?? null, tooltip: evalSamplesTooltip },
    {
      key: 'True range',
      value: rangeText(summary?.y_true_min, summary?.y_true_max),
      tooltip: 'Range of ground-truth values in the evaluation set.',
    },
    {
      key: 'Pred range',
      value: rangeText(summary?.y_pred_min, summary?.y_pred_max),
      tooltip: 'Range of predicted values in the evaluation set.',
    },
  ];

  return {
    ready,
    decoder,
    preview,
    isRegression,

    classes,
    hasDecisionScores,
    hasProbabilities,
    isVoteShare,

    summary,

    isKfold,
    isHoldout,
    nSplits,

    decoderNotes,

    classOptions,
    selectedClass,
    setSelectedClass,
    probStats,

    columns,
    handleExportPreview,

    isExportingFull,
    handleExportFull,

    previewN,
    totalN,

    datasetTitleTip,
    regTitleTip,
    lossTitleTip,
    confTitleTip,

    dataParamsItems,
    lossCalItems,
    confidenceItems,
    regPerfItems,
    regDataItems,

    showCalibrationBins,
    nonEmptyBins,
    showClassSummary,
  };
}
