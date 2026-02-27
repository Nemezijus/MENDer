import { Box, Text, Tooltip } from '@mantine/core';

import MetricPairsTable from '../common/MetricPairsTable.jsx';
import SectionTitle from '../common/SectionTitle.jsx';

import { fmt, fmtPct, safeNum, titleCase } from '../../../utils/resultsFormat.js';
import { fmtIntish } from '../../../utils/resultHelpers.js';


export default function AdaBoostRegSummarySection({ report }) {
  if (!report || report.kind !== 'adaboost') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const ada = report.adaboost || {};
  const weights = report.weights || {};
  const modelErrors = report.model_errors || {};
  const stages = report.stages || {};

  const baseAlgo = ada.base_algo ? titleCase(ada.base_algo) : '—';
  const nEstimatorsConfigured = safeNum(ada.n_estimators);
  const learningRate = safeNum(ada.learning_rate);

  const nEstimatorsFittedMean = safeNum(stages.n_estimators_fitted_mean);
  const nNontrivialMean = safeNum(stages.n_nontrivial_weights_mean);
  const weightEps = safeNum(stages.weight_eps);

  const topkMean = stages.weight_mass_topk_mean || null;
  const top10Mass = topkMean ? safeNum(topkMean['10']) : null;
  const top20Mass = topkMean ? safeNum(topkMean['20']) : null;

  const effectiveN = safeNum(weights.effective_n_mean);

  const ensErr = modelErrors.ensemble || {};
  const gain = modelErrors.gain_vs_best || {};

  const summaryItems = [
    {
      label: 'Eff. # estimators',
      value: effectiveN == null ? '—' : fmt(effectiveN, 2),
      tooltip:
        'Effective number of estimators (ESS) derived from estimator weights. Lower means a few stages dominate.',
    },
    {
      label: 'Top-10 weight mass',
      value: top10Mass == null ? '—' : fmtPct(top10Mass, 1),
      tooltip:
        'Average fraction of total estimator weight contained in the 10 most-weighted stages.',
    },
    {
      label: 'Ensemble RMSE',
      value: ensErr.rmse == null ? '—' : fmt(ensErr.rmse, 4),
      tooltip: 'Root mean squared error of the AdaBoost ensemble predictions.',
    },
    {
      label: 'Ensemble MAE',
      value: ensErr.mae == null ? '—' : fmt(ensErr.mae, 4),
      tooltip: 'Mean absolute error of the AdaBoost ensemble predictions.',
    },
    {
      label: 'RMSE reduction',
      value: gain.rmse_reduction == null ? '—' : fmt(gain.rmse_reduction, 4),
      tooltip:
        'Best-base RMSE − ensemble RMSE. Positive = ensemble improved over the best single estimator.',
    },
    {
      label: 'N samples',
      value: modelErrors.n_total == null ? '—' : fmt(modelErrors.n_total, 0),
      tooltip:
        'Total number of evaluation samples pooled across folds used for AdaBoost reporting.',
    },
  ];

  const rows = [
    [summaryItems[0], summaryItems[1]],
    [summaryItems[2], summaryItems[3]],
    [summaryItems[4], summaryItems[5]],
  ];

  return (
    <Box>
      <SectionTitle title="Summary metrics" />
      <MetricPairsTable rows={rows} tooltipMaw={360} />

      <Text size="sm" c="dimmed" mt="xs" align="center">
        Base estimator: <b>{baseAlgo}</b> • Estimators: <b>{nEstimatorsConfigured == null ? '—' : String(nEstimatorsConfigured)}</b> • Learning rate:{' '}
        <b>{learningRate == null ? '—' : fmt(learningRate, 3)}</b>
      </Text>

      <Text size="sm" c="dimmed" mt={6} align="center">
        <Tooltip
          label="Configured = requested n_estimators. Fitted = actual number of stage estimators trained. Non-trivial weight counts how many stages have weight greater than ε (default 1e-6)."
          multiline
          maw={520}
          withArrow
        >
          <span>
            Stages: configured <b>{fmtIntish(nEstimatorsConfigured)}</b>
            {nEstimatorsFittedMean != null ? (
              <>
                {' '}
                • fitted <b>{fmtIntish(nEstimatorsFittedMean)}</b>
              </>
            ) : null}
            {nNontrivialMean != null ? (
              <>
                {' '}
                • non-trivial weight <b>{fmtIntish(nNontrivialMean)}{weightEps != null ? ` (ε=${weightEps})` : ''}</b>
              </>
            ) : null}
            {top10Mass != null ? (
              <>
                {' '}
                • top-10 weight mass <b>{fmtPct(top10Mass, 1)}</b>
              </>
            ) : null}
            {top20Mass != null ? (
              <>
                {' '}
                • top-20 weight mass <b>{fmtPct(top20Mass, 1)}</b>
              </>
            ) : null}
          </span>
        </Tooltip>
      </Text>
    </Box>
  );
}
