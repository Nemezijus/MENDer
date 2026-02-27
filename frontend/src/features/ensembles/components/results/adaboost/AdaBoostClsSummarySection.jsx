import { Box, Text, Tooltip } from '@mantine/core';

import MetricPairsTable from '../common/MetricPairsTable.jsx';
import SectionTitle from '../common/SectionTitle.jsx';

import { fmt, fmtPct, safeNum, titleCase } from '../../../utils/resultsFormat.js';
import { fmtIntish } from '../../../utils/resultHelpers.js';


export default function AdaBoostClsSummarySection({ report }) {
  if (!report || report.kind !== 'adaboost') return null;

  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  const ada = report.adaboost || {};
  const vote = report.vote || {};
  const weights = report.weights || {};
  const stages = report.stages || {};

  const baseAlgo = ada.base_algo ? titleCase(ada.base_algo) : '—';
  const nEstimatorsConfigured = safeNum(ada.n_estimators);
  const learningRate = safeNum(ada.learning_rate);
  const algorithm = ada.algorithm ? String(ada.algorithm) : null;

  const nEstimatorsFittedMean = safeNum(stages.n_estimators_fitted_mean);
  const nNontrivialMean = safeNum(stages.n_nontrivial_weights_mean);
  const weightEps = safeNum(stages.weight_eps);

  const topkMean = stages.weight_mass_topk_mean || null;
  const top10Mass = topkMean ? safeNum(topkMean['10']) : null;
  const top20Mass = topkMean ? safeNum(topkMean['20']) : null;

  const effectiveN = safeNum(weights.effective_n_mean);

  const summaryItems = [
    {
      label: 'Mean margin',
      value: vote.mean_margin == null ? '—' : fmt(vote.mean_margin, 3),
      tooltip: 'Mean weighted vote margin: (top weight − runner-up weight) / total weight.',
    },
    {
      label: 'Mean strength',
      value: vote.mean_strength == null ? '—' : fmtPct(vote.mean_strength, 1),
      tooltip: 'Mean weighted vote strength: top weight / total weight.',
    },
    {
      label: 'Tie rate',
      value: vote.tie_rate == null ? '—' : fmtPct(vote.tie_rate, 1),
      tooltip: 'Fraction of samples where top two labels have equal weight (weak consensus).',
    },
    {
      label: 'Eff. # estimators',
      value: effectiveN == null ? '—' : fmt(effectiveN, 2),
      tooltip:
        'Effective number of estimators (ESS) derived from estimator weights. Lower means a few stages dominate.',
    },
  ];

  const rows = [
    [summaryItems[0], summaryItems[1]],
    [summaryItems[2], summaryItems[3]],
  ];

  const showConcentrationHint =
    effectiveN != null &&
    ((nEstimatorsConfigured != null && effectiveN < 0.25 * nEstimatorsConfigured) ||
      (nEstimatorsFittedMean != null && effectiveN < 0.25 * nEstimatorsFittedMean));

  return (
    <Box>
      <SectionTitle title="Summary metrics" />
      <MetricPairsTable rows={rows} tooltipMaw={360} />

      <Text size="sm" c="dimmed" mt="xs" align="center">
        Base estimator: <b>{baseAlgo}</b> • Estimators: <b>{nEstimatorsConfigured == null ? '—' : String(nEstimatorsConfigured)}</b> • Learning rate:{' '}
        <b>{learningRate == null ? '—' : fmt(learningRate, 3)}</b>
        {algorithm ? (
          <>
            {' '}
            • Algorithm: <b>{algorithm}</b>
          </>
        ) : null}
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

      {showConcentrationHint ? (
        <Text size="sm" c="dimmed" mt={6} align="center">
          Note: weights are highly concentrated (effective # is much smaller than configured). Some histograms may collapse into few bins.
        </Text>
      ) : null}
    </Box>
  );
}
