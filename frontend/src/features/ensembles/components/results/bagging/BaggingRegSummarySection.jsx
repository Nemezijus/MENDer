import { Box, Text } from '@mantine/core';

import MetricPairsTable from '../common/MetricPairsTable.jsx';
import SectionTitle from '../common/SectionTitle.jsx';

import { fmt, fmtPct, safeNum } from '../../../utils/resultsFormat.js';

export default function BaggingRegSummarySection({ report }) {
  if (!report || report.kind !== 'bagging') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const bag = report.bagging || {};
  const oob = report.oob || {};
  const sim = report.similarity || report.diversity || {};
  const errors = report.errors || {};

  const nEstimators = safeNum(bag.n_estimators);
  const bootstrap = typeof bag.bootstrap === 'boolean' ? bag.bootstrap : null;
  const bootstrapFeatures = typeof bag.bootstrap_features === 'boolean' ? bag.bootstrap_features : null;

  const metricItems = [
    {
      label: 'OOB score',
      value: oob.score == null ? '—' : fmt(oob.score, 4),
      tooltip:
        'Out-of-bag (OOB) score: estimated generalization performance using only samples not used to train each bootstrap estimator (requires oob_score=True).',
    },
    {
      label: 'OOB coverage',
      value: oob.coverage_rate == null ? '—' : fmtPct(oob.coverage_rate, 1),
      tooltip:
        'Fraction of training samples that received an OOB prediction. Low coverage can make OOB score noisy.',
    },
    {
      label: 'Avg pairwise corr',
      value: sim.pairwise_mean_corr == null ? '—' : fmt(sim.pairwise_mean_corr, 3),
      tooltip:
        'Average Pearson correlation between base predictions (off-diagonal). Higher = more similarity (lower diversity).',
    },
    {
      label: 'Avg pairwise |Δ|',
      value: sim.pairwise_mean_absdiff == null ? '—' : fmt(sim.pairwise_mean_absdiff, 4),
      tooltip:
        'Average absolute difference between base predictions (off-diagonal). Higher = more diversity.',
    },
    {
      label: 'Mean pred spread',
      value: sim.prediction_spread_mean == null ? '—' : fmt(sim.prediction_spread_mean, 4),
      tooltip:
        'Mean per-sample standard deviation across base predictions. Higher = more disagreement.',
    },
    {
      label: 'N samples',
      value: errors.n_total == null ? '—' : fmt(errors.n_total, 0),
      tooltip:
        'Total number of evaluation samples pooled across folds used for bagging-specific reporting.',
    },
  ];

  const metricRows = [
    [metricItems[0], metricItems[1]],
    [metricItems[2], metricItems[3]],
    [metricItems[4], metricItems[5]],
  ];

  return (
    <Box>
      <SectionTitle title="Summary metrics" />
      <MetricPairsTable rows={metricRows} tooltipMaw={360} />

      <Text size="sm" c="dimmed" mt="xs" align="center">
        Estimators: <b>{nEstimators == null ? '—' : String(nEstimators)}</b> • Bootstrap:{' '}
        <b>{bootstrap == null ? '—' : bootstrap ? 'on' : 'off'}</b> • Feature bootstrap:{' '}
        <b>{bootstrapFeatures == null ? '—' : bootstrapFeatures ? 'on' : 'off'}</b>
      </Text>
    </Box>
  );
}
