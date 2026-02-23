import { Box, Text } from '@mantine/core';

import MetricPairsTable from '../common/MetricPairsTable.jsx';
import SectionTitle from '../common/SectionTitle.jsx';

import { fmt, fmtPct, safeNum, titleCase } from '../../../utils/resultsFormat.js';

export default function BaggingClsSummarySection({ report }) {
  if (!report || report.kind !== 'bagging') return null;

  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  const bag = report.bagging || {};
  const oob = report.oob || {};
  const diversity = report.diversity || {};
  const vote = report.vote || {};

  const baseAlgo = bag.base_algo ? titleCase(bag.base_algo) : '—';
  const nEstimators = safeNum(bag.n_estimators);

  const bootstrap = typeof bag.bootstrap === 'boolean' ? bag.bootstrap : null;
  const bootstrapFeatures = typeof bag.bootstrap_features === 'boolean' ? bag.bootstrap_features : null;

  const isBalanced = !!bag.balanced;

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
      label: 'All-agree',
      value: diversity.all_agree_rate == null ? '—' : fmtPct(diversity.all_agree_rate, 1),
      tooltip: 'Fraction of evaluation samples where ALL base estimators predicted the same label.',
    },
    {
      label: 'Avg pairwise agreement',
      value:
        diversity.pairwise_mean_agreement == null ? '—' : fmtPct(diversity.pairwise_mean_agreement, 1),
      tooltip:
        'Average agreement between estimator pairs. Higher = lower diversity (more redundancy).',
    },
    {
      label: 'Tie rate',
      value: vote.tie_rate == null ? '—' : fmtPct(vote.tie_rate, 1),
      tooltip: 'Fraction of samples where the vote was tied for the top label (weak consensus).',
    },
    {
      label: 'Mean margin',
      value: vote.mean_margin == null ? '—' : fmt(vote.mean_margin, 3),
      tooltip:
        'Average vote margin: (top vote count − runner-up vote count). Larger = clearer majorities.',
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
        Base estimator: <b>{baseAlgo}</b> • Estimators: <b>{nEstimators == null ? '—' : String(nEstimators)}</b> • Bootstrap:{' '}
        <b>{bootstrap == null ? '—' : bootstrap ? 'on' : 'off'}</b> • Feature bootstrap:{' '}
        <b>{bootstrapFeatures == null ? '—' : bootstrapFeatures ? 'on' : 'off'}</b>
        {isBalanced ? (
          <>
            {' '}
            • <b>Balanced</b>
          </>
        ) : null}
      </Text>
    </Box>
  );
}
