import { Box, Text } from '@mantine/core';

import MetricPairsTable from '../common/MetricPairsTable.jsx';
import SectionTitle from '../common/SectionTitle.jsx';

import { fmt } from '../../../utils/resultsFormat.js';

export default function VotingRegSummarySection({ report }) {
  if (!report || report.kind !== 'voting') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const similarity = report.similarity || {};
  const errors = report.errors || {};
  const ensErr = errors.ensemble || {};
  const gain = errors.gain_vs_best || {};

  const metricItems = [
    {
      label: 'Avg pairwise corr',
      value: fmt(similarity.pairwise_mean_corr, 3),
      tooltip:
        'Average Pearson correlation between base predictions (off-diagonal). Higher = models behave more similarly (lower diversity).',
    },
    {
      label: 'Avg pairwise |Δ|',
      value: fmt(similarity.pairwise_mean_absdiff, 4),
      tooltip:
        'Average absolute difference between base predictions (off-diagonal). Higher = more diversity in predictions.',
    },
    {
      label: 'Mean pred spread',
      value: fmt(similarity.prediction_spread_mean, 4),
      tooltip:
        'Mean per-sample standard deviation across base predictions. Higher = base estimators disagree more.',
    },
    {
      label: 'Ensemble RMSE',
      value: fmt(ensErr.rmse, 4),
      tooltip:
        'Root mean squared error of the ensemble predictions on the pooled evaluation samples.',
    },
    {
      label: 'RMSE reduction',
      value: fmt(gain.rmse_reduction, 4),
      tooltip:
        'Best-base RMSE − ensemble RMSE. Positive = ensemble improved over the best single estimator.',
    },
    {
      label: 'N samples',
      value: fmt(errors.n_total, 0),
      tooltip:
        'Total number of evaluation samples pooled across folds used for the ensemble-specific report.',
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
        Voting: <b>{report.voting}</b> • Estimators: <b>{report.n_estimators}</b>
      </Text>
    </Box>
  );
}
