import { Box, Text } from '@mantine/core';

import MetricPairsTable from '../common/MetricPairsTable.jsx';
import SectionTitle from '../common/SectionTitle.jsx';

import {
  fmt,
  fmtPct,
  prettyEstimatorName,
} from '../../../utils/resultsFormat.js';

export default function VotingClsSummarySection({ report }) {
  if (!report || report.kind !== 'voting') return null;

  const agreement = report.agreement || {};
  const vote = report.vote || {};
  const change = report.change_vs_best || {};

  const bestNamePretty = prettyEstimatorName(change.best_name || '');

  const metricItems = [
    {
      label: 'All-agree',
      value: fmtPct(agreement.all_agree_rate),
      tooltip:
        'Fraction of samples where ALL base estimators predicted the same label. Range 0–100%. Higher = more consensus (not necessarily higher accuracy).',
    },
    {
      label: 'Avg pairwise agreement',
      value: fmtPct(agreement.pairwise_mean_agreement),
      tooltip:
        'Average agreement between all estimator pairs. Range 0–100%. Higher = models behave more similarly (lower diversity).',
    },
    {
      label: 'Tie rate',
      value: fmtPct(vote.tie_rate),
      tooltip:
        'Fraction of samples where the vote resulted in a tie. Range 0–100%. Lower is better; ties mean low consensus.',
    },
    {
      label: 'Mean margin',
      value: fmt(vote.mean_margin, 3),
      tooltip:
        'Average vote margin: (top vote count − runner-up vote count). Larger margins mean clearer majorities. For N estimators, values are roughly 0…N.',
    },
    {
      label: 'Mean strength',
      value: fmt(vote.mean_strength, 3),
      tooltip:
        'Average vote strength: (top vote count / total estimators). Range 0…1. Higher = winning label got a larger share of votes.',
    },
    {
      label: 'Net vs best',
      value: fmt(change.net, 0),
      tooltip:
        'Net change vs the best single estimator: corrected − harmed (counts). Positive = ensemble helped more than it hurt.',
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
      <MetricPairsTable rows={metricRows} tooltipMaw={320} />

      <Text size="sm" c="dimmed" mt="xs" align="center">
        Voting: <b>{report.voting}</b> • Estimators: <b>{report.n_estimators}</b>
        {bestNamePretty ? (
          <>
            {' '}
            • Best estimator: <b>{bestNamePretty}</b>
          </>
        ) : null}
      </Text>
    </Box>
  );
}
