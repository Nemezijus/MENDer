import { Box, Text } from '@mantine/core';

import MetricPairsTable from '../common/MetricPairsTable.jsx';
import SectionTitle from '../common/SectionTitle.jsx';

import { fmt, makeUniqueLabels, niceEstimatorLabel } from '../../../utils/resultsFormat.js';

export default function VotingRegErrorsSection({ report }) {
  if (!report || report.kind !== 'voting') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const estimators = Array.isArray(report.estimators) ? report.estimators : [];

  const baseLabelsLong = estimators.map((e) =>
    niceEstimatorLabel({ name: e?.name, algo: e?.algo }),
  );
  const namesPretty = makeUniqueLabels(baseLabelsLong);

  const nameToPretty = new Map();
  estimators.forEach((e, i) => nameToPretty.set(e?.name, namesPretty[i]));

  const errors = report.errors || {};
  const ensErr = errors.ensemble || {};
  const bestErr = errors.best_base || {};
  const gain = errors.gain_vs_best || {};

  const bestNamePretty = bestErr?.name
    ? nameToPretty.get(bestErr.name) || niceEstimatorLabel({ name: bestErr.name, algo: bestErr.algo })
    : '—';

  const errorItems = [
    {
      label: 'Ensemble RMSE',
      value: fmt(ensErr.rmse, 4),
      tooltip: 'Root mean squared error of ensemble predictions.',
    },
    {
      label: 'Ensemble MAE',
      value: fmt(ensErr.mae, 4),
      tooltip: 'Mean absolute error of ensemble predictions.',
    },
    {
      label: 'Best base RMSE',
      value: fmt(bestErr.rmse, 4),
      tooltip: `RMSE of the best single estimator (${bestNamePretty}).`,
    },
    {
      label: 'Best base MAE',
      value: fmt(bestErr.mae, 4),
      tooltip: `MAE of the best single estimator (${bestNamePretty}).`,
    },
    {
      label: 'RMSE reduction',
      value: fmt(gain.rmse_reduction, 4),
      tooltip: 'Best-base RMSE − ensemble RMSE. Positive = improvement.',
    },
    {
      label: 'MAE reduction',
      value: fmt(gain.mae_reduction, 4),
      tooltip: 'Best-base MAE − ensemble MAE. Positive = improvement.',
    },
  ];

  const errorRows = [
    [errorItems[0], errorItems[1]],
    [errorItems[2], errorItems[3]],
    [errorItems[4], errorItems[5]],
  ];

  return (
    <Box>
      <SectionTitle title="Errors" />
      <MetricPairsTable rows={errorRows} tooltipMaw={360} />

      <Text size="sm" c="dimmed" mt="xs" align="center">
        Best single estimator: <b>{bestNamePretty}</b>
      </Text>
    </Box>
  );
}
