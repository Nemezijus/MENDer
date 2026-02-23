import { Box } from '@mantine/core';

import MetricPairsTable from '../common/MetricPairsTable.jsx';
import SectionTitle from '../common/SectionTitle.jsx';

import { fmt } from '../../../utils/resultsFormat.js';

export default function BaggingRegErrorsSection({ report }) {
  if (!report || report.kind !== 'bagging') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  const errors = report.errors || {};
  const ensErr = errors.ensemble || {};

  const errItems = [
    {
      label: 'Ensemble RMSE',
      value: ensErr.rmse == null ? '—' : fmt(ensErr.rmse, 4),
      tooltip: 'Root mean squared error of the bagging ensemble predictions.',
    },
    {
      label: 'Ensemble MAE',
      value: ensErr.mae == null ? '—' : fmt(ensErr.mae, 4),
      tooltip: 'Mean absolute error of the bagging ensemble predictions.',
    },
    {
      label: 'Ensemble R²',
      value: ensErr.r2 == null ? '—' : fmt(ensErr.r2, 4),
      tooltip: 'Coefficient of determination (R²) for the bagging ensemble.',
    },
    {
      label: '—',
      value: '—',
      tooltip: '',
    },
  ];

  const errRows = [
    [errItems[0], errItems[1]],
    [errItems[2], errItems[3]],
  ];

  return (
    <Box>
      <SectionTitle title="Errors" />
      <MetricPairsTable rows={errRows} tooltipMaw={360} />
    </Box>
  );
}
