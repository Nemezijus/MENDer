import { Divider } from '@mantine/core';

import ResultsCardShell from './results/common/ResultsCardShell.jsx';

import BaggingRegSummarySection from './results/bagging/BaggingRegSummarySection.jsx';
import BaggingRegMatricesSection from './results/bagging/BaggingRegMatricesSection.jsx';
import BaggingRegScoreHistogramSection from './results/bagging/BaggingRegScoreHistogramSection.jsx';
import BaggingRegErrorsSection from './results/bagging/BaggingRegErrorsSection.jsx';

export default function BaggingEnsembleRegressionResults({ report }) {
  if (!report || report.kind !== 'bagging') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  return (
    <ResultsCardShell title="Bagging ensemble insights">
      <BaggingRegSummarySection report={report} />
      <Divider my="xs" />

      <BaggingRegMatricesSection report={report} />
      <Divider my="xs" />

      <BaggingRegScoreHistogramSection report={report} />
      <Divider my="xs" />

      <BaggingRegErrorsSection report={report} />
    </ResultsCardShell>
  );
}
