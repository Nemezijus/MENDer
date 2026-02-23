import { Divider } from '@mantine/core';

import ResultsCardShell from './results/common/ResultsCardShell.jsx';

import AdaBoostRegSummarySection from './results/adaboost/AdaBoostRegSummarySection.jsx';
import AdaBoostRegWeightsAndErrorsSection from './results/adaboost/AdaBoostRegWeightsAndErrorsSection.jsx';
import AdaBoostRegStageScoreSection from './results/adaboost/AdaBoostRegStageScoreSection.jsx';

export default function AdaBoostEnsembleRegressionResults({ report }) {
  if (!report || report.kind !== 'adaboost') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  return (
    <ResultsCardShell title="AdaBoost ensemble insights">
      <AdaBoostRegSummarySection report={report} />
      <Divider my="xs" />

      <AdaBoostRegWeightsAndErrorsSection report={report} />
      <Divider my="xs" />

      <AdaBoostRegStageScoreSection report={report} />
    </ResultsCardShell>
  );
}
