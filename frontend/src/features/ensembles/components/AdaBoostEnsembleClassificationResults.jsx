import { Divider } from '@mantine/core';

import ResultsCardShell from './results/common/ResultsCardShell.jsx';

import AdaBoostClsSummarySection from './results/adaboost/AdaBoostClsSummarySection.jsx';
import AdaBoostClsVotePlotsSection from './results/adaboost/AdaBoostClsVotePlotsSection.jsx';
import AdaBoostClsWeightsAndErrorsSection from './results/adaboost/AdaBoostClsWeightsAndErrorsSection.jsx';
import AdaBoostClsStageScoreSection from './results/adaboost/AdaBoostClsStageScoreSection.jsx';

export default function AdaBoostEnsembleClassificationResults({ report }) {
  if (!report || report.kind !== 'adaboost') return null;

  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  return (
    <ResultsCardShell title="AdaBoost ensemble insights">
      <AdaBoostClsSummarySection report={report} />
      <Divider my="xs" />

      <AdaBoostClsVotePlotsSection report={report} />
      <Divider my="xs" />

      <AdaBoostClsWeightsAndErrorsSection report={report} />
      <Divider my="xs" />

      <AdaBoostClsStageScoreSection report={report} />
    </ResultsCardShell>
  );
}
