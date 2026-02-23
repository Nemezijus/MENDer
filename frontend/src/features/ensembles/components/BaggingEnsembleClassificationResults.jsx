import { Divider } from '@mantine/core';

import ResultsCardShell from './results/common/ResultsCardShell.jsx';

import BaggingClsSummarySection from './results/bagging/BaggingClsSummarySection.jsx';
import BaggingClsAgreementAndScoresSection from './results/bagging/BaggingClsAgreementAndScoresSection.jsx';
import BaggingClsVoteHistogramsSection from './results/bagging/BaggingClsVoteHistogramsSection.jsx';

export default function BaggingEnsembleClassificationResults({ report }) {
  if (!report || report.kind !== 'bagging') return null;

  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  return (
    <ResultsCardShell title="Bagging ensemble insights">
      <BaggingClsSummarySection report={report} />
      <Divider my="xs" />

      <BaggingClsAgreementAndScoresSection report={report} />
      <Divider my="xs" />

      <BaggingClsVoteHistogramsSection report={report} />
    </ResultsCardShell>
  );
}
