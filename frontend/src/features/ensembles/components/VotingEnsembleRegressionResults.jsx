import { Divider } from '@mantine/core';

import ResultsCardShell from './results/common/ResultsCardShell.jsx';

import VotingRegSummarySection from './results/voting/VotingRegSummarySection.jsx';
import VotingRegBaseEstimatorsSection from './results/voting/VotingRegBaseEstimatorsSection.jsx';
import VotingRegPairwiseMatricesSection from './results/voting/VotingRegPairwiseMatricesSection.jsx';
import VotingRegErrorsSection from './results/voting/VotingRegErrorsSection.jsx';

export default function VotingEnsembleRegressionResults({ report }) {
  if (!report || report.kind !== 'voting') return null;
  const task = (report.task || 'classification').toLowerCase();
  if (task !== 'regression') return null;

  return (
    <ResultsCardShell title="Voting ensemble insights">
      <VotingRegSummarySection report={report} />
      <Divider my="xs" />

      <VotingRegBaseEstimatorsSection report={report} />
      <Divider my="xs" />

      <VotingRegPairwiseMatricesSection report={report} />
      <Divider my="xs" />

      <VotingRegErrorsSection report={report} />
    </ResultsCardShell>
  );
}
