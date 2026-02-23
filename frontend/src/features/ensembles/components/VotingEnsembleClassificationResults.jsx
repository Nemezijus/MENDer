import { Divider } from '@mantine/core';

import ResultsCardShell from './results/common/ResultsCardShell.jsx';

import VotingClsSummarySection from './results/voting/VotingClsSummarySection.jsx';
import VotingClsBaseAndChangeSection from './results/voting/VotingClsBaseAndChangeSection.jsx';
import VotingClsAgreementHeatmapSection from './results/voting/VotingClsAgreementHeatmapSection.jsx';
import VotingClsVoteHistogramsSection from './results/voting/VotingClsVoteHistogramsSection.jsx';

export default function VotingEnsembleClassificationResults({ report }) {
  if (!report || report.kind !== 'voting') return null;

  const task = (report.task || 'classification').toLowerCase();
  if (task === 'regression') return null;

  return (
    <ResultsCardShell title="Voting ensemble insights">
      <VotingClsSummarySection report={report} />
      <Divider my="xs" />

      <VotingClsBaseAndChangeSection report={report} />
      <Divider my="xs" />

      <VotingClsAgreementHeatmapSection report={report} />
      <Divider my="xs" />

      <VotingClsVoteHistogramsSection report={report} />
    </ResultsCardShell>
  );
}
