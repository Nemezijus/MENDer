import { Stack } from '@mantine/core';

import SplitOptionsCard from '../../../shared/ui/config/SplitOptionsCard.jsx';

import VotingEnsembleClassificationResults from './VotingEnsembleClassificationResults.jsx';
import VotingEnsembleRegressionResults from './VotingEnsembleRegressionResults.jsx';

import VotingEnsembleConfigCard from './voting/VotingEnsembleConfigCard.jsx';
import VotingEnsembleFooter from './voting/VotingEnsembleFooter.jsx';

import { useVotingEnsembleController } from '../hooks/useVotingEnsembleController.js';

export default function VotingEnsemblePanel() {
  const vm = useVotingEnsembleController();

  const {
    voting,
    setVoting,
    trainResult,

    effectiveSplitMode,
    handleRun,
    loading,
    trainDisabled,
  } = vm;

  return (
    <Stack gap="md">
      <VotingEnsembleConfigCard vm={vm} />

      <SplitOptionsCard
        title="Data split"
        allowedModes={['holdout', 'kfold']}
        mode={effectiveSplitMode}
        onModeChange={(m) => setVoting({ splitMode: m })}
        trainFrac={voting.trainFrac}
        onTrainFracChange={(v) => setVoting({ trainFrac: v })}
        nSplits={voting.nSplits}
        onNSplitsChange={(v) => setVoting({ nSplits: v })}
        stratified={voting.stratified}
        onStratifiedChange={(v) => setVoting({ stratified: v })}
        shuffle={voting.shuffle}
        onShuffleChange={(v) => setVoting({ shuffle: v })}
        seed={voting.seed}
        onSeedChange={(v) => setVoting({ seed: v })}
      />

      {trainResult?.ensemble_report?.kind === 'voting' &&
        trainResult.ensemble_report.task === 'classification' && (
          <VotingEnsembleClassificationResults report={trainResult.ensemble_report} />
        )}

      {trainResult?.ensemble_report?.kind === 'voting' &&
        trainResult.ensemble_report.task === 'regression' && (
          <VotingEnsembleRegressionResults report={trainResult.ensemble_report} />
        )}

      <VotingEnsembleFooter onRun={handleRun} loading={loading} disabled={trainDisabled} />
    </Stack>
  );
}
