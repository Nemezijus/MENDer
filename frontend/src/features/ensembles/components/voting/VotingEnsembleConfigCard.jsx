import { Card, Stack, Group, Button, Divider } from '@mantine/core';

import EnsemblePanelHeader from '../common/EnsemblePanelHeader.jsx';
import EnsembleErrorAlert from '../common/EnsembleErrorAlert.jsx';

import VotingConfigHelpRow from './VotingConfigHelpRow.jsx';
import VotingHelpBlock from './VotingHelpBlock.jsx';
import VotingWarnings from './VotingWarnings.jsx';
import VotingSimpleEstimatorList from './VotingSimpleEstimatorList.jsx';
import VotingAdvancedEstimatorList from './VotingAdvancedEstimatorList.jsx';

export default function VotingEnsembleConfigCard({ vm }) {
  const {
    voting,
    setVoting,
    resetToDefaults,
    handleRun,
    loading,
    trainDisabled,
    err,

    effectiveTask,
    effectiveVotingType,
    handleVotingTypeChange,

    estimators,
    clampEstimatorCount,
    addEstimator,

    algoOptions,

    showHelp,
    toggleHelp,

    duplicateAlgosLabel,
    metricIsAllowed,
    metricOverride,
    defaultMetricFromSchema,
    effectiveSplitMode,

    updateEstimatorAlgoSimple,
    removeVotingEstimatorAt,
    updateVotingEstimatorAt,

    models,
    enums,
  } = vm;

  return (
    <Card withBorder shadow="sm" padding="lg">
      <Stack gap="md">
        <EnsemblePanelHeader
          title="Voting ensemble"
          mode={voting.mode}
          onModeChange={(v) => setVoting({ mode: v })}
          onReset={resetToDefaults}
        />

        <Group justify="flex-end">
          <Button onClick={handleRun} loading={loading} disabled={trainDisabled}>
            Train voting ensemble
          </Button>
        </Group>

        <EnsembleErrorAlert error={err} />

        <VotingConfigHelpRow
          effectiveTask={effectiveTask}
          effectiveVotingType={effectiveVotingType}
          onVotingTypeChange={handleVotingTypeChange}
          mode={voting.mode}
          estimatorsCount={estimators.length}
          onClampEstimatorCount={clampEstimatorCount}
          onAddEstimator={addEstimator}
          algoOptionsLength={algoOptions.length}
          showHelp={showHelp}
          onToggleHelp={toggleHelp}
          votingType={voting.votingType}
        />

        {showHelp && (
          <VotingHelpBlock effectiveTask={effectiveTask} votingType={voting.votingType} mode={voting.mode} />
        )}

        <Divider />

        <VotingWarnings
          duplicateAlgosLabel={duplicateAlgosLabel}
          metricIsAllowed={metricIsAllowed}
          metricOverride={metricOverride}
          effectiveTask={effectiveTask}
          defaultMetricFromSchema={defaultMetricFromSchema}
          algoOptionsLength={algoOptions.length}
          effectiveSplitMode={effectiveSplitMode}
        />

        {voting.mode === 'simple' && (
          <VotingSimpleEstimatorList
            estimators={estimators}
            algoOptions={algoOptions}
            onAlgoChangeAt={updateEstimatorAlgoSimple}
            onRemoveAt={removeVotingEstimatorAt}
          />
        )}

        {voting.mode === 'advanced' && (
          <VotingAdvancedEstimatorList
            estimators={estimators}
            onUpdateAt={updateVotingEstimatorAt}
            onRemoveAt={removeVotingEstimatorAt}
            models={models}
            enums={enums}
          />
        )}
      </Stack>
    </Card>
  );
}
