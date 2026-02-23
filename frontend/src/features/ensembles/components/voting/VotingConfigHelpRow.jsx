import { Stack, Group, Button, Box } from '@mantine/core';
import { IconPlus } from '@tabler/icons-react';

import { VotingIntroText } from '../../../../shared/content/help/EnsembleHelpText.jsx';

import ParamNumber from '../common/ParamNumber.jsx';
import ParamSelect from '../common/ParamSelect.jsx';

export default function VotingConfigHelpRow({
  effectiveTask,
  effectiveVotingType,
  onVotingTypeChange,
  mode,
  estimatorsCount,
  onClampEstimatorCount,
  onAddEstimator,
  algoOptionsLength,
  showHelp,
  onToggleHelp,
  votingType,
}) {
  return (
    <Group align="stretch" justify="space-between" wrap="wrap" gap="md">
      {/* Left: A and B stacked */}
      <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
        <ParamSelect
          label="Voting type"
          value={effectiveVotingType}
          onChange={onVotingTypeChange}
          data={[
            { value: 'hard', label: 'Hard (labels)' },
            { value: 'soft', label: 'Soft (probabilities)' },
          ]}
          disabled={effectiveTask === 'regression'}
          description={
            effectiveTask === 'regression'
              ? 'VotingRegressor is used for regression; voting type is ignored.'
              : 'Soft voting requires all estimators to support predict_proba.'
          }
        />

        {mode === 'simple' ? (
          <ParamNumber
            label="Number of models"
            min={2}
            step={1}
            value={estimatorsCount}
            onChange={onClampEstimatorCount}
            disabled={algoOptionsLength === 0}
          />
        ) : (
          <Button
            leftSection={<IconPlus size={16} />}
            variant="light"
            onClick={onAddEstimator}
            disabled={algoOptionsLength === 0}
          >
            Add estimator
          </Button>
        )}
      </Stack>

      {/* Right: C help preview (same height as left stack) */}
      <Box style={{ flex: 1, minWidth: 260 }}>
        <Stack justify="space-between" style={{ height: '100%' }} gap="xs">
          <Box>
            <VotingIntroText effectiveTask={effectiveTask} votingType={votingType} />
          </Box>

          <Group justify="flex-end">
            <Button size="xs" variant="subtle" onClick={onToggleHelp}>
              {showHelp ? 'Show less' : 'Show more'}
            </Button>
          </Group>
        </Stack>
      </Box>
    </Group>
  );
}
