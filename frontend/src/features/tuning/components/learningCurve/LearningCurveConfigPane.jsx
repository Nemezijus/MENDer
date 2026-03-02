import { Box, Stack, Divider, Text, NumberInput } from '@mantine/core';

import SplitOptionsCard from '../../../../shared/ui/config/SplitOptionsCard.jsx';
import ModelSelectionCard from '../../../training/components/ModelSelectionCard.jsx';
import NumberOverrideInput from '../common/NumberOverrideInput.jsx';

export default function LearningCurveConfigPane({
  // Split
  nSplits,
  onNSplitsChange,
  stratified,
  onStratifiedChange,
  shuffle,
  onShuffleChange,
  seed,
  onSeedChange,

  // Model
  model,
  onModelChange,
  models,
  schema,
  enums,

  // Learning curve params
  nStepsOverride,
  defaultNSteps,
  onNStepsChangeOverride,
  nJobsOverride,
  defaultNJobs,
  onNJobsChangeOverride,
  trainSizesCSV,
  onTrainSizesCSVChange,

  // Recommendation threshold
  withinPct,
  onWithinPctChange,
}) {
  return (
    <Box w="100%" className="tuningPaneRoot">
      <Stack gap="sm">
        <SplitOptionsCard
          allowedModes={['kfold']}
          nSplits={nSplits}
          onNSplitsChange={onNSplitsChange}
          stratified={stratified}
          onStratifiedChange={onStratifiedChange}
          shuffle={shuffle}
          onShuffleChange={onShuffleChange}
          seed={seed}
          onSeedChange={onSeedChange}
        />

        <Divider my="xs" />

        <ModelSelectionCard
          model={model}
          onChange={onModelChange}
          schema={schema}
          enums={enums}
          models={models}
        />

        <Divider my="xs" />

        <NumberOverrideInput
          label="Steps (used if Train sizes empty)"
          min={2}
          max={50}
          step={1}
          valueOverride={nStepsOverride}
          defaultValue={defaultNSteps}
          onChangeOverride={onNStepsChangeOverride}
        />

        <NumberOverrideInput
          label="n_jobs"
          min={1}
          step={1}
          valueOverride={nJobsOverride}
          defaultValue={defaultNJobs}
          onChangeOverride={onNJobsChangeOverride}
        />

        <Text size="sm" c="dimmed">
          Optional Train sizes (CSV): fractions in (0,1] or absolute integers. Example:
          <Text span fw={500}> 0.1,0.3,0.5,0.7,1.0 </Text> or{' '}
          <Text span fw={500}> 50,100,200 </Text>
        </Text>
        <textarea
          className="tuningCsvTextarea"
          placeholder="e.g. 0.1,0.3,0.5,0.7,1.0"
          value={trainSizesCSV}
          onChange={(e) => onTrainSizesCSVChange(e.currentTarget.value)}
        />

        <Divider my="xs" />

        <NumberInput
          label="Recommend the smallest train size achieving at least this fraction of the peak validation score"
          description="e.g., 0.99 = within 1% of peak"
          min={0.5}
          max={1.0}
          step={0.01}
          value={withinPct}
          onChange={onWithinPctChange}
          precision={2}
        />
      </Stack>
    </Box>
  );
}
