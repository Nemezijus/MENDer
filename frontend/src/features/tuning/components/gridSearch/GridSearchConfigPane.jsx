import { Box, Stack, Divider, Text } from '@mantine/core';

import SplitOptionsCard from '../../../../shared/ui/config/SplitOptionsCard.jsx';
import ModelSelectionCard from '../../../training/components/ModelSelectionCard.jsx';
import HyperparameterSelector from '../helpers/HyperparameterSelector.jsx';
import NumberOverrideInput from '../common/NumberOverrideInput.jsx';

export default function GridSearchConfigPane({
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

  // Hyperparams
  hyperParam1,
  onHyperParam1Change,
  hyperParam2,
  onHyperParam2Change,

  // n_jobs
  nJobsOverride,
  defaultNJobs,
  onNJobsChangeOverride,
}) {
  return (
    <Box style={{ margin: '0 auto', width: '100%' }}>
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

        <Stack gap="sm">
          <Text size="sm" fw={500}>
            Select the 1st parameter to vary:
          </Text>
          <HyperparameterSelector
            schema={schema}
            model={model}
            value={hyperParam1}
            onChange={onHyperParam1Change}
            label="1st hyperparameter"
          />
        </Stack>

        <Stack gap="sm">
          <Text size="sm" fw={500}>
            Select the 2nd parameter to vary:
          </Text>
          <HyperparameterSelector
            schema={schema}
            model={model}
            value={hyperParam2}
            onChange={onHyperParam2Change}
            label="2nd hyperparameter"
          />
        </Stack>

        <Box style={{ maxWidth: 180 }}>
          <NumberOverrideInput
            label="n_jobs"
            min={1}
            step={1}
            valueOverride={nJobsOverride}
            defaultValue={defaultNJobs}
            onChangeOverride={onNJobsChangeOverride}
          />
        </Box>
      </Stack>
    </Box>
  );
}
