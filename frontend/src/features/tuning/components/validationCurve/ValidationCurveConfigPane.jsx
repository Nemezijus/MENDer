import { Box, Stack, Divider } from '@mantine/core';

import SplitOptionsCard from '../../../../shared/ui/config/SplitOptionsCard.jsx';
import ModelSelectionCard from '../../../training/components/ModelSelectionCard.jsx';
import HyperparameterSelector from '../helpers/HyperparameterSelector.jsx';
import NumberOverrideInput from '../common/NumberOverrideInput.jsx';

export default function ValidationCurveConfigPane({
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

  // Hyperparam
  hyperParam,
  onHyperParamChange,

  // n_jobs
  nJobsOverride,
  defaultNJobs,
  onNJobsChangeOverride,
}) {
  return (
    <Box className="tuningPaneRoot">
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

        <HyperparameterSelector
          schema={schema}
          model={model}
          value={hyperParam}
          onChange={onHyperParamChange}
        />

        <Box className="tuningNarrowInput">
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
