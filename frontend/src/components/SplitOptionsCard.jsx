// frontend/src/components/SplitOptionsCard.jsx
import { useEffect } from 'react';
import { Card, Stack, Text, Select, NumberInput, Group, Checkbox } from '@mantine/core';
import { useDataStore } from '../state/useDataStore.js';

export default function SplitOptionsCard({
  title = 'Data split',
  allowedModes = ['holdout', 'kfold'],

  // mode + callbacks (ignored if only one mode)
  mode,
  onModeChange,

  // holdout only
  trainFrac,
  onTrainFracChange,
  minTrainFrac = 0.5,
  maxTrainFrac = 0.95,

  // kfold only
  nSplits,
  onNSplitsChange,
  minNSplits = 2,
  maxNSplits = 20,

  // common options
  stratified,
  onStratifiedChange,
  shuffle,
  onShuffleChange,
  seed,
  onSeedChange,
}) {
  const effectiveTask = useDataStore(
    (s) => s.taskSelected || s.inspectReport?.task_inferred || null,
  );

  const hasHoldout = allowedModes.includes('holdout');
  const hasKFold = allowedModes.includes('kfold');
  const showModeSelect = hasHoldout && hasKFold;

  // If consumer passes only one mode, we respect their state; mode select is hidden.
  const effectiveMode = showModeSelect ? (mode || 'holdout') : (hasKFold ? 'kfold' : 'holdout');

  // Classification can use stratified splits; regression should not.
  const allowStratified = effectiveTask !== 'regression';

  // Auto-correct stale "stratified = true" when switching to regression
  useEffect(() => {
    if (!allowStratified && stratified && onStratifiedChange) {
      onStratifiedChange(false);
    }
  }, [allowStratified, stratified, onStratifiedChange]);

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="sm">
        <Text fw={500}>{title}</Text>

        {showModeSelect && (
          <Select
            label="Split strategy"
            data={[
              { value: 'holdout', label: 'Hold-out' },
              { value: 'kfold', label: 'K-fold cross-validation' },
            ]}
            value={mode}
            onChange={(v) => onModeChange?.(v || 'holdout')}
          />
        )}

        {effectiveMode === 'holdout' && hasHoldout && (
          <NumberInput
            label="Train fraction"
            min={minTrainFrac}
            max={maxTrainFrac}
            step={0.05}
            value={trainFrac}
            onChange={onTrainFracChange}
          />
        )}

        {effectiveMode === 'kfold' && hasKFold && (
          <NumberInput
            label="K-Fold (n_splits)"
            min={minNSplits}
            max={maxNSplits}
            step={1}
            value={nSplits}
            onChange={onNSplitsChange}
          />
        )}

        <Group grow>
          <Checkbox
            label="Stratified"
            checked={!!stratified}
            disabled={!allowStratified}
            onChange={(e) => onStratifiedChange?.(e.currentTarget.checked)}
          />
          <Checkbox
            label="Shuffle split"
            checked={!!shuffle}
            onChange={(e) => onShuffleChange?.(e.currentTarget.checked)}
          />
        </Group>

        <NumberInput
          label="Seed (used if shuffle split)"
          value={seed}
          onChange={onSeedChange}
          allowDecimal={false}
          disabled={!shuffle}
        />
      </Stack>
    </Card>
  );
}
