import { useEffect, useState } from 'react';
import {
  Card,
  Stack,
  Text,
  Select,
  NumberInput,
  Group,
  Checkbox,
  Box,
  Button,
} from '@mantine/core';
import { useDataStore } from '../state/useDataStore.js';
import SplitHelpText, {
  SplitIntroText,
} from './helpers/helpTexts/SplitHelpText.jsx';

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
  const effectiveMode = showModeSelect
    ? mode || 'holdout'
    : hasKFold
    ? 'kfold'
    : 'holdout';

  // Classification can use stratified splits; regression should not.
  const allowStratified = effectiveTask !== 'regression';

  // Toggle for showing/hiding the detailed help block (C)
  const [showDetails, setShowDetails] = useState(false);

  // Auto-correct stale "stratified = true" when switching to regression
  useEffect(() => {
    if (!allowStratified && stratified && onStratifiedChange) {
      onStratifiedChange(false);
    }
  }, [allowStratified, stratified, onStratifiedChange]);

  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        {/* Centered title */}
        <Text fw={700} size="lg" align="center">
          {title}
        </Text>

        {/* A + B row: controls on the left, short intro on the right */}
        <Group align="flex-start" gap="xl" grow wrap="nowrap">
          {/* Block A: split controls */}
          <Box style={{ flex: 1, minWidth: 0 }}>
            <Stack gap="sm">
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
                  onChange={(e) =>
                    onStratifiedChange?.(e.currentTarget.checked)
                  }
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
          </Box>

          {/* Block B: short intro help text + toggle button */}
          <Box
            style={{
              flex: 1,
              minWidth: 220,
            }}
          >
            <Stack gap="xs">
              <SplitIntroText />
              <Button
                size="xs"
                variant="subtle"
                onClick={() => setShowDetails((prev) => !prev)}
              >
                {showDetails ? 'Show less' : 'Show more'}
              </Button>
            </Stack>
          </Box>
        </Group>

        {/* Block C: full-width detailed help text, toggled */}
        {showDetails && (
          <Box mt="md">
            <SplitHelpText
              selectedMode={effectiveMode}
              allowStratified={allowStratified}
            />
          </Box>
        )}
      </Stack>
    </Card>
  );
}
