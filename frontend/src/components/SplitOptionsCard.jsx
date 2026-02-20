import { useState } from 'react';
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
import { useSchemaDefaults } from '../state/SchemaDefaultsContext.jsx';
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
  const { split } = useSchemaDefaults();

  const effectiveTask = useDataStore(
    (s) => s.taskSelected || s.inspectReport?.task_inferred || null,
  );

  // Schema defaults (used for display only; parents should still send override-only payloads)
  const holdoutDefaults = split?.holdout?.defaults ?? null;
  const kfoldDefaults = split?.kfold?.defaults ?? null;

  const hasHoldout = allowedModes.includes('holdout');
  const hasKFold = allowedModes.includes('kfold');
  const showModeSelect = hasHoldout && hasKFold;

  // If consumer passes only one mode, we respect their state; mode select is hidden.
  const effectiveMode = showModeSelect
    ? mode || (hasHoldout ? 'holdout' : 'kfold')
    : hasKFold
    ? 'kfold'
    : 'holdout';

  // NOTE: we do not enforce task rules here; backend/engine owns the contract.
  // We only use this for help text.
  const allowStratified = effectiveTask !== 'regression';

  // Toggle for showing/hiding the detailed help block (C)
  const [showDetails, setShowDetails] = useState(false);

  const effectiveTrainFrac =
    trainFrac ?? holdoutDefaults?.train_frac ?? undefined;
  const effectiveNSplits =
    nSplits ?? kfoldDefaults?.n_splits ?? undefined;

  const defaultStratified =
    effectiveMode === 'kfold'
      ? kfoldDefaults?.stratified
      : holdoutDefaults?.stratified;

  const defaultShuffle =
    effectiveMode === 'kfold'
      ? kfoldDefaults?.shuffle
      : holdoutDefaults?.shuffle;

  const effectiveStratified = stratified ?? defaultStratified;
  const effectiveShuffle = shuffle ?? defaultShuffle;

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
                  value={showModeSelect ? (mode ?? effectiveMode) : effectiveMode}
                  onChange={(v) => onModeChange?.(v || undefined)}
                />
              )}

              {effectiveMode === 'holdout' && hasHoldout && (
                <NumberInput
                  label="Train fraction"
                  min={minTrainFrac}
                  max={maxTrainFrac}
                  step={0.05}
                  value={effectiveTrainFrac}
                  onChange={onTrainFracChange}
                />
              )}

              {effectiveMode === 'kfold' && hasKFold && (
                <NumberInput
                  label="K-Fold (n_splits)"
                  min={minNSplits}
                  max={maxNSplits}
                  step={1}
                  value={effectiveNSplits}
                  onChange={onNSplitsChange}
                />
              )}

              <Group grow>
                <Checkbox
                  label="Stratified"
                  checked={Boolean(effectiveStratified)}
                  onChange={(e) =>
                    onStratifiedChange?.(e.currentTarget.checked)
                  }
                />
                <Checkbox
                  label="Shuffle split"
                  checked={Boolean(effectiveShuffle)}
                  onChange={(e) => onShuffleChange?.(e.currentTarget.checked)}
                />
              </Group>

              <NumberInput
                label="Seed (used if shuffle split)"
                value={seed}
                onChange={onSeedChange}
                allowDecimal={false}
                disabled={!Boolean(effectiveShuffle)}
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
