import {
  Stack,
  Select,
  Divider,
  Group,
  NumberInput,
  Alert,
} from '@mantine/core';

import SplitOptionsCard from '../../../../shared/ui/config/SplitOptionsCard.jsx';
import ModelSelectionCard from '../../../training/components/ModelSelectionCard.jsx';

export default function AdaBoostConfigPane({
  mode,
  algoOptions,
  compatibleAlgos,
  models,
  effectiveTask,
  effectiveBaseEstimator,
  getModelDefaults,
  onBaseEstimatorChange,
  onBaseEstimatorConfigChange,
  showSampleWeightWarning,
  algorithmOptions,
  adaboost,
  adaboostDefaults,
  metricForPayload,
  setAdaBoost,
}) {
  const dispNEstimators = adaboost.n_estimators ?? adaboostDefaults?.n_estimators;
  const dispLearningRate = adaboost.learning_rate ?? adaboostDefaults?.learning_rate;
  const dispRandomState = adaboost.random_state ?? adaboostDefaults?.random_state;

  const dispAlgorithm =
    effectiveTask === 'regression'
      ? undefined
      : adaboost.algorithm ?? adaboostDefaults?.algorithm ?? undefined;

  return (
    <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
      <Select
        label="Base estimator"
        placeholder={algoOptions.length ? 'Select model' : 'Loading…'}
        data={algoOptions}
        value={effectiveBaseEstimator?.algo ?? null}
        onChange={(v) => {
          if (!v) return;
          const base = getModelDefaults?.(v) || { algo: v };
          onBaseEstimatorChange(base);
        }}
        disabled={algoOptions.length === 0}
      />

      {mode === 'advanced' && (
        <ModelSelectionCard
          title="Base estimator configuration"
          model={effectiveBaseEstimator}
          models={models}
          compatibleAlgos={compatibleAlgos}
          onChange={onBaseEstimatorConfigChange}
        />
      )}

      {showSampleWeightWarning && (
        <Alert color="yellow" title="Base estimator">
          The selected base estimator may not support sample weights. AdaBoost can still run,
          but performance may differ.
        </Alert>
      )}

      <Divider my="xs" label="AdaBoost" labelPosition="center" />

      <Group grow align="flex-end" wrap="wrap">
        <NumberInput
          label="Estimators"
          min={1}
          step={10}
          value={dispNEstimators}
          onChange={(v) =>
            setAdaBoost({ n_estimators: v === '' || v == null ? undefined : v })
          }
        />

        <NumberInput
          label="Learning rate"
          min={0.001}
          step={0.05}
          value={dispLearningRate}
          onChange={(v) =>
            setAdaBoost({ learning_rate: v === '' || v == null ? undefined : v })
          }
        />

        <NumberInput
          label="Random state"
          min={0}
          step={1}
          value={dispRandomState}
          onChange={(v) =>
            setAdaBoost({ random_state: v === '' || v == null ? undefined : v })
          }
        />
      </Group>

      {effectiveTask !== 'regression' && (
        <Select
          label="Algorithm"
          data={algorithmOptions}
          value={dispAlgorithm ?? '__default__'}
          onChange={(v) => setAdaBoost({ algorithm: v || undefined })}
        />
      )}

      <SplitOptionsCard
        title="Data split"
        allowedModes={['holdout', 'kfold']}
        mode={adaboost.splitMode}
        onModeChange={(v) => setAdaBoost({ splitMode: v })}
        trainFrac={adaboost.trainFrac}
        onTrainFracChange={(v) => setAdaBoost({ trainFrac: v })}
        nSplits={adaboost.nSplits}
        onNSplitsChange={(v) => setAdaBoost({ nSplits: v })}
        stratified={adaboost.stratified}
        onStratifiedChange={(v) => setAdaBoost({ stratified: v })}
        shuffle={adaboost.shuffle}
        onShuffleChange={(v) => setAdaBoost({ shuffle: v })}
        seed={adaboost.seed}
        onSeedChange={(v) => setAdaBoost({ seed: v })}
      />

      {effectiveTask === 'regression' && !metricForPayload && (
        <Alert color="yellow" title="Metric">
          No regression metric is selected. Choose a metric in Settings → Metric.
        </Alert>
      )}
    </Stack>
  );
}
