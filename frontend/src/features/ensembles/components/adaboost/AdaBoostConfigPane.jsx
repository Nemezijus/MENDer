import { Stack, Divider, Group, Alert } from '@mantine/core';

import ModelSelectionCard from '../../../training/components/ModelSelectionCard.jsx';

import ParamNumber from '../common/ParamNumber.jsx';
import ParamSelect from '../common/ParamSelect.jsx';

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
    effectiveTask === 'regression' ? undefined : adaboost.algorithm ?? adaboostDefaults?.algorithm ?? undefined;

  return (
    <Stack style={{ width: '100%' }} gap="sm">
      <ParamSelect
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
          The selected base estimator may not support sample weights. AdaBoost can still run, but performance may
          differ.
        </Alert>
      )}

      <Divider my="xs" label="AdaBoost" labelPosition="center" />

      <Group grow align="flex-end" wrap="wrap">
        <ParamNumber label="Estimators" min={1} step={10} value={dispNEstimators} onChange={(v) => setAdaBoost({ n_estimators: v })} />

        <ParamNumber
          label="Learning rate"
          min={0.001}
          step={0.05}
          value={dispLearningRate}
          onChange={(v) => setAdaBoost({ learning_rate: v })}
        />

        <ParamNumber
          label="Random state"
          min={0}
          step={1}
          value={dispRandomState}
          onChange={(v) => setAdaBoost({ random_state: v })}
        />
      </Group>

      {effectiveTask !== 'regression' && (
        <ParamSelect
          label="Algorithm"
          data={algorithmOptions}
          value={dispAlgorithm ?? '__default__'}
          onChange={(v) => setAdaBoost({ algorithm: v })}
        />
      )}

      {effectiveTask === 'regression' && !metricForPayload && (
        <Alert color="yellow" title="Metric">
          No regression metric is selected. Choose a metric in Settings → Metric.
        </Alert>
      )}
    </Stack>
  );
}
