import { Stack, Divider, Group } from '@mantine/core';

import ModelSelectionCard from '../../../training/components/ModelSelectionCard.jsx';

import ParamNumber from '../common/ParamNumber.jsx';
import ParamSelect from '../common/ParamSelect.jsx';
import ParamSwitch from '../common/ParamSwitch.jsx';

/**
 * Presentational config pane for Bagging.
 * Keeps value->override conversion close to inputs, while the parent owns state.
 */
export default function BaggingConfigPane({
  mode,
  algoOptions,
  compatibleAlgos,
  models,
  enums,
  effectiveTask,
  effectiveBaseEstimator,
  onBaseEstimatorChange,
  onBaseEstimatorConfigChange,
  bagging,
  baggingDefaults,
  samplingStrategyOptions,
  setBagging,
}) {
  const dispNEstimators = bagging.n_estimators ?? baggingDefaults?.n_estimators;
  const dispMaxSamples = bagging.max_samples ?? baggingDefaults?.max_samples;
  const dispMaxFeatures = bagging.max_features ?? baggingDefaults?.max_features;
  const dispNJobs = bagging.n_jobs ?? baggingDefaults?.n_jobs;
  const dispRandomState = bagging.random_state ?? baggingDefaults?.random_state;

  const dispBootstrap = bagging.bootstrap ?? baggingDefaults?.bootstrap ?? false;
  const dispBootstrapFeatures = bagging.bootstrap_features ?? baggingDefaults?.bootstrap_features ?? false;
  const dispOobScore = bagging.oob_score ?? baggingDefaults?.oob_score ?? false;

  const dispBalanced = bagging.balanced ?? baggingDefaults?.balanced ?? false;
  const dispSamplingStrategy = bagging.sampling_strategy ?? baggingDefaults?.sampling_strategy;
  const dispReplacement = bagging.replacement ?? baggingDefaults?.replacement ?? false;

  return (
    <Stack style={{ width: '100%' }} gap="sm">
      <ParamSelect
        label="Base estimator"
        placeholder={algoOptions.length ? 'Select model' : 'Loading…'}
        data={algoOptions}
        value={effectiveBaseEstimator?.algo ?? null}
        onChange={(v) => {
          if (!v) return;
          onBaseEstimatorChange({ algo: v });
        }}
        disabled={algoOptions.length === 0}
      />

      {mode === 'advanced' && (
        <ModelSelectionCard
          title="Base estimator configuration"
          model={effectiveBaseEstimator}
          models={models}
          schema={models?.schema}
          enums={enums}
          taskOverride={effectiveTask}
          compatibleAlgos={compatibleAlgos}
          onChange={onBaseEstimatorConfigChange}
        />
      )}

      <Divider my="xs" label="Bagging" labelPosition="center" />

      <Group grow align="flex-end" wrap="wrap">
        <ParamNumber
          label="Estimators"
          min={1}
          step={10}
          value={dispNEstimators}
          onChange={(v) => setBagging({ n_estimators: v })}
        />

        <ParamNumber
          label="Max samples"
          min={0.05}
          max={1}
          step={0.05}
          value={dispMaxSamples}
          onChange={(v) => setBagging({ max_samples: v })}
        />

        <ParamNumber
          label="Max features"
          min={0.05}
          max={1}
          step={0.05}
          value={dispMaxFeatures}
          onChange={(v) => setBagging({ max_features: v })}
        />
      </Group>

      <Group grow align="flex-end" wrap="wrap">
        <ParamNumber label="n_jobs" min={-1} step={1} value={dispNJobs} onChange={(v) => setBagging({ n_jobs: v })} />

        <ParamNumber
          label="Random state"
          min={0}
          step={1}
          value={dispRandomState}
          onChange={(v) => setBagging({ random_state: v })}
        />
      </Group>

      <Group grow>
        <ParamSwitch
          label="Bootstrap"
          checked={Boolean(dispBootstrap)}
          onChange={(checked) => setBagging({ bootstrap: checked })}
        />

        <ParamSwitch
          label="Bootstrap features"
          checked={Boolean(dispBootstrapFeatures)}
          onChange={(checked) => setBagging({ bootstrap_features: checked })}
        />

        <ParamSwitch
          label="OOB score"
          checked={Boolean(dispOobScore)}
          onChange={(checked) => setBagging({ oob_score: checked })}
        />
      </Group>

      <Divider my="xs" label="Balanced bagging" labelPosition="center" />

      <Group grow>
        <ParamSwitch
          label="Balanced"
          checked={Boolean(dispBalanced)}
          onChange={(checked) => setBagging({ balanced: checked })}
        />

        <ParamSwitch
          label="Replacement"
          checked={Boolean(dispReplacement)}
          onChange={(checked) => setBagging({ replacement: checked })}
          disabled={!Boolean(dispBalanced)}
        />
      </Group>

      <ParamSelect
        label="Sampling strategy"
        data={samplingStrategyOptions}
        value={dispSamplingStrategy ?? null}
        onChange={(v) => setBagging({ sampling_strategy: v })}
        disabled={!Boolean(dispBalanced)}
      />
    </Stack>
  );
}
