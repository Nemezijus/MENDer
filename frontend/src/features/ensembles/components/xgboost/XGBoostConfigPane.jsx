import { Stack, Group, NumberInput, Divider, Switch } from '@mantine/core';

import SplitOptionsCard from '../../../../shared/ui/config/SplitOptionsCard.jsx';

export default function XGBoostConfigPane({
  mode,
  xgb,
  xgbDefaults,
  setXGBoost,
}) {
  const useEarlyStopping = Boolean(xgb.use_early_stopping ?? xgbDefaults?.use_early_stopping);

  return (
    <Stack style={{ flex: 1, minWidth: 260 }} gap="sm">
      {/* Core */}
      <Group grow align="flex-end" wrap="wrap">
        <NumberInput
          label="Estimators"
          min={1}
          step={10}
          value={xgb.n_estimators ?? xgbDefaults?.n_estimators ?? undefined}
          onChange={(v) => setXGBoost({ n_estimators: v === '' ? undefined : v })}
        />
        <NumberInput
          label="Learning rate"
          min={0.0001}
          step={0.05}
          value={xgb.learning_rate ?? xgbDefaults?.learning_rate ?? undefined}
          onChange={(v) => setXGBoost({ learning_rate: v === '' ? undefined : v })}
        />
        <NumberInput
          label="Max depth"
          min={1}
          step={1}
          value={xgb.max_depth ?? xgbDefaults?.max_depth ?? undefined}
          onChange={(v) => setXGBoost({ max_depth: v === '' ? undefined : v })}
        />
      </Group>

      <Group grow align="flex-end" wrap="wrap">
        <NumberInput
          label="Subsample"
          min={0.05}
          max={1}
          step={0.05}
          value={xgb.subsample ?? xgbDefaults?.subsample ?? undefined}
          onChange={(v) => setXGBoost({ subsample: v === '' ? undefined : v })}
        />
        <NumberInput
          label="Colsample (bytree)"
          min={0.05}
          max={1}
          step={0.05}
          value={xgb.colsample_bytree ?? xgbDefaults?.colsample_bytree ?? undefined}
          onChange={(v) => setXGBoost({ colsample_bytree: v === '' ? undefined : v })}
        />
      </Group>

      {mode === 'advanced' && (
        <>
          <Divider my="xs" label="Regularization" labelPosition="center" />

          <Group grow align="flex-end" wrap="wrap">
            <NumberInput
              label="Lambda (L2)"
              min={0}
              step={0.1}
              value={xgb.reg_lambda ?? xgbDefaults?.reg_lambda ?? undefined}
              onChange={(v) => setXGBoost({ reg_lambda: v === '' ? undefined : v })}
            />
            <NumberInput
              label="Alpha (L1)"
              min={0}
              step={0.1}
              value={xgb.reg_alpha ?? xgbDefaults?.reg_alpha ?? undefined}
              onChange={(v) => setXGBoost({ reg_alpha: v === '' ? undefined : v })}
            />
          </Group>

          <Group grow align="flex-end" wrap="wrap">
            <NumberInput
              label="Min child weight"
              min={0}
              step={0.5}
              value={xgb.min_child_weight ?? xgbDefaults?.min_child_weight ?? undefined}
              onChange={(v) => setXGBoost({ min_child_weight: v === '' ? undefined : v })}
            />
            <NumberInput
              label="Gamma"
              min={0}
              step={0.1}
              value={xgb.gamma ?? xgbDefaults?.gamma ?? undefined}
              onChange={(v) => setXGBoost({ gamma: v === '' ? undefined : v })}
            />
          </Group>

          <Divider my="xs" label="Early stopping" labelPosition="center" />

          <Switch
            label="Use early stopping"
            checked={useEarlyStopping}
            onChange={(e) => setXGBoost({ use_early_stopping: e.currentTarget.checked })}
          />

          <Group grow align="flex-end" wrap="wrap">
            <NumberInput
              label="Early stopping rounds"
              min={1}
              step={1}
              value={xgb.early_stopping_rounds ?? xgbDefaults?.early_stopping_rounds ?? undefined}
              onChange={(v) =>
                setXGBoost({ early_stopping_rounds: v === '' ? undefined : v })
              }
              disabled={!Boolean(xgb.use_early_stopping)}
            />

            <NumberInput
              label="Eval set fraction"
              min={0.05}
              max={0.5}
              step={0.05}
              value={xgb.eval_set_fraction ?? xgbDefaults?.eval_set_fraction ?? undefined}
              onChange={(v) => setXGBoost({ eval_set_fraction: v === '' ? undefined : v })}
              disabled={!Boolean(xgb.use_early_stopping)}
            />
          </Group>

          <Divider my="xs" label="Other" labelPosition="center" />

          <Group grow align="flex-end" wrap="wrap">
            <NumberInput
              label="n_jobs"
              min={-1}
              step={1}
              value={xgb.n_jobs ?? xgbDefaults?.n_jobs ?? undefined}
              onChange={(v) => setXGBoost({ n_jobs: v === '' ? undefined : v })}
            />
            <NumberInput
              label="Random state"
              min={0}
              step={1}
              value={xgb.random_state ?? xgbDefaults?.random_state ?? undefined}
              onChange={(v) => setXGBoost({ random_state: v === '' ? undefined : v })}
            />
          </Group>
        </>
      )}

      <SplitOptionsCard
        title="Data split"
        allowedModes={['holdout', 'kfold']}
        mode={xgb.splitMode}
        onModeChange={(v) => setXGBoost({ splitMode: v })}
        trainFrac={xgb.trainFrac}
        onTrainFracChange={(v) => setXGBoost({ trainFrac: v })}
        nSplits={xgb.nSplits}
        onNSplitsChange={(v) => setXGBoost({ nSplits: v })}
        stratified={xgb.stratified}
        onStratifiedChange={(v) => setXGBoost({ stratified: v })}
        shuffle={xgb.shuffle}
        onShuffleChange={(v) => setXGBoost({ shuffle: v })}
        seed={xgb.seed}
        onSeedChange={(v) => setXGBoost({ seed: v })}
      />
    </Stack>
  );
}
