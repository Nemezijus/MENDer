import { Stack, Group, Divider } from '@mantine/core';

import ParamNumber from '../common/ParamNumber.jsx';
import ParamSwitch from '../common/ParamSwitch.jsx';

export default function XGBoostConfigPane({ mode, xgb, xgbDefaults, setXGBoost }) {
  const useEarlyStopping = Boolean(xgb.use_early_stopping ?? xgbDefaults?.use_early_stopping);

  return (
    <Stack style={{ width: '100%' }} gap="sm">
      {/* Core */}
      <Group grow align="flex-end" wrap="wrap">
        <ParamNumber
          label="Estimators"
          min={1}
          step={10}
          value={xgb.n_estimators ?? xgbDefaults?.n_estimators ?? undefined}
          onChange={(v) => setXGBoost({ n_estimators: v })}
        />
        <ParamNumber
          label="Learning rate"
          min={0.0001}
          step={0.05}
          value={xgb.learning_rate ?? xgbDefaults?.learning_rate ?? undefined}
          onChange={(v) => setXGBoost({ learning_rate: v })}
        />
        <ParamNumber
          label="Max depth"
          min={1}
          step={1}
          value={xgb.max_depth ?? xgbDefaults?.max_depth ?? undefined}
          onChange={(v) => setXGBoost({ max_depth: v })}
        />
      </Group>

      <Group grow align="flex-end" wrap="wrap">
        <ParamNumber
          label="Subsample"
          min={0.05}
          max={1}
          step={0.05}
          value={xgb.subsample ?? xgbDefaults?.subsample ?? undefined}
          onChange={(v) => setXGBoost({ subsample: v })}
        />
        <ParamNumber
          label="Colsample (bytree)"
          min={0.05}
          max={1}
          step={0.05}
          value={xgb.colsample_bytree ?? xgbDefaults?.colsample_bytree ?? undefined}
          onChange={(v) => setXGBoost({ colsample_bytree: v })}
        />
      </Group>

      {mode === 'advanced' && (
        <>
          <Divider my="xs" label="Regularization" labelPosition="center" />

          <Group grow align="flex-end" wrap="wrap">
            <ParamNumber
              label="Lambda (L2)"
              min={0}
              step={0.1}
              value={xgb.reg_lambda ?? xgbDefaults?.reg_lambda ?? undefined}
              onChange={(v) => setXGBoost({ reg_lambda: v })}
            />
            <ParamNumber
              label="Alpha (L1)"
              min={0}
              step={0.1}
              value={xgb.reg_alpha ?? xgbDefaults?.reg_alpha ?? undefined}
              onChange={(v) => setXGBoost({ reg_alpha: v })}
            />
          </Group>

          <Group grow align="flex-end" wrap="wrap">
            <ParamNumber
              label="Min child weight"
              min={0}
              step={0.5}
              value={xgb.min_child_weight ?? xgbDefaults?.min_child_weight ?? undefined}
              onChange={(v) => setXGBoost({ min_child_weight: v })}
            />
            <ParamNumber
              label="Gamma"
              min={0}
              step={0.1}
              value={xgb.gamma ?? xgbDefaults?.gamma ?? undefined}
              onChange={(v) => setXGBoost({ gamma: v })}
            />
          </Group>

          <Divider my="xs" label="Early stopping" labelPosition="center" />

          <ParamSwitch
            label="Use early stopping"
            checked={useEarlyStopping}
            onChange={(checked) => setXGBoost({ use_early_stopping: checked })}
          />

          <Group grow align="flex-end" wrap="wrap">
            <ParamNumber
              label="Early stopping rounds"
              min={1}
              step={1}
              value={xgb.early_stopping_rounds ?? xgbDefaults?.early_stopping_rounds ?? undefined}
              onChange={(v) => setXGBoost({ early_stopping_rounds: v })}
              disabled={!useEarlyStopping}
            />

            <ParamNumber
              label="Eval set fraction"
              min={0.05}
              max={0.5}
              step={0.05}
              value={xgb.eval_set_fraction ?? xgbDefaults?.eval_set_fraction ?? undefined}
              onChange={(v) => setXGBoost({ eval_set_fraction: v })}
              disabled={!useEarlyStopping}
            />
          </Group>

          <Divider my="xs" label="Other" labelPosition="center" />

          <Group grow align="flex-end" wrap="wrap">
            <ParamNumber
              label="n_jobs"
              min={-1}
              step={1}
              value={xgb.n_jobs ?? xgbDefaults?.n_jobs ?? undefined}
              onChange={(v) => setXGBoost({ n_jobs: v })}
            />
            <ParamNumber
              label="Random state"
              min={0}
              step={1}
              value={xgb.random_state ?? xgbDefaults?.random_state ?? undefined}
              onChange={(v) => setXGBoost({ random_state: v })}
            />
          </Group>
        </>
      )}
    </Stack>
  );
}
