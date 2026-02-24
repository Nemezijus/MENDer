import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function SgdSection({ m, set, sub, enums }) {
  const sgdLoss = makeSelectData(sub, 'loss', enums?.SGDLoss);
  const sgdPenalty = makeSelectData(sub, 'penalty', enums?.SGDPenalty);
  const sgdLR = makeSelectData(sub, 'learning_rate', enums?.SGDLearningRate);
  const sgdClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  const sgdAvgMode = typeof m.average === 'number' ? 'int' : (m.average ? 'true' : 'false');
  const sgdAvgValue = typeof m.average === 'number' ? m.average : 10;
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <Select
          label="Loss"
          data={sgdLoss}
          value={m.loss ?? 'hinge'}
          onChange={(v) => set({ loss: v })}
        />
        <Select
          label="Penalty"
          data={sgdPenalty}
          value={m.penalty ?? 'l2'}
          onChange={(v) => set({ penalty: v })}
        />
        <NumberInput
          label="Alpha"
          value={m.alpha ?? 0.0001}
          onChange={(v) => set({ alpha: v })}
          min={0}
          step={0.0001}
        />
        <NumberInput
          label="L1 ratio"
          value={m.l1_ratio ?? 0.15}
          onChange={(v) => set({ l1_ratio: v })}
          min={0}
          max={1}
          step={0.01}
        />
        <Checkbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
        />
        <NumberInput
          label="Max iterations"
          value={m.max_iter ?? 1000}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="Tolerance (tol)"
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={0.0001}
          min={0}
        />
        <Checkbox
          label="Shuffle"
          checked={m.shuffle ?? true}
          onChange={(e) => set({ shuffle: e.currentTarget.checked })}
        />
        <Select
          label="Learning rate"
          data={sgdLR}
          value={m.learning_rate ?? 'optimal'}
          onChange={(v) => set({ learning_rate: v })}
        />
        <NumberInput
          label="Eta0"
          value={m.eta0 ?? 0.0}
          onChange={(v) => set({ eta0: v })}
          min={0}
          step={0.01}
        />
        <NumberInput
          label="Power t"
          value={m.power_t ?? 0.5}
          onChange={(v) => set({ power_t: v })}
          step={0.01}
        />
        <Checkbox
          label="Early stopping"
          checked={!!m.early_stopping}
          onChange={(e) => set({ early_stopping: e.currentTarget.checked })}
        />
        <NumberInput
          label="Validation fraction"
          value={m.validation_fraction ?? 0.1}
          onChange={(v) => set({ validation_fraction: v })}
          min={0}
          max={1}
          step={0.01}
        />
        <NumberInput
          label="No-change rounds"
          value={m.n_iter_no_change ?? 5}
          onChange={(v) => set({ n_iter_no_change: v })}
          allowDecimal={false}
          min={1}
        />
        <Select
          label="Class weight"
          data={sgdClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        <Select
          label="Average"
          data={[
            { value: 'false', label: 'false' },
            { value: 'true', label: 'true' },
            { value: 'int', label: 'int' },
          ]}
          value={sgdAvgMode}
          onChange={(mode) => {
            if (mode === 'int') set({ average: sgdAvgValue });
            else if (mode === 'true') set({ average: true });
            else set({ average: false });
          }}
        />
        {sgdAvgMode === 'int' && (
          <NumberInput
            label="Average window"
            value={sgdAvgValue}
            onChange={(v) => set({ average: v })}
            allowDecimal={false}
            min={1}
          />
        )}
        <NumberInput
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
      </SimpleGrid>
    </Stack>
  );
}
