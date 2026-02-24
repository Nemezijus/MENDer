import { Stack, SimpleGrid, NumberInput, Select } from '@mantine/core';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function LogregSection({ m, set, sub, enums }) {
  const lrPenalty = makeSelectData(sub, 'penalty', enums?.PenaltyName);
  const lrSolver = makeSelectData(sub, 'solver', enums?.LogRegSolver);
  const lrClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="C (strength)"
          value={m.C ?? 1.0}
          onChange={(v) => set({ C: v })}
          min={0}
          step={0.1}
        />
        <Select
          label="Penalty"
          data={lrPenalty}
          value={m.penalty ?? 'l2'}
          onChange={(v) => set({ penalty: v })}
        />
        <Select
          label="Solver"
          data={lrSolver}
          value={m.solver ?? 'lbfgs'}
          onChange={(v) => set({ solver: v })}
        />
        <NumberInput
          label="Max iterations"
          value={m.max_iter ?? 1000}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <Select
          label="Class weight"
          data={lrClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        {m.penalty === 'elasticnet' && (
          <NumberInput
            label="L1 ratio"
            value={m.l1_ratio ?? 0.5}
            onChange={(v) => set({ l1_ratio: v })}
            min={0}
            max={1}
            step={0.01}
          />
        )}
      </SimpleGrid>
    </Stack>
  );
}
