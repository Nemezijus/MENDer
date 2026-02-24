import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function RidgeSection({ m, set, sub, enums }) {
  const ridgeSolver = makeSelectData(sub, 'solver', enums?.RidgeSolver);
  const ridgeClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Alpha (regularization)"
          value={m.alpha ?? 1.0}
          onChange={(v) => set({ alpha: v })}
          min={0}
          step={0.1}
        />
        <Select
          label="Solver"
          data={ridgeSolver}
          value={m.solver ?? 'auto'}
          onChange={(v) => set({ solver: v })}
        />
        <Checkbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
        />
        <Select
          label="Class weight"
          data={ridgeClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        <NumberInput
          label="Max iterations"
          value={m.max_iter ?? null}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="Tolerance (tol)"
          value={m.tol ?? 1e-4}
          onChange={(v) => set({ tol: v })}
          step={1e-5}
          min={0}
        />
      </SimpleGrid>
    </Stack>
  );
}
