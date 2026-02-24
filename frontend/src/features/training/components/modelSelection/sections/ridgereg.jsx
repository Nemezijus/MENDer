import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function RidgeregSection({ m, set, sub, enums }) {
  const ridgeSolver = makeSelectData(sub, 'solver', enums?.RidgeSolver);
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
        <NumberInput
          label="Max iterations"
          value={m.max_iter ?? null}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
        />
        <NumberInput
          label="Tolerance (tol)"
          value={m.tol ?? 1e-4}
          onChange={(v) => set({ tol: v })}
          step={1e-5}
          min={0}
        />
        <NumberInput
          label="Random state"
          value={m.random_state ?? null}
          onChange={(v) => set({ random_state: v })}
          allowDecimal={false}
        />
        <Checkbox
          label="Positive coefficients"
          checked={!!m.positive}
          onChange={(e) => set({ positive: e.currentTarget.checked })}
        />
      </SimpleGrid>
    </Stack>
  );
}
