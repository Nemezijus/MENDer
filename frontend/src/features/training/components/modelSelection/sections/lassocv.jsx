import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function LassocvSection({ m, set, sub, enums }) {
  const cdSelection = makeSelectData(sub, 'selection', enums?.CoordinateDescentSelection);
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="eps"
          value={m.eps ?? 1e-3}
          onChange={(v) => set({ eps: v })}
          min={0}
          step={1e-4}
        />
        <NumberInput
          label="n_alphas"
          value={m.n_alphas ?? 100}
          onChange={(v) => set({ n_alphas: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="CV folds"
          value={m.cv ?? 5}
          onChange={(v) => set({ cv: v })}
          allowDecimal={false}
          min={2}
        />
        <Checkbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
        />
        <Select
          label="Selection"
          data={cdSelection}
          value={m.selection ?? 'cyclic'}
          onChange={(v) => set({ selection: v })}
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
          value={m.tol ?? 1e-4}
          onChange={(v) => set({ tol: v })}
          step={1e-5}
          min={0}
        />
        <NumberInput
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
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
