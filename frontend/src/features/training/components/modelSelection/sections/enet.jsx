import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function EnetSection({ m, set, sub, enums }) {
  const cdSelection = makeSelectData(sub, 'selection', enums?.CoordinateDescentSelection);
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Alpha"
          value={m.alpha ?? 1.0}
          onChange={(v) => set({ alpha: v })}
          min={0}
          step={0.1}
        />
        <NumberInput
          label="L1 ratio"
          value={m.l1_ratio ?? 0.5}
          onChange={(v) => set({ l1_ratio: v })}
          min={0}
          max={1}
          step={0.05}
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
