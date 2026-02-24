import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function LinsvrSection({ m, set, sub, enums }) {
  const linsvrLoss = makeSelectData(sub, 'loss', enums?.LinearSVRLoss);
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="C"
          value={m.C ?? 1.0}
          onChange={(v) => set({ C: v })}
          min={0}
          step={0.1}
        />
        <Select
          label="Loss"
          data={linsvrLoss}
          value={m.loss ?? 'epsilon_insensitive'}
          onChange={(v) => set({ loss: v })}
        />
        <NumberInput
          label="Epsilon"
          value={m.epsilon ?? 0.0}
          onChange={(v) => set({ epsilon: v })}
          min={0}
          step={0.01}
        />
        <Checkbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
        />
        <NumberInput
          label="Intercept scaling"
          value={m.intercept_scaling ?? 1.0}
          onChange={(v) => set({ intercept_scaling: v })}
          min={0}
          step={0.1}
        />
        <Checkbox
          label="Dual"
          checked={m.dual ?? true}
          onChange={(e) => set({ dual: e.currentTarget.checked })}
        />
        <NumberInput
          label="Tolerance (tol)"
          value={m.tol ?? 1e-4}
          onChange={(v) => set({ tol: v })}
          step={1e-5}
          min={0}
        />
        <NumberInput
          label="Max iterations"
          value={m.max_iter ?? 1000}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="Random state"
          value={m.random_state ?? null}
          onChange={(v) => set({ random_state: v })}
          allowDecimal={false}
        />
      </SimpleGrid>
    </Stack>
  );
}
