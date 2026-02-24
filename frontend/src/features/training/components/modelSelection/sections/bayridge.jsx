import { Stack, SimpleGrid, NumberInput, Checkbox } from '@mantine/core';

export default function BayridgeSection({ m, set, sub, enums }) {  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Iterations (n_iter)"
          value={m.n_iter ?? 300}
          onChange={(v) => set({ n_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="Tolerance (tol)"
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={1e-4}
          min={0}
        />
        <NumberInput
          label="alpha_1"
          value={m.alpha_1 ?? 1e-6}
          onChange={(v) => set({ alpha_1: v })}
          step={1e-6}
          min={0}
        />
        <NumberInput
          label="alpha_2"
          value={m.alpha_2 ?? 1e-6}
          onChange={(v) => set({ alpha_2: v })}
          step={1e-6}
          min={0}
        />
        <NumberInput
          label="lambda_1"
          value={m.lambda_1 ?? 1e-6}
          onChange={(v) => set({ lambda_1: v })}
          step={1e-6}
          min={0}
        />
        <NumberInput
          label="lambda_2"
          value={m.lambda_2 ?? 1e-6}
          onChange={(v) => set({ lambda_2: v })}
          step={1e-6}
          min={0}
        />
        <Checkbox
          label="Compute score"
          checked={!!m.compute_score}
          onChange={(e) => set({ compute_score: e.currentTarget.checked })}
        />
        <Checkbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
        />
        <Checkbox
          label="Copy X"
          checked={m.copy_X ?? true}
          onChange={(e) => set({ copy_X: e.currentTarget.checked })}
        />
        <Checkbox
          label="Verbose"
          checked={!!m.verbose}
          onChange={(e) => set({ verbose: e.currentTarget.checked })}
        />
      </SimpleGrid>
    </Stack>
  );
}
