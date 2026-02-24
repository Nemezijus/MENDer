import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';

export default function GmmSection({ m, set, sub, enums }) {  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Components (n_components)"
          value={m.n_components ?? 1}
          onChange={(v) => set({ n_components: v })}
          allowDecimal={false}
          min={1}
        />
        <Select
          label="Covariance type"
          data={toSelectData(enumFromSubSchema(sub, 'covariance_type', ['full', 'tied', 'diag', 'spherical']))}
          value={m.covariance_type ?? 'full'}
          onChange={(v) => set({ covariance_type: v })}
        />
        <NumberInput
          label="Tolerance (tol)"
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={1e-4}
          min={0}
        />
        <NumberInput
          label="Regularization (reg_covar)"
          value={m.reg_covar ?? 1e-6}
          onChange={(v) => set({ reg_covar: v })}
          step={1e-6}
          min={0}
        />
        <NumberInput
          label="Max iterations"
          value={m.max_iter ?? 100}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="Initializations (n_init)"
          value={m.n_init ?? 1}
          onChange={(v) => set({ n_init: v })}
          allowDecimal={false}
          min={1}
        />
        <Select
          label="Init params"
          data={toSelectData(enumFromSubSchema(sub, 'init_params', ['kmeans', 'k-means++', 'random', 'random_from_data']))}
          value={m.init_params ?? 'kmeans'}
          onChange={(v) => set({ init_params: v })}
        />
        <Checkbox
          label="Warm start"
          checked={!!m.warm_start}
          onChange={(e) => set({ warm_start: e.currentTarget.checked })}
        />
        <NumberInput
          label="Verbose"
          value={m.verbose ?? 0}
          onChange={(v) => set({ verbose: v })}
          allowDecimal={false}
          min={0}
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
