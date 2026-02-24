import { Stack, SimpleGrid, NumberInput, Select } from '@mantine/core';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';

export default function SpectralSection({ m, set, sub, enums }) {  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Clusters (n_clusters)"
          value={m.n_clusters ?? 8}
          onChange={(v) => set({ n_clusters: v })}
          allowDecimal={false}
          min={2}
        />
        <Select
          label="Affinity"
          data={toSelectData(enumFromSubSchema(sub, 'affinity', ['rbf', 'nearest_neighbors']))}
          value={m.affinity ?? 'rbf'}
          onChange={(v) => set({ affinity: v })}
        />
        <Select
          label="Assign labels"
          data={toSelectData(enumFromSubSchema(sub, 'assign_labels', ['kmeans', 'discretize', 'cluster_qr']))}
          value={m.assign_labels ?? 'kmeans'}
          onChange={(v) => set({ assign_labels: v })}
        />
        <NumberInput
          label="Initializations (n_init)"
          value={m.n_init ?? 10}
          onChange={(v) => set({ n_init: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="Gamma"
          value={m.gamma ?? 1.0}
          onChange={(v) => set({ gamma: v })}
          step={0.1}
          min={0}
        />
        <NumberInput
          label="Neighbours (n_neighbors)"
          value={m.n_neighbors ?? 10}
          onChange={(v) => set({ n_neighbors: v })}
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
