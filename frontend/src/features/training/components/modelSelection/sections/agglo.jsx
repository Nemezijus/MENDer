import { Stack, SimpleGrid, NumberInput, Select, Checkbox, TextInput } from '@mantine/core';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';

export default function AggloSection({ m, set, sub, enums }) {  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Clusters (n_clusters)"
          value={m.n_clusters ?? 2}
          onChange={(v) => set({ n_clusters: v })}
          allowDecimal={false}
          min={2}
        />
        <Select
          label="Linkage"
          data={toSelectData(enumFromSubSchema(sub, 'linkage', ['ward', 'complete', 'average', 'single']))}
          value={m.linkage ?? 'ward'}
          onChange={(v) => set({ linkage: v })}
        />
        <TextInput
          label="Distance metric (metric)"
          value={m.metric ?? 'euclidean'}
          onChange={(e) => set({ metric: e.currentTarget.value })}
        />
        <NumberInput
          label="Distance threshold"
          value={m.distance_threshold ?? null}
          onChange={(v) => set({ distance_threshold: v })}
          min={0}
        />
        <Select
          label="Compute full tree"
          data={[
            { value: 'auto', label: 'auto' },
            { value: 'true', label: 'true' },
            { value: 'false', label: 'false' },
          ]}
          value={
            m.compute_full_tree === true
              ? 'true'
              : m.compute_full_tree === false
              ? 'false'
              : 'auto'
          }
          onChange={(v) => {
            if (v === 'true') set({ compute_full_tree: true });
            else if (v === 'false') set({ compute_full_tree: false });
            else set({ compute_full_tree: 'auto' });
          }}
        />
        <Checkbox
          label="Compute distances"
          checked={!!m.compute_distances}
          onChange={(e) => set({ compute_distances: e.currentTarget.checked })}
        />
      </SimpleGrid>
    </Stack>
  );
}
