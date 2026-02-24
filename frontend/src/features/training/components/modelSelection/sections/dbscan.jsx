import { Stack, SimpleGrid, NumberInput, Select, TextInput } from '@mantine/core';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';

export default function DbscanSection({ m, set, sub, enums }) {  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Epsilon (eps)"
          value={m.eps ?? 0.5}
          onChange={(v) => set({ eps: v })}
          step={0.01}
          min={0}
        />
        <NumberInput
          label="Minimum samples (min_samples)"
          value={m.min_samples ?? 5}
          onChange={(v) => set({ min_samples: v })}
          allowDecimal={false}
          min={1}
        />
        <TextInput
          label="Distance metric (metric)"
          value={m.metric ?? 'euclidean'}
          onChange={(e) => set({ metric: e.currentTarget.value })}
        />
        <Select
          label="Search algorithm (algorithm)"
          data={toSelectData(enumFromSubSchema(sub, 'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']))}
          value={m.algorithm ?? 'auto'}
          onChange={(v) => set({ algorithm: v })}
        />
        <NumberInput
          label="Leaf size (leaf_size)"
          value={m.leaf_size ?? 30}
          onChange={(v) => set({ leaf_size: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="Minkowski power (p)"
          value={m.p ?? null}
          onChange={(v) => set({ p: v })}
        />
        <NumberInput
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
      </SimpleGrid>
    </Stack>
  );
}
