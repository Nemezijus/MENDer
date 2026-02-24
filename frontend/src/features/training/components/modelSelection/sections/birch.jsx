import { Stack, SimpleGrid, NumberInput, Checkbox } from '@mantine/core';

export default function BirchSection({ m, set, sub, enums }) {  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Threshold"
          value={m.threshold ?? 0.5}
          onChange={(v) => set({ threshold: v })}
          step={0.01}
          min={0}
        />
        <NumberInput
          label="Branching factor"
          value={m.branching_factor ?? 50}
          onChange={(v) => set({ branching_factor: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="n_clusters (optional)"
          value={m.n_clusters ?? 3}
          onChange={(v) => set({ n_clusters: v })}
          allowDecimal={false}
          min={1}
        />
        <Checkbox
          label="Compute labels"
          checked={m.compute_labels ?? true}
          onChange={(e) => set({ compute_labels: e.currentTarget.checked })}
        />
        <Checkbox
          label="Copy"
          checked={m.copy ?? true}
          onChange={(e) => set({ copy: e.currentTarget.checked })}
        />
      </SimpleGrid>
    </Stack>
  );
}
