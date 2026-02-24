import { Stack, SimpleGrid, NumberInput, Checkbox } from '@mantine/core';

export default function MeanshiftSection({ m, set, sub, enums }) {  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Bandwidth (optional)"
          value={m.bandwidth ?? null}
          onChange={(v) => set({ bandwidth: v })}
          min={0}
        />
        <Checkbox
          label="Bin seeding"
          checked={!!m.bin_seeding}
          onChange={(e) => set({ bin_seeding: e.currentTarget.checked })}
        />
        <NumberInput
          label="Min bin freq"
          value={m.min_bin_freq ?? 1}
          onChange={(v) => set({ min_bin_freq: v })}
          allowDecimal={false}
          min={1}
        />
        <Checkbox
          label="Cluster all"
          checked={m.cluster_all ?? true}
          onChange={(e) => set({ cluster_all: e.currentTarget.checked })}
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
