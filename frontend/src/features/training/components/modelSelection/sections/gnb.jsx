import { Stack, SimpleGrid, NumberInput, TextInput } from '@mantine/core';
import { parseCsvFloats, formatCsvFloats } from '../../../utils/modelSelectionUtils.js';

export default function GnbSection({ m, set, sub, enums }) {  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Variance smoothing"
          value={m.var_smoothing ?? 1e-9}
          onChange={(v) => set({ var_smoothing: v })}
          step={1e-9}
          min={0}
        />
        <TextInput
          label="Priors (comma-separated)"
          placeholder="e.g. 0.2, 0.8"
          value={formatCsvFloats(m.priors)}
          onChange={(e) => set({ priors: parseCsvFloats(e.currentTarget.value) })}
        />
      </SimpleGrid>
    </Stack>
  );
}
