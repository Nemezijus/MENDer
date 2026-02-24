import { Stack, SimpleGrid, NumberInput, Select, Checkbox, TextInput } from '@mantine/core';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { parseCsvFloats, formatCsvFloats, makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function RidgecvSection({ m, set, sub, enums }) {
  const ridgecvGcvMode = makeSelectData(sub, 'gcv_mode', ['auto', 'svd', 'eigen', null], { includeNoneLabel: true });
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <TextInput
          label="Alphas (comma-separated)"
          placeholder="e.g. 0.1, 1.0, 10.0"
          value={formatCsvFloats(m.alphas)}
          onChange={(e) => set({ alphas: parseCsvFloats(e.currentTarget.value) || [0.1, 1.0, 10.0] })}
        />
        <Checkbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(e) => set({ fit_intercept: e.currentTarget.checked })}
        />
        <TextInput
          label="Scoring"
          placeholder="(optional)"
          value={m.scoring ?? ''}
          onChange={(e) => {
            const t = e.currentTarget.value;
            set({ scoring: t === '' ? null : t });
          }}
        />
        <NumberInput
          label="CV folds"
          value={m.cv ?? null}
          onChange={(v) => set({ cv: v })}
          allowDecimal={false}
          min={2}
        />
        <Select
          label="GCV mode"
          data={ridgecvGcvMode}
          value={m.gcv_mode == null ? 'none' : String(m.gcv_mode)}
          onChange={(v) => set({ gcv_mode: fromSelectNullable(v) })}
        />
        <Checkbox
          label="Alpha per target"
          checked={!!m.alpha_per_target}
          onChange={(e) => set({ alpha_per_target: e.currentTarget.checked })}
        />
      </SimpleGrid>
    </Stack>
  );
}
