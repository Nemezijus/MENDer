import { Stack, SimpleGrid, NumberInput, Select, TextInput } from '@mantine/core';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function HgbSection({ m, set, sub, enums }) {
  const hgbLoss = makeSelectData(sub, 'loss', enums?.HGBLoss);
  const hgbES = m.early_stopping === 'auto' ? 'auto' : (m.early_stopping ? 'true' : 'false');
  const hgbMaxFeaturesValue = typeof m.max_features === 'number' ? m.max_features : 1.0;
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <Select
          label="Loss"
          data={hgbLoss}
          value={m.loss ?? 'log_loss'}
          onChange={(v) => set({ loss: v })}
        />
        <NumberInput
          label="Learning rate"
          value={m.learning_rate ?? 0.1}
          onChange={(v) => set({ learning_rate: v })}
          min={0}
          step={0.01}
        />
        <NumberInput
          label="Iterations (max_iter)"
          value={m.max_iter ?? 100}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="Max leaf nodes"
          value={m.max_leaf_nodes ?? 31}
          onChange={(v) => set({ max_leaf_nodes: v })}
          allowDecimal={false}
          min={2}
        />
        <NumberInput
          label="Max depth"
          value={m.max_depth ?? null}
          onChange={(v) => set({ max_depth: v })}
          allowDecimal={false}
        />
        <NumberInput
          label="Min samples leaf"
          value={m.min_samples_leaf ?? 20}
          onChange={(v) => set({ min_samples_leaf: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="L2 regularization"
          value={m.l2_regularization ?? 0.0}
          onChange={(v) => set({ l2_regularization: v })}
          min={0}
          step={0.01}
        />
        <NumberInput
          label="Max features (fraction)"
          value={hgbMaxFeaturesValue}
          onChange={(v) => {
            // Mantine NumberInput may provide a string; backend expects a number.
            if (v == null || v === '') set({ max_features: null });
            else set({ max_features: typeof v === 'number' ? v : Number(v) });
          }}
          min={0}
          max={1}
          step={0.01}
        />
        <NumberInput
          label="Max bins"
          value={m.max_bins ?? 255}
          onChange={(v) => set({ max_bins: v })}
          allowDecimal={false}
          min={2}
        />
        <Select
          label="Early stopping"
          data={[
            { value: 'auto', label: 'auto' },
            { value: 'true', label: 'true' },
            { value: 'false', label: 'false' },
          ]}
          value={hgbES}
          onChange={(v) => {
            if (v === 'auto') set({ early_stopping: 'auto' });
            else if (v === 'true') set({ early_stopping: true });
            else set({ early_stopping: false });
          }}
        />
        <TextInput
          label="Scoring"
          placeholder="loss"
          value={m.scoring ?? 'loss'}
          onChange={(e) => {
            const t = e.currentTarget.value;
            set({ scoring: t === '' ? null : t });
          }}
        />
        <NumberInput
          label="Validation fraction"
          value={m.validation_fraction ?? 0.1}
          onChange={(v) => set({ validation_fraction: v })}
          min={0}
          max={1}
          step={0.01}
        />
        <NumberInput
          label="No-change rounds"
          value={m.n_iter_no_change ?? 10}
          onChange={(v) => set({ n_iter_no_change: v })}
          allowDecimal={false}
          min={1}
        />
        <NumberInput
          label="Tolerance"
          value={m.tol ?? 1e-7}
          onChange={(v) => set({ tol: v })}
          step={1e-7}
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
