import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { maxFeatToModeVal, modeValToMaxFeat, makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function RfregSection({ m, set, sub, enums }) {
  const regTreeCriterion = makeSelectData(sub, 'criterion', enums?.RegTreeCriterion);
  const fMF = maxFeatToModeVal(m.max_features);
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="Trees (n_estimators)"
          value={m.n_estimators ?? 100}
          onChange={(v) => set({ n_estimators: v })}
          allowDecimal={false}
          min={1}
        />
        <Select
          label="Criterion"
          data={regTreeCriterion}
          value={m.criterion ?? 'squared_error'}
          onChange={(v) => set({ criterion: v })}
        />
        <NumberInput
          label="Max depth"
          value={m.max_depth ?? null}
          onChange={(v) => set({ max_depth: v })}
        />
        <NumberInput
          label="Min samples split"
          value={m.min_samples_split ?? 2}
          onChange={(v) => set({ min_samples_split: v })}
        />
        <NumberInput
          label="Min samples leaf"
          value={m.min_samples_leaf ?? 1}
          onChange={(v) => set({ min_samples_leaf: v })}
        />
        <NumberInput
          label="Min weight fraction"
          value={m.min_weight_fraction_leaf ?? 0.0}
          onChange={(v) => set({ min_weight_fraction_leaf: v })}
          step={0.01}
          min={0}
          max={1}
        />
        <Select
          label="Max features mode"
          data={[
            { value: 'sqrt', label: 'sqrt' },
            { value: 'log2', label: 'log2' },
            { value: 'int', label: 'int' },
            { value: 'float', label: 'float' },
            { value: 'none', label: 'none' },
          ]}
          value={fMF.mode}
          onChange={(mode) => set({ max_features: modeValToMaxFeat(mode, fMF.value) })}
        />
        {(fMF.mode === 'int' || fMF.mode === 'float') && (
          <NumberInput
            label="Max features value"
            value={fMF.value ?? null}
            onChange={(v) => set({ max_features: modeValToMaxFeat(fMF.mode, v) })}
            step={fMF.mode === 'int' ? 1 : 0.01}
            allowDecimal={fMF.mode === 'float'}
          />
        )}
        <NumberInput
          label="Max leaf nodes"
          value={m.max_leaf_nodes ?? null}
          onChange={(v) => set({ max_leaf_nodes: v })}
          allowDecimal={false}
        />
        <NumberInput
          label="Min impurity decrease"
          value={m.min_impurity_decrease ?? 0.0}
          onChange={(v) => set({ min_impurity_decrease: v })}
          step={0.0001}
        />
        <Checkbox
          label="Use bootstrap"
          checked={m.bootstrap ?? true}
          onChange={(e) => set({ bootstrap: e.currentTarget.checked })}
        />
        <Checkbox
          label="OOB score"
          checked={!!m.oob_score}
          onChange={(e) => set({ oob_score: e.currentTarget.checked })}
        />
        <NumberInput
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
        <NumberInput
          label="Random state"
          value={m.random_state ?? null}
          onChange={(v) => set({ random_state: v })}
          allowDecimal={false}
        />
        <Checkbox
          label="Warm start"
          checked={!!m.warm_start}
          onChange={(e) => set({ warm_start: e.currentTarget.checked })}
        />
        <NumberInput
          label="CCP alpha"
          value={m.ccp_alpha ?? 0.0}
          onChange={(v) => set({ ccp_alpha: v })}
          step={0.0001}
        />
        <NumberInput
          label="Max samples"
          value={m.max_samples ?? null}
          onChange={(v) => set({ max_samples: v })}
        />
      </SimpleGrid>
    </Stack>
  );
}
