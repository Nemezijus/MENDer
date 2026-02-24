import { Stack, SimpleGrid, NumberInput, Select } from '@mantine/core';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { maxFeatToModeVal, modeValToMaxFeat, makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function TreeSection({ m, set, sub, enums }) {
  const treeCriterion = makeSelectData(sub, 'criterion', enums?.TreeCriterion);
  const treeSplitter = makeSelectData(sub, 'splitter', enums?.TreeSplitter);
  const treeClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  const tMF = maxFeatToModeVal(m.max_features);
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <Select
          label="Criterion"
          data={treeCriterion}
          value={m.criterion ?? 'gini'}
          onChange={(v) => set({ criterion: v })}
        />
        <Select
          label="Splitter"
          data={treeSplitter}
          value={m.splitter ?? 'best'}
          onChange={(v) => set({ splitter: v })}
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
          onChange={(v) =>
            set({ min_weight_fraction_leaf: v })
          }
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
          value={tMF.mode}
          onChange={(mode) =>
            set({ max_features: modeValToMaxFeat(mode, tMF.value) })
          }
        />
        {(tMF.mode === 'int' || tMF.mode === 'float') && (
          <NumberInput
            label="Max features value"
            value={tMF.value ?? null}
            onChange={(v) =>
              set({
                max_features: modeValToMaxFeat(tMF.mode, v),
              })
            }
            step={tMF.mode === 'int' ? 1 : 0.01}
            allowDecimal={tMF.mode === 'float'}
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
          onChange={(v) =>
            set({ min_impurity_decrease: v })
          }
          step={0.0001}
        />
        <Select
          label="Class weight"
          data={treeClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        <NumberInput
          label="CCP alpha"
          value={m.ccp_alpha ?? 0.0}
          onChange={(v) => set({ ccp_alpha: v })}
          step={0.0001}
        />
      </SimpleGrid>
    </Stack>
  );
}
