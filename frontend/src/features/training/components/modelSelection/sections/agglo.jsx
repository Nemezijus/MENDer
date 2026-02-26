import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function AggloSection({ m, set, sub, enums, d }) {
  const defComputeFullTree = d?.compute_full_tree;
  const defComputeFullTreeValue =
    defComputeFullTree === true ? 'true' : defComputeFullTree === false ? 'false' : defComputeFullTree;

  const computeFullTreeValue =
    m.compute_full_tree === undefined
      ? undefined
      : m.compute_full_tree === true
      ? 'true'
      : m.compute_full_tree === false
      ? 'false'
      : 'auto';

  return (
    <ParamGrid>
      <ParamNumber
        label="Clusters (n_clusters)"
        value={m.n_clusters}
        placeholder={defaultPlaceholder(d?.n_clusters)}
        onChange={(v) => set({ n_clusters: overrideOrUndef(v, d?.n_clusters) })}
        allowDecimal={false}
        min={2}
      />

      <ParamSelect
        label="Linkage"
        data={toSelectData(enumFromSubSchema(sub, 'linkage', ['ward', 'complete', 'average', 'single']))}
        value={m.linkage}
        placeholder={defaultPlaceholder(d?.linkage)}
        onChange={(v) => set({ linkage: overrideOrUndef(v, d?.linkage) })}
      />

      <TextInput
        label="Distance metric (metric)"
        value={m.metric ?? ''}
        placeholder={defaultPlaceholder(d?.metric)}
        onChange={(e) => {
          const t = String(e.currentTarget.value ?? '');
          set({ metric: overrideOrUndef(t.trim() === '' ? undefined : t, d?.metric) });
        }}
      />

      <ParamNumber
        label="Distance threshold"
        value={m.distance_threshold}
        placeholder={defaultPlaceholder(d?.distance_threshold)}
        onChange={(v) => set({ distance_threshold: overrideOrUndef(v, d?.distance_threshold) })}
        min={0}
      />

      <ParamSelect
        label="Compute full tree"
        data={[
          { value: 'auto', label: 'auto' },
          { value: 'true', label: 'true' },
          { value: 'false', label: 'false' },
        ]}
        value={computeFullTreeValue}
        placeholder={defaultPlaceholder(defComputeFullTreeValue)}
        onChange={(v) => {
          if (v === undefined) {
            set({ compute_full_tree: undefined });
            return;
          }

          const next = v === 'true' ? true : v === 'false' ? false : 'auto';
          set({ compute_full_tree: overrideOrUndef(next, defComputeFullTree) });
        }}
      />

      <ParamCheckbox
        label="Compute distances"
        checked={effectiveValue(m.compute_distances, d?.compute_distances)}
        onChange={(checked) => set({ compute_distances: overrideOrUndef(checked, d?.compute_distances) })}
      />
    </ParamGrid>
  );
}
