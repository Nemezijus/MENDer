import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';
import { defaultPlaceholder, overrideOrUndef } from '../utils/paramDefaults.js';

export default function DbscanSection({ m, set, sub, enums, d }) {
  return (
    <ParamGrid>
      <ParamNumber
        label="Epsilon (eps)"
        value={m.eps}
        placeholder={defaultPlaceholder(d?.eps)}
        onChange={(v) => set({ eps: overrideOrUndef(v, d?.eps) })}
        step={0.01}
        min={0}
      />

      <ParamNumber
        label="Minimum samples (min_samples)"
        value={m.min_samples}
        placeholder={defaultPlaceholder(d?.min_samples)}
        onChange={(v) => set({ min_samples: overrideOrUndef(v, d?.min_samples) })}
        allowDecimal={false}
        min={1}
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

      <ParamSelect
        label="Search algorithm (algorithm)"
        data={toSelectData(enumFromSubSchema(sub, 'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']))}
        value={m.algorithm}
        placeholder={defaultPlaceholder(d?.algorithm)}
        onChange={(v) => set({ algorithm: overrideOrUndef(v, d?.algorithm) })}
      />

      <ParamNumber
        label="Leaf size (leaf_size)"
        value={m.leaf_size}
        placeholder={defaultPlaceholder(d?.leaf_size)}
        onChange={(v) => set({ leaf_size: overrideOrUndef(v, d?.leaf_size) })}
        allowDecimal={false}
        min={1}
      />

      <ParamNumber
        label="Minkowski power (p)"
        value={m.p}
        placeholder={defaultPlaceholder(d?.p)}
        onChange={(v) => set({ p: overrideOrUndef(v, d?.p) })}
      />

      <ParamNumber
        label="Jobs (n_jobs)"
        value={m.n_jobs}
        placeholder={defaultPlaceholder(d?.n_jobs)}
        onChange={(v) => set({ n_jobs: overrideOrUndef(v, d?.n_jobs) })}
        allowDecimal={false}
      />
    </ParamGrid>
  );
}
