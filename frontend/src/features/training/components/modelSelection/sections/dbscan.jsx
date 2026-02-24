import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';

export default function DbscanSection({ m, set, sub, enums }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Epsilon (eps)"
          value={m.eps ?? 0.5}
          onChange={(v) => set({ eps: v })}
          step={0.01}
          min={0}
        />
        <ParamNumber
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
        <ParamSelect
          label="Search algorithm (algorithm)"
          data={toSelectData(enumFromSubSchema(sub, 'algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']))}
          value={m.algorithm ?? 'auto'}
          onChange={(v) => set({ algorithm: v })}
        />
        <ParamNumber
          label="Leaf size (leaf_size)"
          value={m.leaf_size ?? 30}
          onChange={(v) => set({ leaf_size: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Minkowski power (p)"
          value={m.p ?? null}
          onChange={(v) => set({ p: v })}
        />
        <ParamNumber
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
