import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';

export default function SpectralSection({ m, set, sub, enums }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Clusters (n_clusters)"
          value={m.n_clusters ?? 8}
          onChange={(v) => set({ n_clusters: v })}
          allowDecimal={false}
          min={2}
        />
        <ParamSelect
          label="Affinity"
          data={toSelectData(enumFromSubSchema(sub, 'affinity', ['rbf', 'nearest_neighbors']))}
          value={m.affinity ?? 'rbf'}
          onChange={(v) => set({ affinity: v })}
        />
        <ParamSelect
          label="Assign labels"
          data={toSelectData(enumFromSubSchema(sub, 'assign_labels', ['kmeans', 'discretize', 'cluster_qr']))}
          value={m.assign_labels ?? 'kmeans'}
          onChange={(v) => set({ assign_labels: v })}
        />
        <ParamNumber
          label="Initializations (n_init)"
          value={m.n_init ?? 10}
          onChange={(v) => set({ n_init: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Gamma"
          value={m.gamma ?? 1.0}
          onChange={(v) => set({ gamma: v })}
          step={0.1}
          min={0}
        />
        <ParamNumber
          label="Neighbours (n_neighbors)"
          value={m.n_neighbors ?? 10}
          onChange={(v) => set({ n_neighbors: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Random state"
          value={m.random_state ?? null}
          onChange={(v) => set({ random_state: v })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
