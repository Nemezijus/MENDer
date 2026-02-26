import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';
import { defaultPlaceholder, overrideOrUndef } from '../utils/paramDefaults.js';

export default function SpectralSection({ m, set, sub, enums, d }) {
  const affinityData = toSelectData(enumFromSubSchema(sub, 'affinity'));
  const assignLabelsData = toSelectData(enumFromSubSchema(sub, 'assign_labels'));

  const affinityUnavailable = affinityData.length === 0;
  const assignLabelsUnavailable = assignLabelsData.length === 0;

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
        label="Affinity"
        data={affinityData}
        value={m.affinity}
        disabled={affinityUnavailable}
        placeholder={affinityUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.affinity)}
        description={affinityUnavailable ? 'Schema did not provide affinity options.' : undefined}
        onChange={(v) => set({ affinity: overrideOrUndef(v, d?.affinity) })}
      />

      <ParamSelect
        label="Assign labels"
        data={assignLabelsData}
        value={m.assign_labels}
        disabled={assignLabelsUnavailable}
        placeholder={assignLabelsUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.assign_labels)}
        description={assignLabelsUnavailable ? 'Schema did not provide assign_labels options.' : undefined}
        onChange={(v) => set({ assign_labels: overrideOrUndef(v, d?.assign_labels) })}
      />

      <ParamNumber
        label="Initializations (n_init)"
        value={m.n_init}
        placeholder={defaultPlaceholder(d?.n_init)}
        onChange={(v) => set({ n_init: overrideOrUndef(v, d?.n_init) })}
        allowDecimal={false}
        min={1}
      />

      <ParamNumber
        label="Gamma"
        value={m.gamma}
        placeholder={defaultPlaceholder(d?.gamma)}
        onChange={(v) => set({ gamma: overrideOrUndef(v, d?.gamma) })}
        step={0.1}
        min={0}
      />

      <ParamNumber
        label="Neighbours (n_neighbors)"
        value={m.n_neighbors}
        placeholder={defaultPlaceholder(d?.n_neighbors)}
        onChange={(v) => set({ n_neighbors: overrideOrUndef(v, d?.n_neighbors) })}
        allowDecimal={false}
        min={1}
      />

      <ParamNumber
        label="Random state"
        value={m.random_state}
        placeholder={defaultPlaceholder(d?.random_state)}
        onChange={(v) => set({ random_state: overrideOrUndef(v, d?.random_state) })}
        allowDecimal={false}
      />
    </ParamGrid>
  );
}
