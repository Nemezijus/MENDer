import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function BirchSection({ m, set, sub, enums, d }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Threshold"
          value={m.threshold}

          placeholder={defaultPlaceholder(d?.threshold)}
          onChange={(v) => set({ threshold: overrideOrUndef(v, d?.threshold) })}
          step={0.01}
          min={0}
        />
        <ParamNumber
          label="Branching factor"
          value={m.branching_factor}

          placeholder={defaultPlaceholder(d?.branching_factor)}
          onChange={(v) => set({ branching_factor: overrideOrUndef(v, d?.branching_factor) })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="n_clusters (optional)"
          value={m.n_clusters}

          placeholder={defaultPlaceholder(d?.n_clusters)}
          onChange={(v) => set({ n_clusters: overrideOrUndef(v, d?.n_clusters) })}
          allowDecimal={false}
          min={1}
        />
        <ParamCheckbox
          label="Compute labels"
          checked={effectiveValue(m.compute_labels, d?.compute_labels)}
          onChange={(checked) => set({ compute_labels: overrideOrUndef(checked, d?.compute_labels) })}
        />
        <ParamCheckbox
          label="Copy"
          checked={effectiveValue(m.copy, d?.copy)}
          onChange={(checked) => set({ copy: overrideOrUndef(checked, d?.copy) })}
        />
      </ParamGrid>
  );
}
