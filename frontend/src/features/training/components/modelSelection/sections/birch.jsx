import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';

export default function BirchSection({ m, set, sub, enums }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Threshold"
          value={m.threshold ?? 0.5}
          onChange={(v) => set({ threshold: v })}
          step={0.01}
          min={0}
        />
        <ParamNumber
          label="Branching factor"
          value={m.branching_factor ?? 50}
          onChange={(v) => set({ branching_factor: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="n_clusters (optional)"
          value={m.n_clusters ?? 3}
          onChange={(v) => set({ n_clusters: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamCheckbox
          label="Compute labels"
          checked={m.compute_labels ?? true}
          onChange={(checked) => set({ compute_labels: checked })}
        />
        <ParamCheckbox
          label="Copy"
          checked={m.copy ?? true}
          onChange={(checked) => set({ copy: checked })}
        />
      </ParamGrid>
  );
}
