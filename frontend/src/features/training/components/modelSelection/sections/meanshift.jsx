import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';

export default function MeanshiftSection({ m, set, sub, enums }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Bandwidth (optional)"
          value={m.bandwidth ?? null}
          onChange={(v) => set({ bandwidth: v })}
          min={0}
        />
        <ParamCheckbox
          label="Bin seeding"
          checked={!!m.bin_seeding}
          onChange={(checked) => set({ bin_seeding: checked })}
        />
        <ParamNumber
          label="Min bin freq"
          value={m.min_bin_freq ?? 1}
          onChange={(v) => set({ min_bin_freq: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamCheckbox
          label="Cluster all"
          checked={m.cluster_all ?? true}
          onChange={(checked) => set({ cluster_all: checked })}
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
