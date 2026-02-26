import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function MeanshiftSection({ m, set, sub, enums, d }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Bandwidth (optional)"
          value={m.bandwidth}

          placeholder={defaultPlaceholder(d?.bandwidth)}
          onChange={(v) => set({ bandwidth: overrideOrUndef(v, d?.bandwidth) })}
          min={0}
        />
        <ParamCheckbox
          label="Bin seeding"
          checked={effectiveValue(m.bin_seeding, d?.bin_seeding)}
          onChange={(checked) => set({ bin_seeding: overrideOrUndef(checked, d?.bin_seeding) })}
        />
        <ParamNumber
          label="Min bin freq"
          value={m.min_bin_freq}

          placeholder={defaultPlaceholder(d?.min_bin_freq)}
          onChange={(v) => set({ min_bin_freq: overrideOrUndef(v, d?.min_bin_freq) })}
          allowDecimal={false}
          min={1}
        />
        <ParamCheckbox
          label="Cluster all"
          checked={effectiveValue(m.cluster_all, d?.cluster_all)}
          onChange={(checked) => set({ cluster_all: overrideOrUndef(checked, d?.cluster_all) })}
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
