import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';

export default function KmeansSection({ m, set, sub, enums }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Clusters (n_clusters)"
          value={m.n_clusters ?? 8}
          onChange={(v) => set({ n_clusters: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Init"
          data={toSelectData(enumFromSubSchema(sub, 'init', ['k-means++', 'random']))}
          value={m.init ?? 'k-means++'}
          onChange={(v) => set({ init: v })}
        />
        <TextInput
          label="n_init (auto or int)"
          value={m.n_init == null ? 'auto' : String(m.n_init)}
          onChange={(e) => {
            const raw = e.currentTarget.value;
            const t = String(raw ?? '').trim();
            if (!t || t.toLowerCase() === 'auto') {
              set({ n_init: 'auto' });
              return;
            }
            const n = Number(t);
            set({ n_init: Number.isFinite(n) ? Math.max(1, Math.trunc(n)) : 'auto' });
          }}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter ?? 300}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol ?? 1e-4}
          onChange={(v) => set({ tol: v })}
          step={1e-5}
          min={0}
        />
        <ParamNumber
          label="Verbose"
          value={m.verbose ?? 0}
          onChange={(v) => set({ verbose: v })}
          allowDecimal={false}
          min={0}
        />
        <ParamSelect
          label="Algorithm"
          data={toSelectData(enumFromSubSchema(sub, 'algorithm', ['lloyd', 'elkan', 'auto']))}
          value={m.algorithm ?? 'lloyd'}
          onChange={(v) => set({ algorithm: v })}
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
