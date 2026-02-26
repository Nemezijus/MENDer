import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function KmeansSection({ m, set, sub, enums, d }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Clusters (n_clusters)"
          value={m.n_clusters}

          placeholder={defaultPlaceholder(d?.n_clusters)}
          onChange={(v) => set({ n_clusters: overrideOrUndef(v, d?.n_clusters) })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Init"
          data={toSelectData(enumFromSubSchema(sub, 'init', ['k-means++', 'random']))}
          value={m.init}

          placeholder={defaultPlaceholder(d?.init)}
          onChange={(v) => set({ init: overrideOrUndef(v, d?.init) })}
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
          value={m.max_iter}

          placeholder={defaultPlaceholder(d?.max_iter)}
          onChange={(v) => set({ max_iter: overrideOrUndef(v, d?.max_iter) })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol}

          placeholder={defaultPlaceholder(d?.tol)}
          onChange={(v) => set({ tol: overrideOrUndef(v, d?.tol) })}
          step={1e-5}
          min={0}
        />
        <ParamNumber
          label="Verbose"
          value={m.verbose}

          placeholder={defaultPlaceholder(d?.verbose)}
          onChange={(v) => set({ verbose: overrideOrUndef(v, d?.verbose) })}
          allowDecimal={false}
          min={0}
        />
        <ParamSelect
          label="Algorithm"
          data={toSelectData(enumFromSubSchema(sub, 'algorithm', ['lloyd', 'elkan', 'auto']))}
          value={m.algorithm}

          placeholder={defaultPlaceholder(d?.algorithm)}
          onChange={(v) => set({ algorithm: overrideOrUndef(v, d?.algorithm) })}
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
