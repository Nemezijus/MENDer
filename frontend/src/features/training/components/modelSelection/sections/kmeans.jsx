import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { enumFromSubSchema, toSelectData } from '../../../../../shared/utils/schema/jsonSchema.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function KmeansSection({ m, set, sub, enums, d }) {
  const defNInit = d?.n_init;
  const effNInit = effectiveValue(m.n_init, defNInit);
  const effNInitMode = typeof effNInit === 'number' ? 'int' : 'auto';

  const nInitModeValue =
    m.n_init === undefined ? undefined : typeof m.n_init === 'number' ? 'int' : 'auto';

  const nInitNumValue = typeof m.n_init === 'number' ? m.n_init : undefined;
  const nInitNumPlaceholder = typeof effNInit === 'number' ? defaultPlaceholder(effNInit) : undefined;

  const defNInitMode = typeof defNInit === 'number' ? 'int' : defNInit;
  const initData = toSelectData(enumFromSubSchema(sub, 'init'));
  const initUnavailable = initData.length === 0;
  const algorithmData = toSelectData(enumFromSubSchema(sub, 'algorithm'));
  const algorithmUnavailable = algorithmData.length === 0;


  return (
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
        data={initData}
        value={m.init}
        disabled={initUnavailable}
        placeholder={initUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.init)}
        description={initUnavailable ? 'Schema did not provide init options.' : undefined}
        onChange={(v) => set({ init: overrideOrUndef(v, d?.init) })}
      />

      <ParamSelect
        label="n_init mode"
        data={[
          { value: 'auto', label: 'auto' },
          { value: 'int', label: 'int' },
        ]}
        value={nInitModeValue}
        placeholder={defaultPlaceholder(defNInitMode)}
        onChange={(mode) => {
          if (mode === undefined) {
            set({ n_init: undefined });
            return;
          }

          if (mode === 'auto') {
            set({ n_init: overrideOrUndef('auto', defNInit) });
            return;
          }

          // int
          const init = typeof effNInit === 'number' ? Math.trunc(effNInit) : 10;
          const next = Math.max(1, init);
          const defForCompare = typeof defNInit === 'number' ? Math.trunc(defNInit) : undefined;
          set({ n_init: overrideOrUndef(next, defForCompare) });
        }}
      />

      {effNInitMode === 'int' && (
        <ParamNumber
          label="n_init value"
          value={nInitNumValue}
          placeholder={nInitNumPlaceholder}
          onChange={(v) => {
            if (v === undefined) {
              set({ n_init: undefined });
              return;
            }

            const next = Math.max(1, Math.trunc(v));
            const defForCompare = typeof defNInit === 'number' ? Math.trunc(defNInit) : undefined;
            set({ n_init: overrideOrUndef(next, defForCompare) });
          }}
          allowDecimal={false}
          min={1}
        />
      )}

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
        data={algorithmData}
        value={m.algorithm}
        disabled={algorithmUnavailable}
        placeholder={algorithmUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.algorithm)}
        description={algorithmUnavailable ? 'Schema did not provide algorithm options.' : undefined}
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
