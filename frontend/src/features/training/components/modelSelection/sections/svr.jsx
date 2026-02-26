import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function SvrSection({ m, set, sub, enums, d }) {
  const svmKernel = makeSelectData(sub, 'kernel', enums?.SVMKernel);
  const defGamma = d?.gamma;
  const effGamma = effectiveValue(m.gamma, defGamma);
  const effGammaMode = typeof effGamma === 'number' ? 'numeric' : (effGamma ?? 'scale');
  const defGammaMode = typeof defGamma === 'number' ? 'numeric' : defGamma;
  const gammaModeValue = typeof m.gamma === 'number' ? 'numeric' : (typeof m.gamma === 'string' ? m.gamma : undefined);
  const gammaNumValue = typeof m.gamma === 'number' ? m.gamma : undefined;
  const gammaNumPlaceholder = typeof effGamma === 'number' ? defaultPlaceholder(effGamma) : undefined;
  return (
    <ParamGrid>
        <ParamNumber
          label="C"
          value={m.C}

          placeholder={defaultPlaceholder(d?.C)}
          onChange={(v) => set({ C: overrideOrUndef(v, d?.C) })}
          min={0}
          step={0.1}
        />
        <ParamSelect
          label="Kernel"
          data={svmKernel}
          value={m.kernel}

          placeholder={defaultPlaceholder(d?.kernel)}
          onChange={(v) => set({ kernel: overrideOrUndef(v, d?.kernel) })}
        />
        <ParamNumber
          label="Degree"
          value={m.degree}

          placeholder={defaultPlaceholder(d?.degree)}
          onChange={(v) => set({ degree: overrideOrUndef(v, d?.degree) })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Gamma mode"
          data={[
            { value: 'scale', label: 'scale' },
            { value: 'auto', label: 'auto' },
            { value: 'numeric', label: 'numeric' },
          ]}
          value={gammaModeValue}
          placeholder={defaultPlaceholder(defGammaMode)}
          onChange={(mode) => {
            if (mode === undefined) {
              set({ gamma: undefined });
              return;
            }

            if (mode === 'numeric') {
              const init = gammaNumValue ?? (typeof effGamma === 'number' ? effGamma : 0.1);
              set({ gamma: overrideOrUndef(init, defGamma) });
              return;
            }

            set({ gamma: overrideOrUndef(mode, defGamma) });
          }}
        />
        {effGammaMode === 'numeric' && (
          <ParamNumber
            label="Gamma value"
            value={gammaNumValue}
            placeholder={gammaNumPlaceholder}
            onChange={(v) => {
              const defForCompare = typeof defGamma === 'number' ? defGamma : undefined;
              set({ gamma: overrideOrUndef(v, defForCompare) });
            }}
            min={0}
            step={0.01}
          />
        )}
        <ParamNumber
          label="Coef0"
          value={m.coef0}

          placeholder={defaultPlaceholder(d?.coef0)}
          onChange={(v) => set({ coef0: overrideOrUndef(v, d?.coef0) })}
          step={0.1}
        />
        <ParamCheckbox
          label="Shrinking"
          checked={effectiveValue(m.shrinking, d?.shrinking)}
          onChange={(checked) => set({ shrinking: overrideOrUndef(checked, d?.shrinking) })}
        />
        <ParamNumber
          label="Epsilon"
          value={m.epsilon}

          placeholder={defaultPlaceholder(d?.epsilon)}
          onChange={(v) => set({ epsilon: overrideOrUndef(v, d?.epsilon) })}
          min={0}
          step={0.01}
        />
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol}

          placeholder={defaultPlaceholder(d?.tol)}
          onChange={(v) => set({ tol: overrideOrUndef(v, d?.tol) })}
          step={1e-4}
          min={0}
        />
        <ParamNumber
          label="Cache size (MB)"
          value={m.cache_size}

          placeholder={defaultPlaceholder(d?.cache_size)}
          onChange={(v) => set({ cache_size: overrideOrUndef(v, d?.cache_size) })}
          min={0}
          step={10}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter}

          placeholder={defaultPlaceholder(d?.max_iter)}
          onChange={(v) => set({ max_iter: overrideOrUndef(v, d?.max_iter) })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
