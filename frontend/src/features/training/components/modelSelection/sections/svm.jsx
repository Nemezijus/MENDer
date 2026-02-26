import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import {
  defaultPlaceholder,
  effectiveValue,
  overrideFromNullableSelect,
  overrideOrUndef,
  toNullableSelectValue,
} from '../utils/paramDefaults.js';

export default function SvmSection({ m, set, sub, enums, d }) {
  const svmKernel = makeSelectData(sub, 'kernel', enums?.SVMKernel);
  const svmDecisionShape = makeSelectData(sub, 'decision_function_shape', enums?.SVMDecisionShape);
  const svmClassWeight = makeSelectData(sub, 'class_weight', enums?.ClassWeightBalanced, { includeNoneLabel: true });
  const svmClassWeightUnavailable = svmClassWeight.length === 0;
  const defGamma = d?.gamma;
  const effGamma = effectiveValue(m.gamma, defGamma);
  const effGammaMode = typeof effGamma === 'number' ? 'numeric' : (effGamma ?? 'scale');
  const defGammaMode = typeof defGamma === 'number' ? 'numeric' : defGamma;
  const gammaModeValue = typeof m.gamma === 'number' ? 'numeric' : (typeof m.gamma === 'string' ? m.gamma : undefined);
  const gammaNumValue = typeof m.gamma === 'number' ? m.gamma : undefined;
  const gammaNumPlaceholder = typeof effGamma === 'number' ? defaultPlaceholder(effGamma) : undefined;
  return (
    <ParamGrid>
        <ParamSelect
          label="Kernel"
          data={svmKernel}
          value={m.kernel}

          placeholder={defaultPlaceholder(d?.kernel)}
          onChange={(v) => set({ kernel: overrideOrUndef(v, d?.kernel) })}
        />
        <ParamNumber
          label="C (penalty)"
          value={m.C}

          placeholder={defaultPlaceholder(d?.C)}
          onChange={(v) => set({ C: overrideOrUndef(v, d?.C) })}
          min={0}
          step={0.1}
        />
        <ParamNumber
          label="Degree (poly)"
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
            step={0.001}
          />
        )}
        <ParamNumber
          label="Coef0"
          value={m.coef0}

          placeholder={defaultPlaceholder(d?.coef0)}
          onChange={(v) => set({ coef0: overrideOrUndef(v, d?.coef0) })}
          step={0.001}
        />
        <ParamCheckbox
          label="Use shrinking"
          checked={effectiveValue(m.shrinking, d?.shrinking)}
          onChange={(checked) => set({ shrinking: overrideOrUndef(checked, d?.shrinking) })}/>
        <ParamCheckbox
          label="Enable probability"
          checked={effectiveValue(m.probability, d?.probability)}
          onChange={(checked) => set({ probability: overrideOrUndef(checked, d?.probability) })}/>
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol}

          placeholder={defaultPlaceholder(d?.tol)}
          onChange={(v) => set({ tol: overrideOrUndef(v, d?.tol) })}
          step={0.0001}
        />
        <ParamNumber
          label="Cache size (MB)"
          value={m.cache_size}

          placeholder={defaultPlaceholder(d?.cache_size)}
          onChange={(v) => set({ cache_size: overrideOrUndef(v, d?.cache_size) })}
        />
        <ParamSelect
          label="Class weight"
          data={svmClassWeight}
          value={toNullableSelectValue(m.class_weight)}
          disabled={svmClassWeightUnavailable}
          placeholder={svmClassWeightUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.class_weight)}
          description={svmClassWeightUnavailable ? 'Schema did not provide class_weight options.' : undefined}
          onChange={(v) => set({ class_weight: overrideFromNullableSelect(v, d?.class_weight) })}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter}

          placeholder={defaultPlaceholder(d?.max_iter)}
          onChange={(v) => set({ max_iter: overrideOrUndef(v, d?.max_iter) })}
          allowDecimal={false}
        />
        <ParamSelect
          label="Decision shape"
          data={svmDecisionShape}
          value={m.decision_function_shape}

          placeholder={defaultPlaceholder(d?.decision_function_shape)}
          onChange={(v) => set({ decision_function_shape: overrideOrUndef(v, d?.decision_function_shape) })}/>
        <ParamCheckbox
          label="Break ties"
          checked={effectiveValue(m.break_ties, d?.break_ties)}
          onChange={(checked) => set({ break_ties: overrideOrUndef(checked, d?.break_ties) })}/>
      </ParamGrid>
  );
}
