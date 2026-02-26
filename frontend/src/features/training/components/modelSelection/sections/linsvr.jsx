import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function LinsvrSection({ m, set, sub, enums, d }) {
  const linsvrLoss = makeSelectData(sub, 'loss', enums?.LinearSVRLoss);
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
          label="Loss"
          data={linsvrLoss}
          value={m.loss}

          placeholder={defaultPlaceholder(d?.loss)}
          onChange={(v) => set({ loss: overrideOrUndef(v, d?.loss) })}
        />
        <ParamNumber
          label="Epsilon"
          value={m.epsilon}

          placeholder={defaultPlaceholder(d?.epsilon)}
          onChange={(v) => set({ epsilon: overrideOrUndef(v, d?.epsilon) })}
          min={0}
          step={0.01}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={effectiveValue(m.fit_intercept, d?.fit_intercept)}
          onChange={(checked) => set({ fit_intercept: overrideOrUndef(checked, d?.fit_intercept) })}
        />
        <ParamNumber
          label="Intercept scaling"
          value={m.intercept_scaling}

          placeholder={defaultPlaceholder(d?.intercept_scaling)}
          onChange={(v) => set({ intercept_scaling: overrideOrUndef(v, d?.intercept_scaling) })}
          min={0}
          step={0.1}
        />
        <ParamCheckbox
          label="Dual"
          checked={effectiveValue(m.dual, d?.dual)}
          onChange={(checked) => set({ dual: overrideOrUndef(checked, d?.dual) })}
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
          label="Max iterations"
          value={m.max_iter}

          placeholder={defaultPlaceholder(d?.max_iter)}
          onChange={(v) => set({ max_iter: overrideOrUndef(v, d?.max_iter) })}
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
