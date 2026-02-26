import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function EnetSection({ m, set, sub, enums, d }) {
  const cdSelection = makeSelectData(sub, 'selection', enums?.CoordinateDescentSelection);
  return (
    <ParamGrid>
        <ParamNumber
          label="Alpha"
          value={m.alpha}

          placeholder={defaultPlaceholder(d?.alpha)}
          onChange={(v) => set({ alpha: overrideOrUndef(v, d?.alpha) })}
          min={0}
          step={0.1}
        />
        <ParamNumber
          label="L1 ratio"
          value={m.l1_ratio ?? 0.5}
          onChange={(v) => set({ l1_ratio: v })}
          min={0}
          max={1}
          step={0.05}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={effectiveValue(m.fit_intercept, d?.fit_intercept)}
          onChange={(checked) => set({ fit_intercept: overrideOrUndef(checked, d?.fit_intercept) })}
        />
        <ParamSelect
          label="Selection"
          data={cdSelection}
          value={m.selection}

          placeholder={defaultPlaceholder(d?.selection)}
          onChange={(v) => set({ selection: overrideOrUndef(v, d?.selection) })}
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
          label="Random state"
          value={m.random_state}

          placeholder={defaultPlaceholder(d?.random_state)}
          onChange={(v) => set({ random_state: overrideOrUndef(v, d?.random_state) })}
          allowDecimal={false}
        />
        <ParamCheckbox
          label="Positive coefficients"
          checked={effectiveValue(m.positive, d?.positive)}
          onChange={(checked) => set({ positive: overrideOrUndef(checked, d?.positive) })}
        />
      </ParamGrid>
  );
}
