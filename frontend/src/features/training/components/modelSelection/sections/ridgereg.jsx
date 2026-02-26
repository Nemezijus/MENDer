import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function RidgeregSection({ m, set, sub, enums, d }) {
  const ridgeSolver = makeSelectData(sub, 'solver', enums?.RidgeSolver);
  return (
    <ParamGrid>
        <ParamNumber
          label="Alpha (regularization)"
          value={m.alpha}

          placeholder={defaultPlaceholder(d?.alpha)}
          onChange={(v) => set({ alpha: overrideOrUndef(v, d?.alpha) })}
          min={0}
          step={0.1}
        />
        <ParamSelect
          label="Solver"
          data={ridgeSolver}
          value={m.solver}

          placeholder={defaultPlaceholder(d?.solver)}
          onChange={(v) => set({ solver: overrideOrUndef(v, d?.solver) })}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={effectiveValue(m.fit_intercept, d?.fit_intercept)}
          onChange={(checked) => set({ fit_intercept: overrideOrUndef(checked, d?.fit_intercept) })}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter}

          placeholder={defaultPlaceholder(d?.max_iter)}
          onChange={(v) => set({ max_iter: overrideOrUndef(v, d?.max_iter) })}
          allowDecimal={false}
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
