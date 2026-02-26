import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function LassocvSection({ m, set, sub, enums, d }) {
  const cdSelection = makeSelectData(sub, 'selection', enums?.CoordinateDescentSelection);
  return (
    <ParamGrid>
        <ParamNumber
          label="eps"
          value={m.eps}

          placeholder={defaultPlaceholder(d?.eps)}
          onChange={(v) => set({ eps: overrideOrUndef(v, d?.eps) })}
          min={0}
          step={1e-4}
        />
        <ParamNumber
          label="n_alphas"
          value={m.n_alphas}

          placeholder={defaultPlaceholder(d?.n_alphas)}
          onChange={(v) => set({ n_alphas: overrideOrUndef(v, d?.n_alphas) })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="CV folds"
          value={m.cv}

          placeholder={defaultPlaceholder(d?.cv)}
          onChange={(v) => set({ cv: overrideOrUndef(v, d?.cv) })}
          allowDecimal={false}
          min={2}
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
          label="Jobs (n_jobs)"
          value={m.n_jobs}

          placeholder={defaultPlaceholder(d?.n_jobs)}
          onChange={(v) => set({ n_jobs: overrideOrUndef(v, d?.n_jobs) })}
          allowDecimal={false}
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
