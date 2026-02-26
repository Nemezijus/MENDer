import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function LinregSection({ m, set, sub, enums, d }) {  return (
    <ParamGrid>
        <ParamCheckbox
          label="Fit intercept"
          checked={effectiveValue(m.fit_intercept, d?.fit_intercept)}
          onChange={(checked) => set({ fit_intercept: overrideOrUndef(checked, d?.fit_intercept) })}/>
        <ParamCheckbox
          label="Copy X"
          checked={effectiveValue(m.copy_X, d?.copy_X)}
          onChange={(checked) => set({ copy_X: overrideOrUndef(checked, d?.copy_X) })}/>
        <ParamNumber
          label="Jobs (n_jobs)"
          value={m.n_jobs}

          placeholder={defaultPlaceholder(d?.n_jobs)}
          onChange={(v) => set({ n_jobs: overrideOrUndef(v, d?.n_jobs) })}
          allowDecimal={false}
        />
        <ParamCheckbox
          label="Positive coefficients"
          checked={effectiveValue(m.positive, d?.positive)}
          onChange={(checked) => set({ positive: overrideOrUndef(checked, d?.positive) })}/>
      </ParamGrid>
  );
}
