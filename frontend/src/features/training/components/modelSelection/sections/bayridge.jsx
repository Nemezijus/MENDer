import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function BayridgeSection({ m, set, sub, enums, d }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Iterations (n_iter)"
          value={m.n_iter}

          placeholder={defaultPlaceholder(d?.n_iter)}
          onChange={(v) => set({ n_iter: overrideOrUndef(v, d?.n_iter) })}
          allowDecimal={false}
          min={1}
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
          label="alpha_1"
          value={m.alpha_1}

          placeholder={defaultPlaceholder(d?.alpha_1)}
          onChange={(v) => set({ alpha_1: overrideOrUndef(v, d?.alpha_1) })}
          step={1e-6}
          min={0}
        />
        <ParamNumber
          label="alpha_2"
          value={m.alpha_2}

          placeholder={defaultPlaceholder(d?.alpha_2)}
          onChange={(v) => set({ alpha_2: overrideOrUndef(v, d?.alpha_2) })}
          step={1e-6}
          min={0}
        />
        <ParamNumber
          label="lambda_1"
          value={m.lambda_1}

          placeholder={defaultPlaceholder(d?.lambda_1)}
          onChange={(v) => set({ lambda_1: overrideOrUndef(v, d?.lambda_1) })}
          step={1e-6}
          min={0}
        />
        <ParamNumber
          label="lambda_2"
          value={m.lambda_2}

          placeholder={defaultPlaceholder(d?.lambda_2)}
          onChange={(v) => set({ lambda_2: overrideOrUndef(v, d?.lambda_2) })}
          step={1e-6}
          min={0}
        />
        <ParamCheckbox
          label="Compute score"
          checked={effectiveValue(m.compute_score, d?.compute_score)}
          onChange={(checked) => set({ compute_score: overrideOrUndef(checked, d?.compute_score) })}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={effectiveValue(m.fit_intercept, d?.fit_intercept)}
          onChange={(checked) => set({ fit_intercept: overrideOrUndef(checked, d?.fit_intercept) })}
        />
        <ParamCheckbox
          label="Copy X"
          checked={effectiveValue(m.copy_X, d?.copy_X)}
          onChange={(checked) => set({ copy_X: overrideOrUndef(checked, d?.copy_X) })}
        />
        <ParamCheckbox
          label="Verbose"
          checked={effectiveValue(m.verbose, d?.verbose)}
          onChange={(checked) => set({ verbose: overrideOrUndef(checked, d?.verbose) })}
        />
      </ParamGrid>
  );
}
