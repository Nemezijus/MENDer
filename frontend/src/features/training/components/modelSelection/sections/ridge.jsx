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

export default function RidgeSection({ m, set, sub, enums, d }) {
  const ridgeSolver = makeSelectData(sub, 'solver', enums?.RidgeSolver);
  const ridgeClassWeight = makeSelectData(sub, 'class_weight', enums?.ClassWeightBalanced, { includeNoneLabel: true });
  const ridgeClassWeightUnavailable = ridgeClassWeight.length === 0;
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
        <ParamSelect
          label="Class weight"
          data={ridgeClassWeight}
          value={toNullableSelectValue(m.class_weight)}
          disabled={ridgeClassWeightUnavailable}
          placeholder={ridgeClassWeightUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.class_weight)}
          description={ridgeClassWeightUnavailable ? 'Schema did not provide class_weight options.' : undefined}
          onChange={(v) => set({ class_weight: overrideFromNullableSelect(v, d?.class_weight) })}
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
      </ParamGrid>
  );
}
