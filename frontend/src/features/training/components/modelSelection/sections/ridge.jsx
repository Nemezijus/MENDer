import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function RidgeSection({ m, set, sub, enums, d }) {
  const ridgeSolver = makeSelectData(sub, 'solver', enums?.RidgeSolver);
  const ridgeClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
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
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
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
