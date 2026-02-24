import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function RidgeSection({ m, set, sub, enums }) {
  const ridgeSolver = makeSelectData(sub, 'solver', enums?.RidgeSolver);
  const ridgeClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  return (
    <ParamGrid>
        <ParamNumber
          label="Alpha (regularization)"
          value={m.alpha ?? 1.0}
          onChange={(v) => set({ alpha: v })}
          min={0}
          step={0.1}
        />
        <ParamSelect
          label="Solver"
          data={ridgeSolver}
          value={m.solver ?? 'auto'}
          onChange={(v) => set({ solver: v })}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(checked) => set({ fit_intercept: checked })}
        />
        <ParamSelect
          label="Class weight"
          data={ridgeClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter ?? null}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol ?? 1e-4}
          onChange={(v) => set({ tol: v })}
          step={1e-5}
          min={0}
        />
      </ParamGrid>
  );
}
