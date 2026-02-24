import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function LassoSection({ m, set, sub, enums }) {
  const cdSelection = makeSelectData(sub, 'selection', enums?.CoordinateDescentSelection);
  return (
    <ParamGrid>
        <ParamNumber
          label="Alpha"
          value={m.alpha ?? 1.0}
          onChange={(v) => set({ alpha: v })}
          min={0}
          step={0.1}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(checked) => set({ fit_intercept: checked })}
        />
        <ParamSelect
          label="Selection"
          data={cdSelection}
          value={m.selection ?? 'cyclic'}
          onChange={(v) => set({ selection: v })}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter ?? 1000}
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
        <ParamNumber
          label="Random state"
          value={m.random_state ?? null}
          onChange={(v) => set({ random_state: v })}
          allowDecimal={false}
        />
        <ParamCheckbox
          label="Positive coefficients"
          checked={!!m.positive}
          onChange={(checked) => set({ positive: checked })}
        />
      </ParamGrid>
  );
}
