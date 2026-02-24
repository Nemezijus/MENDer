import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function LinsvrSection({ m, set, sub, enums }) {
  const linsvrLoss = makeSelectData(sub, 'loss', enums?.LinearSVRLoss);
  return (
    <ParamGrid>
        <ParamNumber
          label="C"
          value={m.C ?? 1.0}
          onChange={(v) => set({ C: v })}
          min={0}
          step={0.1}
        />
        <ParamSelect
          label="Loss"
          data={linsvrLoss}
          value={m.loss ?? 'epsilon_insensitive'}
          onChange={(v) => set({ loss: v })}
        />
        <ParamNumber
          label="Epsilon"
          value={m.epsilon ?? 0.0}
          onChange={(v) => set({ epsilon: v })}
          min={0}
          step={0.01}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(checked) => set({ fit_intercept: checked })}
        />
        <ParamNumber
          label="Intercept scaling"
          value={m.intercept_scaling ?? 1.0}
          onChange={(v) => set({ intercept_scaling: v })}
          min={0}
          step={0.1}
        />
        <ParamCheckbox
          label="Dual"
          checked={m.dual ?? true}
          onChange={(checked) => set({ dual: checked })}
        />
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol ?? 1e-4}
          onChange={(v) => set({ tol: v })}
          step={1e-5}
          min={0}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter ?? 1000}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Random state"
          value={m.random_state ?? null}
          onChange={(v) => set({ random_state: v })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
