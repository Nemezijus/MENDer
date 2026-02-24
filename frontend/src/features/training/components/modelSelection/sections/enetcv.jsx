import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { parseCsvFloats, formatCsvFloats, makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function EnetcvSection({ m, set, sub, enums }) {
  const cdSelection = makeSelectData(sub, 'selection', enums?.CoordinateDescentSelection);
  return (
    <ParamGrid>
        <TextInput
          label="L1 ratio list (comma-separated)"
          placeholder="e.g. 0.1, 0.5, 0.9"
          value={formatCsvFloats(m.l1_ratio)}
          onChange={(e) => {
            const vals = parseCsvFloats(e.currentTarget.value);
            set({ l1_ratio: vals || [0.1, 0.5, 0.9] });
          }}
        />
        <ParamNumber
          label="eps"
          value={m.eps ?? 1e-3}
          onChange={(v) => set({ eps: v })}
          min={0}
          step={1e-4}
        />
        <ParamNumber
          label="n_alphas"
          value={m.n_alphas ?? 100}
          onChange={(v) => set({ n_alphas: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="CV folds"
          value={m.cv ?? 5}
          onChange={(v) => set({ cv: v })}
          allowDecimal={false}
          min={2}
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
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
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
