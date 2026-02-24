import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function LogregSection({ m, set, sub, enums }) {
  const lrPenalty = makeSelectData(sub, 'penalty', enums?.PenaltyName);
  const lrSolver = makeSelectData(sub, 'solver', enums?.LogRegSolver);
  const lrClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  return (
    <ParamGrid>
        <ParamNumber
          label="C (strength)"
          value={m.C ?? 1.0}
          onChange={(v) => set({ C: v })}
          min={0}
          step={0.1}
        />
        <ParamSelect
          label="Penalty"
          data={lrPenalty}
          value={m.penalty ?? 'l2'}
          onChange={(v) => set({ penalty: v })}
        />
        <ParamSelect
          label="Solver"
          data={lrSolver}
          value={m.solver ?? 'lbfgs'}
          onChange={(v) => set({ solver: v })}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter ?? 1000}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Class weight"
          data={lrClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        {m.penalty === 'elasticnet' && (
          <ParamNumber
            label="L1 ratio"
            value={m.l1_ratio ?? 0.5}
            onChange={(v) => set({ l1_ratio: v })}
            min={0}
            max={1}
            step={0.01}
          />
        )}
      </ParamGrid>
  );
}
