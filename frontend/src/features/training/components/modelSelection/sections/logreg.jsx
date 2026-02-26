import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function LogregSection({ m, set, sub, enums, d }) {
  const lrPenalty = makeSelectData(sub, 'penalty', enums?.PenaltyName);
  const lrSolver = makeSelectData(sub, 'solver', enums?.LogRegSolver);
  const lrClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  return (
    <ParamGrid>
        <ParamNumber
          label="C (strength)"
          value={m.C}

          placeholder={defaultPlaceholder(d?.C)}
          onChange={(v) => set({ C: overrideOrUndef(v, d?.C) })}
          min={0}
          step={0.1}
        />
        <ParamSelect
          label="Penalty"
          data={lrPenalty}
          value={m.penalty}

          placeholder={defaultPlaceholder(d?.penalty)}
          onChange={(v) => set({ penalty: overrideOrUndef(v, d?.penalty) })}
        />
        <ParamSelect
          label="Solver"
          data={lrSolver}
          value={m.solver}

          placeholder={defaultPlaceholder(d?.solver)}
          onChange={(v) => set({ solver: overrideOrUndef(v, d?.solver) })}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter}

          placeholder={defaultPlaceholder(d?.max_iter)}
          onChange={(v) => set({ max_iter: overrideOrUndef(v, d?.max_iter) })}
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
