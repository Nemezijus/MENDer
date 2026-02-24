import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function SvmSection({ m, set, sub, enums }) {
  const svmKernel = makeSelectData(sub, 'kernel', enums?.SVMKernel);
  const svmDecisionShape = makeSelectData(sub, 'decision_function_shape', enums?.SVMDecisionShape);
  const svmClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  const gammaMode = typeof m.gamma === 'number' ? 'numeric' : (m.gamma ?? 'scale');
  const gammaValue = typeof m.gamma === 'number' ? m.gamma : 0.1;
  return (
    <ParamGrid>
        <ParamSelect
          label="Kernel"
          data={svmKernel}
          value={m.kernel ?? 'rbf'}
          onChange={(v) => set({ kernel: v })}
        />
        <ParamNumber
          label="C (penalty)"
          value={m.C ?? 1.0}
          onChange={(v) => set({ C: v })}
          min={0}
          step={0.1}
        />
        <ParamNumber
          label="Degree (poly)"
          value={m.degree ?? 3}
          onChange={(v) => set({ degree: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Gamma mode"
          data={[
            { value: 'scale', label: 'scale' },
            { value: 'auto', label: 'auto' },
            { value: 'numeric', label: 'numeric' },
          ]}
          value={gammaMode}
          onChange={(mode) => {
            if (mode === 'numeric') {
              set({
                gamma:
                  typeof gammaValue === 'number' ? gammaValue : 0.1,
              });
            } else {
              set({ gamma: mode });
            }
          }}
        />
        {gammaMode === 'numeric' && (
          <ParamNumber
            label="Gamma value"
            value={gammaValue}
            onChange={(v) => set({ gamma: v })}
            min={0}
            step={0.001}
          />
        )}
        <ParamNumber
          label="Coef0"
          value={m.coef0 ?? 0.0}
          onChange={(v) => set({ coef0: v })}
          step={0.001}
        />
        <ParamCheckbox
          label="Use shrinking"
          checked={!!m.shrinking}
          onChange={(checked) =>
            set({ shrinking: checked })
          }
        />
        <ParamCheckbox
          label="Enable probability"
          checked={!!m.probability}
          onChange={(checked) =>
            set({ probability: checked })
          }
        />
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={0.0001}
        />
        <ParamNumber
          label="Cache size (MB)"
          value={m.cache_size ?? 200.0}
          onChange={(v) => set({ cache_size: v })}
        />
        <ParamSelect
          label="Class weight"
          data={svmClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter ?? -1}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
        />
        <ParamSelect
          label="Decision shape"
          data={svmDecisionShape}
          value={m.decision_function_shape ?? 'ovr'}
          onChange={(v) =>
            set({ decision_function_shape: v })
          }
        />
        <ParamCheckbox
          label="Break ties"
          checked={!!m.break_ties}
          onChange={(checked) =>
            set({ break_ties: checked })
          }
        />
      </ParamGrid>
  );
}
