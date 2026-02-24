import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function SvrSection({ m, set, sub, enums }) {
  const svmKernel = makeSelectData(sub, 'kernel', enums?.SVMKernel);
  const gammaMode = typeof m.gamma === 'number' ? 'numeric' : (m.gamma ?? 'scale');
  const gammaValue = typeof m.gamma === 'number' ? m.gamma : 0.1;
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
          label="Kernel"
          data={svmKernel}
          value={m.kernel ?? 'rbf'}
          onChange={(v) => set({ kernel: v })}
        />
        <ParamNumber
          label="Degree"
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
            if (mode === 'numeric') set({ gamma: gammaValue });
            else set({ gamma: mode });
          }}
        />
        {gammaMode === 'numeric' && (
          <ParamNumber
            label="Gamma value"
            value={gammaValue}
            onChange={(v) => set({ gamma: v })}
            min={0}
            step={0.01}
          />
        )}
        <ParamNumber
          label="Coef0"
          value={m.coef0 ?? 0.0}
          onChange={(v) => set({ coef0: v })}
          step={0.1}
        />
        <ParamCheckbox
          label="Shrinking"
          checked={m.shrinking ?? true}
          onChange={(checked) => set({ shrinking: checked })}
        />
        <ParamNumber
          label="Epsilon"
          value={m.epsilon ?? 0.1}
          onChange={(v) => set({ epsilon: v })}
          min={0}
          step={0.01}
        />
        <ParamNumber
          label="Tolerance (tol)"
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={1e-4}
          min={0}
        />
        <ParamNumber
          label="Cache size (MB)"
          value={m.cache_size ?? 200.0}
          onChange={(v) => set({ cache_size: v })}
          min={0}
          step={10}
        />
        <ParamNumber
          label="Max iterations"
          value={m.max_iter ?? -1}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
