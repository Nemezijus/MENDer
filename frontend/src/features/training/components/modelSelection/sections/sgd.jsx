import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function SgdSection({ m, set, sub, enums }) {
  const sgdLoss = makeSelectData(sub, 'loss', enums?.SGDLoss);
  const sgdPenalty = makeSelectData(sub, 'penalty', enums?.SGDPenalty);
  const sgdLR = makeSelectData(sub, 'learning_rate', enums?.SGDLearningRate);
  const sgdClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  const sgdAvgMode = typeof m.average === 'number' ? 'int' : (m.average ? 'true' : 'false');
  const sgdAvgValue = typeof m.average === 'number' ? m.average : 10;
  return (
    <ParamGrid>
        <ParamSelect
          label="Loss"
          data={sgdLoss}
          value={m.loss ?? 'hinge'}
          onChange={(v) => set({ loss: v })}
        />
        <ParamSelect
          label="Penalty"
          data={sgdPenalty}
          value={m.penalty ?? 'l2'}
          onChange={(v) => set({ penalty: v })}
        />
        <ParamNumber
          label="Alpha"
          value={m.alpha ?? 0.0001}
          onChange={(v) => set({ alpha: v })}
          min={0}
          step={0.0001}
        />
        <ParamNumber
          label="L1 ratio"
          value={m.l1_ratio ?? 0.15}
          onChange={(v) => set({ l1_ratio: v })}
          min={0}
          max={1}
          step={0.01}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={m.fit_intercept ?? true}
          onChange={(checked) => set({ fit_intercept: checked })}
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
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={0.0001}
          min={0}
        />
        <ParamCheckbox
          label="Shuffle"
          checked={m.shuffle ?? true}
          onChange={(checked) => set({ shuffle: checked })}
        />
        <ParamSelect
          label="Learning rate"
          data={sgdLR}
          value={m.learning_rate ?? 'optimal'}
          onChange={(v) => set({ learning_rate: v })}
        />
        <ParamNumber
          label="Eta0"
          value={m.eta0 ?? 0.0}
          onChange={(v) => set({ eta0: v })}
          min={0}
          step={0.01}
        />
        <ParamNumber
          label="Power t"
          value={m.power_t ?? 0.5}
          onChange={(v) => set({ power_t: v })}
          step={0.01}
        />
        <ParamCheckbox
          label="Early stopping"
          checked={!!m.early_stopping}
          onChange={(checked) => set({ early_stopping: checked })}
        />
        <ParamNumber
          label="Validation fraction"
          value={m.validation_fraction ?? 0.1}
          onChange={(v) => set({ validation_fraction: v })}
          min={0}
          max={1}
          step={0.01}
        />
        <ParamNumber
          label="No-change rounds"
          value={m.n_iter_no_change ?? 5}
          onChange={(v) => set({ n_iter_no_change: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Class weight"
          data={sgdClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        <ParamSelect
          label="Average"
          data={[
            { value: 'false', label: 'false' },
            { value: 'true', label: 'true' },
            { value: 'int', label: 'int' },
          ]}
          value={sgdAvgMode}
          onChange={(mode) => {
            if (mode === 'int') set({ average: sgdAvgValue });
            else if (mode === 'true') set({ average: true });
            else set({ average: false });
          }}
        />
        {sgdAvgMode === 'int' && (
          <ParamNumber
            label="Average window"
            value={sgdAvgValue}
            onChange={(v) => set({ average: v })}
            allowDecimal={false}
            min={1}
          />
        )}
        <ParamNumber
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
