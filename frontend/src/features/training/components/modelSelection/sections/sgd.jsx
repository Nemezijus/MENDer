import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function SgdSection({ m, set, sub, enums, d }) {
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
          value={m.loss}

          placeholder={defaultPlaceholder(d?.loss)}
          onChange={(v) => set({ loss: overrideOrUndef(v, d?.loss) })}
        />
        <ParamSelect
          label="Penalty"
          data={sgdPenalty}
          value={m.penalty}

          placeholder={defaultPlaceholder(d?.penalty)}
          onChange={(v) => set({ penalty: overrideOrUndef(v, d?.penalty) })}
        />
        <ParamNumber
          label="Alpha"
          value={m.alpha}

          placeholder={defaultPlaceholder(d?.alpha)}
          onChange={(v) => set({ alpha: overrideOrUndef(v, d?.alpha) })}
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
          checked={effectiveValue(m.fit_intercept, d?.fit_intercept)}
          onChange={(checked) => set({ fit_intercept: overrideOrUndef(checked, d?.fit_intercept) })}
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
          step={0.0001}
          min={0}
        />
        <ParamCheckbox
          label="Shuffle"
          checked={effectiveValue(m.shuffle, d?.shuffle)}
          onChange={(checked) => set({ shuffle: overrideOrUndef(checked, d?.shuffle) })}
        />
        <ParamSelect
          label="Learning rate"
          data={sgdLR}
          value={m.learning_rate}

          placeholder={defaultPlaceholder(d?.learning_rate)}
          onChange={(v) => set({ learning_rate: overrideOrUndef(v, d?.learning_rate) })}
        />
        <ParamNumber
          label="Eta0"
          value={m.eta0}

          placeholder={defaultPlaceholder(d?.eta0)}
          onChange={(v) => set({ eta0: overrideOrUndef(v, d?.eta0) })}
          min={0}
          step={0.01}
        />
        <ParamNumber
          label="Power t"
          value={m.power_t}

          placeholder={defaultPlaceholder(d?.power_t)}
          onChange={(v) => set({ power_t: overrideOrUndef(v, d?.power_t) })}
          step={0.01}
        />
        <ParamCheckbox
          label="Early stopping"
          checked={effectiveValue(m.early_stopping, d?.early_stopping)}
          onChange={(checked) => set({ early_stopping: overrideOrUndef(checked, d?.early_stopping) })}
        />
        <ParamNumber
          label="Validation fraction"
          value={m.validation_fraction}

          placeholder={defaultPlaceholder(d?.validation_fraction)}
          onChange={(v) => set({ validation_fraction: overrideOrUndef(v, d?.validation_fraction) })}
          min={0}
          max={1}
          step={0.01}
        />
        <ParamNumber
          label="No-change rounds"
          value={m.n_iter_no_change}

          placeholder={defaultPlaceholder(d?.n_iter_no_change)}
          onChange={(v) => set({ n_iter_no_change: overrideOrUndef(v, d?.n_iter_no_change) })}
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
            onChange={(v) => set({ average: overrideOrUndef(v, d?.average) })}
            allowDecimal={false}
            min={1}
          />
        )}
        <ParamNumber
          label="Jobs (n_jobs)"
          value={m.n_jobs}

          placeholder={defaultPlaceholder(d?.n_jobs)}
          onChange={(v) => set({ n_jobs: overrideOrUndef(v, d?.n_jobs) })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
