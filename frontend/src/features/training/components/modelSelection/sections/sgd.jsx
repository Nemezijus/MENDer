import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import {
  defaultPlaceholder,
  effectiveValue,
  overrideFromNullableSelect,
  overrideOrUndef,
  toNullableSelectValue,
} from '../utils/paramDefaults.js';

export default function SgdSection({ m, set, sub, enums, d }) {
  const sgdLoss = makeSelectData(sub, 'loss', enums?.SGDLoss);
  const sgdPenalty = makeSelectData(sub, 'penalty', enums?.SGDPenalty);
  const sgdLR = makeSelectData(sub, 'learning_rate', enums?.SGDLearningRate);
  const sgdClassWeight = makeSelectData(sub, 'class_weight', enums?.ClassWeightBalanced, { includeNoneLabel: true });
  const sgdClassWeightUnavailable = sgdClassWeight.length === 0;
  const defAvg = d?.average;
  const effAvg = effectiveValue(m.average, defAvg);
  const effAvgMode = typeof effAvg === 'number' ? 'int' : (effAvg ? 'true' : 'false');
  const hasAvgOverride = m.average !== undefined;
  const avgModeValue = hasAvgOverride
    ? (typeof m.average === 'number' ? 'int' : (m.average ? 'true' : 'false'))
    : (typeof defAvg === 'number' ? 'int' : undefined);
  const avgNumValue = typeof m.average === 'number' ? m.average : undefined;
  const avgNumPlaceholder = typeof effAvg === 'number' ? defaultPlaceholder(effAvg) : undefined;
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
          value={m.l1_ratio}

          placeholder={defaultPlaceholder(d?.l1_ratio)}
          onChange={(v) => set({ l1_ratio: overrideOrUndef(v, d?.l1_ratio) })}
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
          value={toNullableSelectValue(m.class_weight)}
          disabled={sgdClassWeightUnavailable}
          placeholder={sgdClassWeightUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.class_weight)}
          description={sgdClassWeightUnavailable ? 'Schema did not provide class_weight options.' : undefined}
          onChange={(v) => set({ class_weight: overrideFromNullableSelect(v, d?.class_weight) })}
        />
        <ParamSelect
          label="Average"
          data={[
            { value: 'false', label: 'false' },
            { value: 'true', label: 'true' },
            { value: 'int', label: 'int' },
          ]}
          value={avgModeValue}
          placeholder={defaultPlaceholder(defAvg)}
          onChange={(mode) => {
            if (mode === undefined) {
              set({ average: undefined });
              return;
            }

            if (mode === 'int') {
              const init = typeof m.average === 'number'
                ? m.average
                : (typeof defAvg === 'number' ? defAvg : 10);
              set({ average: overrideOrUndef(Math.trunc(init), defAvg) });
              return;
            }

            if (mode === 'true') {
              set({ average: overrideOrUndef(true, defAvg) });
              return;
            }

            set({ average: overrideOrUndef(false, defAvg) });
          }}
        />
        {effAvgMode === 'int' && (
          <ParamNumber
            label="Average window"
            value={avgNumValue}
            placeholder={avgNumPlaceholder}
            onChange={(v) => {
              if (v === undefined) {
                set({ average: undefined });
                return;
              }

              const next = Math.trunc(v);
              const defForCompare = typeof defAvg === 'number' ? defAvg : undefined;
              set({ average: overrideOrUndef(next, defForCompare) });
            }}
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
