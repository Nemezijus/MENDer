import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function HgbSection({ m, set, sub, enums, d }) {
  const hgbLoss = makeSelectData(sub, 'loss', enums?.HGBLoss);
  const hgbES = m.early_stopping === 'auto' ? 'auto' : (m.early_stopping ? 'true' : 'false');
  const hgbMaxFeaturesValue = typeof m.max_features === 'number' ? m.max_features : 1.0;
  return (
    <ParamGrid>
        <ParamSelect
          label="Loss"
          data={hgbLoss}
          value={m.loss}

          placeholder={defaultPlaceholder(d?.loss)}
          onChange={(v) => set({ loss: overrideOrUndef(v, d?.loss) })}
        />
        <ParamNumber
          label="Learning rate"
          value={m.learning_rate}

          placeholder={defaultPlaceholder(d?.learning_rate)}
          onChange={(v) => set({ learning_rate: overrideOrUndef(v, d?.learning_rate) })}
          min={0}
          step={0.01}
        />
        <ParamNumber
          label="Iterations (max_iter)"
          value={m.max_iter}

          placeholder={defaultPlaceholder(d?.max_iter)}
          onChange={(v) => set({ max_iter: overrideOrUndef(v, d?.max_iter) })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="Max leaf nodes"
          value={m.max_leaf_nodes}

          placeholder={defaultPlaceholder(d?.max_leaf_nodes)}
          onChange={(v) => set({ max_leaf_nodes: overrideOrUndef(v, d?.max_leaf_nodes) })}
          allowDecimal={false}
          min={2}
        />
        <ParamNumber
          label="Max depth"
          value={m.max_depth}

          placeholder={defaultPlaceholder(d?.max_depth)}
          onChange={(v) => set({ max_depth: overrideOrUndef(v, d?.max_depth) })}
          allowDecimal={false}
        />
        <ParamNumber
          label="Min samples leaf"
          value={m.min_samples_leaf}

          placeholder={defaultPlaceholder(d?.min_samples_leaf)}
          onChange={(v) => set({ min_samples_leaf: overrideOrUndef(v, d?.min_samples_leaf) })}
          allowDecimal={false}
          min={1}
        />
        <ParamNumber
          label="L2 regularization"
          value={m.l2_regularization}

          placeholder={defaultPlaceholder(d?.l2_regularization)}
          onChange={(v) => set({ l2_regularization: overrideOrUndef(v, d?.l2_regularization) })}
          min={0}
          step={0.01}
        />
        <ParamNumber
          label="Max features (fraction)"
          value={hgbMaxFeaturesValue}
          onChange={(v) => {
            // Mantine ParamNumber may provide a string; backend expects a number.
            if (v == null || v === '') set({ max_features: null });
            else set({ max_features: typeof v === 'number' ? v : Number(v) });
          }}
          min={0}
          max={1}
          step={0.01}
        />
        <ParamNumber
          label="Max bins"
          value={m.max_bins}

          placeholder={defaultPlaceholder(d?.max_bins)}
          onChange={(v) => set({ max_bins: overrideOrUndef(v, d?.max_bins) })}
          allowDecimal={false}
          min={2}
        />
        <ParamSelect
          label="Early stopping"
          data={[
            { value: 'auto', label: 'auto' },
            { value: 'true', label: 'true' },
            { value: 'false', label: 'false' },
          ]}
          value={hgbES}
          onChange={(v) => {
            if (v === 'auto') set({ early_stopping: 'auto' });
            else if (v === 'true') set({ early_stopping: true });
            else set({ early_stopping: false });
          }}
        />
        <TextInput
          label="Scoring"
          placeholder="loss"
          value={m.scoring}

          placeholder={defaultPlaceholder(d?.scoring)}
          onChange={(e) => {
            const t = e.currentTarget.value;
            set({ scoring: t === '' ? null : t });
          }}
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
        <ParamNumber
          label="Tolerance"
          value={m.tol}

          placeholder={defaultPlaceholder(d?.tol)}
          onChange={(v) => set({ tol: overrideOrUndef(v, d?.tol) })}
          step={1e-7}
          min={0}
        />
        <ParamNumber
          label="Random state"
          value={m.random_state}

          placeholder={defaultPlaceholder(d?.random_state)}
          onChange={(v) => set({ random_state: overrideOrUndef(v, d?.random_state) })}
          allowDecimal={false}
        />
      </ParamGrid>
  );
}
