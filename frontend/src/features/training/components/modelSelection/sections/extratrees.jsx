import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { maxFeatToModeVal, modeValToMaxFeat, makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function ExtratreesSection({ m, set, sub, enums, d }) {
  const forestCriterion = makeSelectData(sub, 'criterion', enums?.TreeCriterion);
  const forestClassWeight = makeSelectData(sub, 'class_weight', (enums?.ForestClassWeight ?? ['balanced', 'balanced_subsample', null]), { includeNoneLabel: true });
  const fMF = maxFeatToModeVal(m.max_features);
  return (
    <ParamGrid>
        <ParamNumber
          label="Trees (n_estimators)"
          value={m.n_estimators}

          placeholder={defaultPlaceholder(d?.n_estimators)}
          onChange={(v) => set({ n_estimators: overrideOrUndef(v, d?.n_estimators) })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Criterion"
          data={forestCriterion}
          value={m.criterion}

          placeholder={defaultPlaceholder(d?.criterion)}
          onChange={(v) => set({ criterion: overrideOrUndef(v, d?.criterion) })}
        />
        <ParamNumber
          label="Max depth"
          value={m.max_depth}

          placeholder={defaultPlaceholder(d?.max_depth)}
          onChange={(v) => set({ max_depth: overrideOrUndef(v, d?.max_depth) })}
        />
        <ParamNumber
          label="Min samples split"
          value={m.min_samples_split}

          placeholder={defaultPlaceholder(d?.min_samples_split)}
          onChange={(v) => set({ min_samples_split: overrideOrUndef(v, d?.min_samples_split) })}
        />
        <ParamNumber
          label="Min samples leaf"
          value={m.min_samples_leaf}

          placeholder={defaultPlaceholder(d?.min_samples_leaf)}
          onChange={(v) => set({ min_samples_leaf: overrideOrUndef(v, d?.min_samples_leaf) })}
        />
        <ParamSelect
          label="Max features mode"
          data={[
            { value: 'sqrt', label: 'sqrt' },
            { value: 'log2', label: 'log2' },
            { value: 'int', label: 'int' },
            { value: 'float', label: 'float' },
            { value: 'none', label: 'none' },
          ]}
          value={fMF.mode}
          onChange={(mode) => set({ max_features: modeValToMaxFeat(mode, fMF.value) })}
        />
        {(fMF.mode === 'int' || fMF.mode === 'float') && (
          <ParamNumber
            label="Max features value"
            value={fMF.value ?? null}
            onChange={(v) => set({ max_features: modeValToMaxFeat(fMF.mode, v) })}
            step={fMF.mode === 'int' ? 1 : 0.01}
            allowDecimal={fMF.mode === 'float'}
          />
        )}
        <ParamNumber
          label="Max leaf nodes"
          value={m.max_leaf_nodes}

          placeholder={defaultPlaceholder(d?.max_leaf_nodes)}
          onChange={(v) => set({ max_leaf_nodes: overrideOrUndef(v, d?.max_leaf_nodes) })}
          allowDecimal={false}
        />
        <ParamSelect
          label="Class weight"
          data={forestClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        <ParamCheckbox
          label="Use bootstrap"
          checked={effectiveValue(m.bootstrap, d?.bootstrap)}
          onChange={(checked) => set({ bootstrap: overrideOrUndef(checked, d?.bootstrap) })}
        />
        <ParamCheckbox
          label="OOB score"
          checked={effectiveValue(m.oob_score, d?.oob_score)}
          onChange={(checked) => set({ oob_score: overrideOrUndef(checked, d?.oob_score) })}
        />
        <ParamNumber
          label="Jobs (n_jobs)"
          value={m.n_jobs}

          placeholder={defaultPlaceholder(d?.n_jobs)}
          onChange={(v) => set({ n_jobs: overrideOrUndef(v, d?.n_jobs) })}
          allowDecimal={false}
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
