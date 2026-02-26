import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { maxFeatToModeVal, modeValToMaxFeat, makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function RfregSection({ m, set, sub, enums, d }) {
  const regTreeCriterion = makeSelectData(sub, 'criterion', enums?.RegTreeCriterion);
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
          data={regTreeCriterion}
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
        <ParamNumber
          label="Min weight fraction"
          value={m.min_weight_fraction_leaf}

          placeholder={defaultPlaceholder(d?.min_weight_fraction_leaf)}
          onChange={(v) => set({ min_weight_fraction_leaf: overrideOrUndef(v, d?.min_weight_fraction_leaf) })}
          step={0.01}
          min={0}
          max={1}
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
        <ParamNumber
          label="Min impurity decrease"
          value={m.min_impurity_decrease}

          placeholder={defaultPlaceholder(d?.min_impurity_decrease)}
          onChange={(v) => set({ min_impurity_decrease: overrideOrUndef(v, d?.min_impurity_decrease) })}
          step={0.0001}
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
        <ParamCheckbox
          label="Warm start"
          checked={effectiveValue(m.warm_start, d?.warm_start)}
          onChange={(checked) => set({ warm_start: overrideOrUndef(checked, d?.warm_start) })}
        />
        <ParamNumber
          label="CCP alpha"
          value={m.ccp_alpha}

          placeholder={defaultPlaceholder(d?.ccp_alpha)}
          onChange={(v) => set({ ccp_alpha: overrideOrUndef(v, d?.ccp_alpha) })}
          step={0.0001}
        />
        <ParamNumber
          label="Max samples"
          value={m.max_samples}

          placeholder={defaultPlaceholder(d?.max_samples)}
          onChange={(v) => set({ max_samples: overrideOrUndef(v, d?.max_samples) })}
        />
      </ParamGrid>
  );
}
