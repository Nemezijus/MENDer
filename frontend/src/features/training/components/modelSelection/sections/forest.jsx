import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { maxFeatToModeVal, makeSelectData } from '../../../utils/modelSelectionUtils.js';
import {
  defaultPlaceholder,
  effectiveValue,
  overrideFromNullableSelect,
  overrideOrUndef,
  toNullableSelectValue,
} from '../utils/paramDefaults.js';

export default function ForestSection({ m, set, sub, enums, d }) {
  const forestCriterion = makeSelectData(sub, 'criterion', enums?.TreeCriterion);
  const forestClassWeight = makeSelectData(sub, 'class_weight', enums?.ForestClassWeight, { includeNoneLabel: true });
  const forestClassWeightUnavailable = forestClassWeight.length === 0;
  const defMaxFeat = d?.max_features;
  const effMaxFeat = effectiveValue(m.max_features, defMaxFeat);
  const hasMFOverride = m.max_features !== undefined;
  const curMF = hasMFOverride ? maxFeatToModeVal(m.max_features) : { mode: undefined, value: null };
  const effMF = effMaxFeat === undefined ? { mode: undefined, value: null } : maxFeatToModeVal(effMaxFeat);
  const mfNumValue = typeof m.max_features === 'number' ? m.max_features : undefined;
  const mfNumPlaceholder = typeof effMaxFeat === 'number' ? defaultPlaceholder(effMaxFeat) : undefined;
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
          value={hasMFOverride ? curMF.mode : undefined}
          placeholder={defaultPlaceholder(defMaxFeat)}
          onChange={(mode) => {
            if (mode === undefined) {
              set({ max_features: undefined });
              return;
            }

            if (mode === 'none') {
              set({ max_features: overrideOrUndef(null, defMaxFeat) });
              return;
            }

            if (mode === 'sqrt' || mode === 'log2') {
              set({ max_features: overrideOrUndef(mode, defMaxFeat) });
              return;
            }

            // numeric
            const init = mfNumValue ?? (typeof effMaxFeat === 'number' ? effMaxFeat : (mode === 'int' ? 1 : 0.5));
            const defForCompare = typeof defMaxFeat === 'number' ? defMaxFeat : undefined;
            set({ max_features: overrideOrUndef(init, defForCompare) });
          }}
        />
        {(effMF.mode === 'int' || effMF.mode === 'float') && (
          <ParamNumber
            label="Max features value"
            value={mfNumValue}
            placeholder={mfNumPlaceholder}
            onChange={(v) => {
              const defForCompare = typeof defMaxFeat === 'number' ? defMaxFeat : undefined;
              set({ max_features: overrideOrUndef(v, defForCompare) });
            }}
            step={effMF.mode === 'int' ? 1 : 0.01}
            allowDecimal={effMF.mode === 'float'}
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
        <ParamSelect
          label="Class weight"
          data={forestClassWeight}
          value={toNullableSelectValue(m.class_weight)}
          disabled={forestClassWeightUnavailable}
          placeholder={
            forestClassWeightUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.class_weight)
          }
          description={
            forestClassWeightUnavailable ? 'Schema did not provide class_weight options.' : undefined
          }
          onChange={(v) => set({ class_weight: overrideFromNullableSelect(v, d?.class_weight) })}
        />
        <ParamCheckbox
          label="Use bootstrap"
          checked={effectiveValue(m.bootstrap, d?.bootstrap)}
          onChange={(checked) => set({ bootstrap: overrideOrUndef(checked, d?.bootstrap) })}/>
        <ParamCheckbox
          label="OOB score"
          checked={effectiveValue(m.oob_score, d?.oob_score)}
          onChange={(checked) => set({ oob_score: overrideOrUndef(checked, d?.oob_score) })}/>
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
          onChange={(checked) => set({ warm_start: overrideOrUndef(checked, d?.warm_start) })}/>
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
