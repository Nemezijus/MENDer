import { useState } from 'react';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { maxFeatToModeVal, modeValToMaxFeat, makeSelectData } from '../../../utils/modelSelectionUtils.js';
import {
  defaultPlaceholder,
  effectiveValue,
  overrideFromNullableSelect,
  overrideOrUndef,
  toNullableSelectValue,
} from '../utils/paramDefaults.js';

export default function TreeSection({ m, set, sub, enums, d }) {
  const treeCriterion = makeSelectData(sub, 'criterion', enums?.TreeCriterion);
  const treeSplitter = makeSelectData(sub, 'splitter', enums?.TreeSplitter);
  const treeClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  const [mfModeDraft, setMfModeDraft] = useState(null);
  const defMaxFeatures = d?.max_features;
  const effMaxFeatures = effectiveValue(m.max_features, defMaxFeatures);
  const effMF = maxFeatToModeVal(effMaxFeatures);
  const hasOverrideMF = m.max_features !== undefined;
  const mfFromOverride = hasOverrideMF ? maxFeatToModeVal(m.max_features) : null;

  const mfModeValue = hasOverrideMF
    ? mfFromOverride.mode
    : (mfModeDraft ?? ((effMF.mode === 'int' || effMF.mode === 'float') ? effMF.mode : undefined));

  const mfValueMode = mfModeDraft ?? (hasOverrideMF ? mfFromOverride.mode : effMF.mode);
  const showMfValue = mfValueMode === 'int' || mfValueMode === 'float';
  const mfNumValue = typeof m.max_features === 'number' ? m.max_features : undefined;
  const mfNumPlaceholder = (effMF.mode === 'int' || effMF.mode === 'float')
    ? defaultPlaceholder(effMF.value)
    : undefined;
  return (
    <ParamGrid>
        <ParamSelect
          label="Criterion"
          data={treeCriterion}
          value={m.criterion}

          placeholder={defaultPlaceholder(d?.criterion)}
          onChange={(v) => set({ criterion: overrideOrUndef(v, d?.criterion) })}
        />
        <ParamSelect
          label="Splitter"
          data={treeSplitter}
          value={m.splitter}

          placeholder={defaultPlaceholder(d?.splitter)}
          onChange={(v) => set({ splitter: overrideOrUndef(v, d?.splitter) })}
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
          value={mfModeValue}
          placeholder={defaultPlaceholder(defMaxFeatures)}
          onChange={(mode) => {
            if (mode === undefined) {
              setMfModeDraft(null);
              set({ max_features: undefined });
              return;
            }

            if (mode === 'sqrt' || mode === 'log2') {
              setMfModeDraft(null);
              set({ max_features: overrideOrUndef(mode, defMaxFeatures) });
              return;
            }

            if (mode === 'none') {
              setMfModeDraft(null);
              set({ max_features: overrideOrUndef(null, defMaxFeatures) });
              return;
            }

            if (mode === 'int' || mode === 'float') {
              setMfModeDraft(mode);
              if (typeof m.max_features !== 'number') set({ max_features: undefined });
              return;
            }

            setMfModeDraft(null);
            set({ max_features: overrideOrUndef(modeValToMaxFeat(mode, effMF.value), defMaxFeatures) });
          }}
        />
        {showMfValue && (
          <ParamNumber
            label="Max features value"
            value={mfNumValue}
            placeholder={mfNumPlaceholder}
            onChange={(v) => {
              if (v === undefined) {
                set({ max_features: undefined });
                if (mfModeDraft && typeof defMaxFeatures !== 'number') setMfModeDraft(null);
                return;
              }

              const mode = mfModeDraft ?? (effMF.mode === 'int' || effMF.mode === 'float' ? effMF.mode : 'float');
              const next = mode === 'int' ? Math.trunc(v) : v;
              setMfModeDraft(null);
              const defForCompare = typeof defMaxFeatures === 'number' ? defMaxFeatures : undefined;
              set({ max_features: overrideOrUndef(next, defForCompare) });
            }}
            step={mfValueMode === 'int' ? 1 : 0.01}
            allowDecimal={mfValueMode === 'float'}
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
          data={treeClassWeight}
          value={toNullableSelectValue(m.class_weight)}
          placeholder={defaultPlaceholder(d?.class_weight)}
          onChange={(v) => set({ class_weight: overrideFromNullableSelect(v, d?.class_weight) })}
        />
        <ParamNumber
          label="CCP alpha"
          value={m.ccp_alpha}

          placeholder={defaultPlaceholder(d?.ccp_alpha)}
          onChange={(v) => set({ ccp_alpha: overrideOrUndef(v, d?.ccp_alpha) })}
          step={0.0001}
        />
      </ParamGrid>
  );
}
