import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import { maxFeatToModeVal, modeValToMaxFeat, makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function TreeregSection({ m, set, sub, enums }) {
  const treeSplitter = makeSelectData(sub, 'splitter', enums?.TreeSplitter);
  const regTreeCriterion = makeSelectData(sub, 'criterion', enums?.RegTreeCriterion);
  const tMF = maxFeatToModeVal(m.max_features);
  return (
    <ParamGrid>
        <ParamSelect
          label="Criterion"
          data={regTreeCriterion}
          value={m.criterion ?? 'squared_error'}
          onChange={(v) => set({ criterion: v })}
        />
        <ParamSelect
          label="Splitter"
          data={treeSplitter}
          value={m.splitter ?? 'best'}
          onChange={(v) => set({ splitter: v })}
        />
        <ParamNumber
          label="Max depth"
          value={m.max_depth ?? null}
          onChange={(v) => set({ max_depth: v })}
        />
        <ParamNumber
          label="Min samples split"
          value={m.min_samples_split ?? 2}
          onChange={(v) => set({ min_samples_split: v })}
        />
        <ParamNumber
          label="Min samples leaf"
          value={m.min_samples_leaf ?? 1}
          onChange={(v) => set({ min_samples_leaf: v })}
        />
        <ParamNumber
          label="Min weight fraction"
          value={m.min_weight_fraction_leaf ?? 0.0}
          onChange={(v) => set({ min_weight_fraction_leaf: v })}
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
          value={tMF.mode}
          onChange={(mode) => set({ max_features: modeValToMaxFeat(mode, tMF.value) })}
        />
        {(tMF.mode === 'int' || tMF.mode === 'float') && (
          <ParamNumber
            label="Max features value"
            value={tMF.value ?? null}
            onChange={(v) => set({ max_features: modeValToMaxFeat(tMF.mode, v) })}
            step={tMF.mode === 'int' ? 1 : 0.01}
            allowDecimal={tMF.mode === 'float'}
          />
        )}
        <ParamNumber
          label="Max leaf nodes"
          value={m.max_leaf_nodes ?? null}
          onChange={(v) => set({ max_leaf_nodes: v })}
          allowDecimal={false}
        />
        <ParamNumber
          label="Min impurity decrease"
          value={m.min_impurity_decrease ?? 0.0}
          onChange={(v) => set({ min_impurity_decrease: v })}
          step={0.0001}
        />
        <ParamNumber
          label="Random state"
          value={m.random_state ?? null}
          onChange={(v) => set({ random_state: v })}
          allowDecimal={false}
        />
        <ParamNumber
          label="CCP alpha"
          value={m.ccp_alpha ?? 0.0}
          onChange={(v) => set({ ccp_alpha: v })}
          step={0.0001}
        />
      </ParamGrid>
  );
}
