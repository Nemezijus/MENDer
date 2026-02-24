import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { maxFeatToModeVal, modeValToMaxFeat, makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function RfregSection({ m, set, sub, enums }) {
  const regTreeCriterion = makeSelectData(sub, 'criterion', enums?.RegTreeCriterion);
  const fMF = maxFeatToModeVal(m.max_features);
  return (
    <ParamGrid>
        <ParamNumber
          label="Trees (n_estimators)"
          value={m.n_estimators ?? 100}
          onChange={(v) => set({ n_estimators: v })}
          allowDecimal={false}
          min={1}
        />
        <ParamSelect
          label="Criterion"
          data={regTreeCriterion}
          value={m.criterion ?? 'squared_error'}
          onChange={(v) => set({ criterion: v })}
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
        <ParamCheckbox
          label="Use bootstrap"
          checked={m.bootstrap ?? true}
          onChange={(checked) => set({ bootstrap: checked })}
        />
        <ParamCheckbox
          label="OOB score"
          checked={!!m.oob_score}
          onChange={(checked) => set({ oob_score: checked })}
        />
        <ParamNumber
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
        <ParamNumber
          label="Random state"
          value={m.random_state ?? null}
          onChange={(v) => set({ random_state: v })}
          allowDecimal={false}
        />
        <ParamCheckbox
          label="Warm start"
          checked={!!m.warm_start}
          onChange={(checked) => set({ warm_start: checked })}
        />
        <ParamNumber
          label="CCP alpha"
          value={m.ccp_alpha ?? 0.0}
          onChange={(v) => set({ ccp_alpha: v })}
          step={0.0001}
        />
        <ParamNumber
          label="Max samples"
          value={m.max_samples ?? null}
          onChange={(v) => set({ max_samples: v })}
        />
      </ParamGrid>
  );
}
