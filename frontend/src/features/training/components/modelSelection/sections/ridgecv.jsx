import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { parseCsvFloats, formatCsvFloats, makeSelectData } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, effectiveValue, overrideOrUndef } from '../utils/paramDefaults.js';

export default function RidgecvSection({ m, set, sub, enums, d }) {
  const ridgecvGcvMode = makeSelectData(sub, 'gcv_mode', ['auto', 'svd', 'eigen', null], { includeNoneLabel: true });
  return (
    <ParamGrid>
        <TextInput
          label="Alphas (comma-separated)"
          placeholder="e.g. 0.1, 1.0, 10.0"
          value={formatCsvFloats(m.alphas)}
          onChange={(e) => set({ alphas: parseCsvFloats(e.currentTarget.value) || [0.1, 1.0, 10.0] })}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={effectiveValue(m.fit_intercept, d?.fit_intercept)}
          onChange={(checked) => set({ fit_intercept: overrideOrUndef(checked, d?.fit_intercept) })}
        />
        <TextInput
          label="Scoring"
          placeholder="(optional)"
          value={m.scoring}

          placeholder={defaultPlaceholder(d?.scoring)}
          onChange={(e) => {
            const t = e.currentTarget.value;
            set({ scoring: t === '' ? null : t });
          }}
        />
        <ParamNumber
          label="CV folds"
          value={m.cv}

          placeholder={defaultPlaceholder(d?.cv)}
          onChange={(v) => set({ cv: overrideOrUndef(v, d?.cv) })}
          allowDecimal={false}
          min={2}
        />
        <ParamSelect
          label="GCV mode"
          data={ridgecvGcvMode}
          value={m.gcv_mode == null ? 'none' : String(m.gcv_mode)}
          onChange={(v) => set({ gcv_mode: fromSelectNullable(v) })}
        />
        <ParamCheckbox
          label="Alpha per target"
          checked={effectiveValue(m.alpha_per_target, d?.alpha_per_target)}
          onChange={(checked) => set({ alpha_per_target: overrideOrUndef(checked, d?.alpha_per_target) })}
        />
      </ParamGrid>
  );
}
