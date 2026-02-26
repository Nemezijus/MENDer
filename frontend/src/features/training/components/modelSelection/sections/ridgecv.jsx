import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import ParamSelect from '../inputs/ParamSelect.jsx';
import ParamCheckbox from '../inputs/ParamCheckbox.jsx';
import { parseCsvFloats, formatCsvFloats, makeSelectData } from '../../../utils/modelSelectionUtils.js';
import {
  defaultPlaceholder,
  effectiveValue,
  overrideFromNullableSelect,
  overrideOrUndef,
  toNullableSelectValue,
} from '../utils/paramDefaults.js';

export default function RidgecvSection({ m, set, sub, enums, d }) {
  const ridgecvGcvMode = makeSelectData(sub, 'gcv_mode', undefined, { includeNoneLabel: true });
  const ridgecvGcvModeUnavailable = ridgecvGcvMode.length === 0;
  const alphasPlaceholder = defaultPlaceholder(d?.alphas) ?? 'e.g. 0.1, 1.0, 10.0';
  return (
    <ParamGrid>
        <TextInput
          label="Alphas (comma-separated)"
          placeholder={alphasPlaceholder}
          value={formatCsvFloats(m.alphas)}
          onChange={(e) => {
            const vals = parseCsvFloats(e.currentTarget.value);
            set({ alphas: overrideOrUndef(vals ?? undefined, d?.alphas) });
          }}
        />
        <ParamCheckbox
          label="Fit intercept"
          checked={effectiveValue(m.fit_intercept, d?.fit_intercept)}
          onChange={(checked) => set({ fit_intercept: overrideOrUndef(checked, d?.fit_intercept) })}
        />
        <TextInput
          label="Scoring"
          value={m.scoring ?? ''}
          placeholder={defaultPlaceholder(d?.scoring) ?? '(optional)'}
          onChange={(e) => {
            const t = e.currentTarget.value;
            set({ scoring: overrideOrUndef(t === '' ? undefined : t, d?.scoring) });
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
          value={toNullableSelectValue(m.gcv_mode)}
          disabled={ridgecvGcvModeUnavailable}
          placeholder={ridgecvGcvModeUnavailable ? 'Schema enums unavailable' : defaultPlaceholder(d?.gcv_mode)}
          description={ridgecvGcvModeUnavailable ? 'Schema did not provide gcv_mode options.' : undefined}
          onChange={(v) => set({ gcv_mode: overrideFromNullableSelect(v, d?.gcv_mode) })}
        />
        <ParamCheckbox
          label="Alpha per target"
          checked={effectiveValue(m.alpha_per_target, d?.alpha_per_target)}
          onChange={(checked) => set({ alpha_per_target: overrideOrUndef(checked, d?.alpha_per_target) })}
        />
      </ParamGrid>
  );
}
