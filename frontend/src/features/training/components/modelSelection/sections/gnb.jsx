import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import { parseCsvFloats, formatCsvFloats } from '../../../utils/modelSelectionUtils.js';
import { defaultPlaceholder, overrideOrUndef } from '../utils/paramDefaults.js';

export default function GnbSection({ m, set, sub, enums, d }) {
  const priorsPlaceholder = defaultPlaceholder(d?.priors) ?? 'e.g. 0.2, 0.8';

  return (
    <ParamGrid>
      <ParamNumber
        label="Variance smoothing"
        value={m.var_smoothing}
        placeholder={defaultPlaceholder(d?.var_smoothing)}
        onChange={(v) => set({ var_smoothing: overrideOrUndef(v, d?.var_smoothing) })}
        step={1e-9}
        min={0}
      />

      <TextInput
        label="Priors (comma-separated)"
        placeholder={priorsPlaceholder}
        value={formatCsvFloats(m.priors)}
        onChange={(e) => {
          const vals = parseCsvFloats(e.currentTarget.value);
          set({ priors: overrideOrUndef(vals ?? undefined, d?.priors) });
        }}
      />
    </ParamGrid>
  );
}
