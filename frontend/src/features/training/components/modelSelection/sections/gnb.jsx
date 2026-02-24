import { TextInput } from '@mantine/core';
import ParamGrid from '../inputs/ParamGrid.jsx';
import ParamNumber from '../inputs/ParamNumber.jsx';
import { parseCsvFloats, formatCsvFloats } from '../../../utils/modelSelectionUtils.js';

export default function GnbSection({ m, set, sub, enums }) {  return (
    <ParamGrid>
        <ParamNumber
          label="Variance smoothing"
          value={m.var_smoothing ?? 1e-9}
          onChange={(v) => set({ var_smoothing: v })}
          step={1e-9}
          min={0}
        />
        <TextInput
          label="Priors (comma-separated)"
          placeholder="e.g. 0.2, 0.8"
          value={formatCsvFloats(m.priors)}
          onChange={(e) => set({ priors: parseCsvFloats(e.currentTarget.value) })}
        />
      </ParamGrid>
  );
}
