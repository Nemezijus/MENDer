import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function SvrSection({ m, set, sub, enums }) {
  const svmKernel = makeSelectData(sub, 'kernel', enums?.SVMKernel);
  const gammaMode = typeof m.gamma === 'number' ? 'numeric' : (m.gamma ?? 'scale');
  const gammaValue = typeof m.gamma === 'number' ? m.gamma : 0.1;
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <NumberInput
          label="C"
          value={m.C ?? 1.0}
          onChange={(v) => set({ C: v })}
          min={0}
          step={0.1}
        />
        <Select
          label="Kernel"
          data={svmKernel}
          value={m.kernel ?? 'rbf'}
          onChange={(v) => set({ kernel: v })}
        />
        <NumberInput
          label="Degree"
          value={m.degree ?? 3}
          onChange={(v) => set({ degree: v })}
          allowDecimal={false}
          min={1}
        />
        <Select
          label="Gamma mode"
          data={[
            { value: 'scale', label: 'scale' },
            { value: 'auto', label: 'auto' },
            { value: 'numeric', label: 'numeric' },
          ]}
          value={gammaMode}
          onChange={(mode) => {
            if (mode === 'numeric') set({ gamma: gammaValue });
            else set({ gamma: mode });
          }}
        />
        {gammaMode === 'numeric' && (
          <NumberInput
            label="Gamma value"
            value={gammaValue}
            onChange={(v) => set({ gamma: v })}
            min={0}
            step={0.01}
          />
        )}
        <NumberInput
          label="Coef0"
          value={m.coef0 ?? 0.0}
          onChange={(v) => set({ coef0: v })}
          step={0.1}
        />
        <Checkbox
          label="Shrinking"
          checked={m.shrinking ?? true}
          onChange={(e) => set({ shrinking: e.currentTarget.checked })}
        />
        <NumberInput
          label="Epsilon"
          value={m.epsilon ?? 0.1}
          onChange={(v) => set({ epsilon: v })}
          min={0}
          step={0.01}
        />
        <NumberInput
          label="Tolerance (tol)"
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={1e-4}
          min={0}
        />
        <NumberInput
          label="Cache size (MB)"
          value={m.cache_size ?? 200.0}
          onChange={(v) => set({ cache_size: v })}
          min={0}
          step={10}
        />
        <NumberInput
          label="Max iterations"
          value={m.max_iter ?? -1}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
        />
      </SimpleGrid>
    </Stack>
  );
}
