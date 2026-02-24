import { Stack, SimpleGrid, NumberInput, Select, Checkbox } from '@mantine/core';
import { fromSelectNullable } from '../../../../../shared/utils/schema/jsonSchema.js';
import { makeSelectData } from '../../../utils/modelSelectionUtils.js';

export default function SvmSection({ m, set, sub, enums }) {
  const svmKernel = makeSelectData(sub, 'kernel', enums?.SVMKernel);
  const svmDecisionShape = makeSelectData(sub, 'decision_function_shape', enums?.SVMDecisionShape);
  const svmClassWeight = makeSelectData(sub, 'class_weight', (enums?.ClassWeightBalanced ?? ['balanced', null]), { includeNoneLabel: true });
  const gammaMode = typeof m.gamma === 'number' ? 'numeric' : (m.gamma ?? 'scale');
  const gammaValue = typeof m.gamma === 'number' ? m.gamma : 0.1;
  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <Select
          label="Kernel"
          data={svmKernel}
          value={m.kernel ?? 'rbf'}
          onChange={(v) => set({ kernel: v })}
        />
        <NumberInput
          label="C (penalty)"
          value={m.C ?? 1.0}
          onChange={(v) => set({ C: v })}
          min={0}
          step={0.1}
        />
        <NumberInput
          label="Degree (poly)"
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
            if (mode === 'numeric') {
              set({
                gamma:
                  typeof gammaValue === 'number' ? gammaValue : 0.1,
              });
            } else {
              set({ gamma: mode });
            }
          }}
        />
        {gammaMode === 'numeric' && (
          <NumberInput
            label="Gamma value"
            value={gammaValue}
            onChange={(v) => set({ gamma: v })}
            min={0}
            step={0.001}
          />
        )}
        <NumberInput
          label="Coef0"
          value={m.coef0 ?? 0.0}
          onChange={(v) => set({ coef0: v })}
          step={0.001}
        />
        <Checkbox
          label="Use shrinking"
          checked={!!m.shrinking}
          onChange={(e) =>
            set({ shrinking: e.currentTarget.checked })
          }
        />
        <Checkbox
          label="Enable probability"
          checked={!!m.probability}
          onChange={(e) =>
            set({ probability: e.currentTarget.checked })
          }
        />
        <NumberInput
          label="Tolerance (tol)"
          value={m.tol ?? 1e-3}
          onChange={(v) => set({ tol: v })}
          step={0.0001}
        />
        <NumberInput
          label="Cache size (MB)"
          value={m.cache_size ?? 200.0}
          onChange={(v) => set({ cache_size: v })}
        />
        <Select
          label="Class weight"
          data={svmClassWeight}
          value={m.class_weight == null ? 'none' : String(m.class_weight)}
          onChange={(v) => set({ class_weight: fromSelectNullable(v) })}
        />
        <NumberInput
          label="Max iterations"
          value={m.max_iter ?? -1}
          onChange={(v) => set({ max_iter: v })}
          allowDecimal={false}
        />
        <Select
          label="Decision shape"
          data={svmDecisionShape}
          value={m.decision_function_shape ?? 'ovr'}
          onChange={(v) =>
            set({ decision_function_shape: v })
          }
        />
        <Checkbox
          label="Break ties"
          checked={!!m.break_ties}
          onChange={(e) =>
            set({ break_ties: e.currentTarget.checked })
          }
        />
      </SimpleGrid>
    </Stack>
  );
}
