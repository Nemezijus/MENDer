import { Stack, SimpleGrid, NumberInput, Checkbox } from '@mantine/core';

export default function LinregSection({ m, set, sub, enums }) {  return (
    <Stack gap="sm">
      <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="sm">
        <Checkbox
          label="Fit intercept"
          checked={!!m.fit_intercept}
          onChange={(e) =>
            set({ fit_intercept: e.currentTarget.checked })
          }
        />
        <Checkbox
          label="Copy X"
          checked={m.copy_X ?? true}
          onChange={(e) =>
            set({ copy_X: e.currentTarget.checked })
          }
        />
        <NumberInput
          label="Jobs (n_jobs)"
          value={m.n_jobs ?? null}
          onChange={(v) => set({ n_jobs: v })}
          allowDecimal={false}
        />
        <Checkbox
          label="Positive coefficients"
          checked={!!m.positive}
          onChange={(e) =>
            set({ positive: e.currentTarget.checked })
          }
        />
      </SimpleGrid>
    </Stack>
  );
}
