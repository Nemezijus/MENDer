import { Stack, SimpleGrid } from '@mantine/core';

/**
 * ParamGrid
 * - Standard layout used across model selection sections.
 * - Default to a 2-col grid (mobile-first), matching our common pattern.
 */
export default function ParamGrid({
  children,
  cols = { base: 1, sm: 2 },
  spacing = 'sm',
  gap = 'sm',
  ...props
}) {
  return (
    <Stack gap={gap} {...props}>
      <SimpleGrid cols={cols} spacing={spacing}>
        {children}
      </SimpleGrid>
    </Stack>
  );
}
