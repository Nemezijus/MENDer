import { Card, Stack, Group, Text } from '@mantine/core';

/**
 * ParamGroup
 * - Light card wrapper for grouping related params.
 * - Optional helper to reduce repeated Card/Stack boilerplate.
 */
export default function ParamGroup({ title, rightSection, children, ...props }) {
  return (
    <Card withBorder p="md" {...props}>
      <Stack gap="sm">
        {(title || rightSection) && (
          <Group justify="space-between" align="center">
            {title ? <Text fw={600}>{title}</Text> : <span />}
            {rightSection || null}
          </Group>
        )}
        {children}
      </Stack>
    </Card>
  );
}
