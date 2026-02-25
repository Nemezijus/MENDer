import { Card, Stack, Text, Group, Box } from '@mantine/core';

/**
 * Shared layout shell for the config cards in shared/ui/config.
 *
 * Keeps a consistent structure:
 *  - centered title
 *  - two-column row (controls + intro)
 *  - full-width help block
 *  - optional additional content below
 */
export default function ConfigCardShell({
  title,
  left,
  right,
  help,
  helpVisible = true,
  children,
}) {
  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Stack gap="md">
        <Text fw={700} size="lg" align="center">
          {title}
        </Text>

        <Group align="flex-start" gap="xl" grow wrap="nowrap">
          <Box style={{ flex: 1, minWidth: 0 }}>{left}</Box>
          <Box style={{ flex: 1, minWidth: 220 }}>{right}</Box>
        </Group>

        {helpVisible && help ? <Box mt="md">{help}</Box> : null}

        {children}
      </Stack>
    </Card>
  );
}
