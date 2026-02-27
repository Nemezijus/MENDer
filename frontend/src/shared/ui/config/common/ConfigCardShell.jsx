import { Card, Stack, Text, Group, Box } from '@mantine/core';

import '../../styles/cards.css';

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
    <Card className="configCardShell">
      <Stack className="configCardStack">
        <Text className="configCardTitle">{title}</Text>

        <Group className="configCardColumns">
          <Box className="configCardColumnLeft">{left}</Box>
          <Box className="configCardColumnRight">{right}</Box>
        </Group>

        {helpVisible && help ? <Box className="configCardHelp">{help}</Box> : null}

        {children}
      </Stack>
    </Card>
  );
}
