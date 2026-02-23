import { Card, Stack, Text } from '@mantine/core';

export default function ResultsCardShell({ title, children }) {
  return (
    <Card withBorder radius="md" p="md">
      <Stack gap="xs">
        <Text fw={650} size="xl" align="center">
          {title}
        </Text>
        {children}
      </Stack>
    </Card>
  );
}
