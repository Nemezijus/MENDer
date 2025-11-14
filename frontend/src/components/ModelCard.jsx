import { Card, Text } from '@mantine/core';

export default function ModelCard() {
  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Text fw={500} mb="xs">Model</Text>
      <Text size="sm" c="dimmed">
        Details about the last trained model will appear here in the future.
      </Text>
    </Card>
  );
}
