import { Text } from '@mantine/core';

export default function BooleanNote() {
  return (
    <Text size="sm" c="dimmed">
      This is a boolean parameter. The validation curve will automatically
      evaluate both <Text span fw={500}>true</Text> and{' '}
      <Text span fw={500}>false</Text>.
    </Text>
  );
}
