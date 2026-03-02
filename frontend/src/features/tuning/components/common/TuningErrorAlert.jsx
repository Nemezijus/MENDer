import { Alert, Text } from '@mantine/core';

export default function TuningErrorAlert({ error }) {
  if (!error) return null;

  return (
    <Alert color="red" variant="light">
      <Text fw={500}>Error</Text>
      <Text size="sm" className="tuningErrorPreWrap">
        {String(error)}
      </Text>
    </Alert>
  );
}
