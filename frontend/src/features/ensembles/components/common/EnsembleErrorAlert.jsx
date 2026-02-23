import { Alert } from '@mantine/core';

export default function EnsembleErrorAlert({ error }) {
  if (!error) return null;
  return (
    <Alert color="red" title="Error">
      {String(error)}
    </Alert>
  );
}
