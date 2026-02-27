import { Stack, Text } from '@mantine/core';

// NOTE:
// This component is intentionally split out from ModelHelpText.jsx so that
// the large model parameter help blocks can be lazy-loaded only when the user
// expands the help section.

export function ModelIntroText() {
  return (
    <Stack gap="xs">
      <Text fw={500} size="sm">
        What is a model?
      </Text>

      <Text size="xs" c="dimmed">
        A model is the algorithm that learns patterns from your training data
        and makes predictions on new data. Different models make different
        assumptions and trade off accuracy, interpretability, robustness, and
        training speed.
      </Text>
    </Stack>
  );
}

export default ModelIntroText;
