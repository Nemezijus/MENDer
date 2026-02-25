import { Box, Checkbox, Stack, Text } from '@mantine/core';

export default function EnumValueSelector({ allowedValues, selectedValues, onToggle }) {
  const allowed = Array.isArray(allowedValues) ? allowedValues : [];
  const selected = Array.isArray(selectedValues) ? selectedValues : [];

  if (!allowed.length) return null;

  return (
    <Box>
      <Text size="sm" fw={500} mb={4}>
        Values to include
      </Text>
      <Stack gap={4}>
        {allowed.map((v) => (
          <Checkbox
            key={String(v)}
            label={v === null ? 'None' : String(v)}
            checked={selected.includes(v)}
            onChange={() => onToggle?.(v)}
          />
        ))}
      </Stack>
    </Box>
  );
}
