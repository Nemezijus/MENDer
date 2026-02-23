import { Text, Tooltip } from '@mantine/core';

export default function SectionTitle({ title, tooltip, maw = 360, mb = 6, size = 'lg', fw = 500 }) {
  if (tooltip) {
    return (
      <Tooltip label={tooltip} multiline maw={maw} withArrow>
        <Text size={size} fw={fw} align="center" mb={mb}>
          {title}
        </Text>
      </Tooltip>
    );
  }

  return (
    <Text size={size} fw={fw} align="center" mb={mb}>
      {title}
    </Text>
  );
}
