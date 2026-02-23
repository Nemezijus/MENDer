import { Group, Text, ActionIcon, SegmentedControl } from '@mantine/core';
import { IconRefresh } from '@tabler/icons-react';

export default function EnsemblePanelHeader({
  title,
  mode,
  onModeChange,
  onReset,
  showModeToggle = true,
  disableReset = false,
}) {
  return (
    <Group justify="space-between" align="center">
      <Text fw={700} size="lg">
        {title}
      </Text>

      <Group gap="xs">
        <ActionIcon
          variant="subtle"
          onClick={onReset}
          title="Reset to defaults"
          disabled={disableReset}
        >
          <IconRefresh size={18} />
        </ActionIcon>

        {showModeToggle && (
          <SegmentedControl
            value={mode}
            onChange={onModeChange}
            data={[
              { value: 'simple', label: 'Simple' },
              { value: 'advanced', label: 'Advanced' },
            ]}
          />
        )}
      </Group>
    </Group>
  );
}
