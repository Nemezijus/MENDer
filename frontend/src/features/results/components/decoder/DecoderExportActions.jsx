import { Group, Text, Button, Tooltip } from '@mantine/core';

export default function DecoderExportActions({
  onExportPreview,
  onExportFull,
  isExportingFull,
  canExportFull,
}) {
  return (
    <Group justify="space-between" align="center" wrap="wrap">
      <Text size="sm" fw={600}>
        Preview table
      </Text>

      <Group gap="xs" wrap="wrap">
        <Button size="xs" variant="light" onClick={onExportPreview}>
          Export preview CSV
        </Button>
        <Tooltip
          label={
            canExportFull
              ? 'Export full evaluation-set decoder outputs as CSV.'
              : 'No artifact UID available for this run.'
          }
          multiline
          maw={320}
          withArrow
        >
          <Button
            size="xs"
            variant="light"
            loading={isExportingFull}
            disabled={!canExportFull}
            onClick={onExportFull}
          >
            Export full CSV
          </Button>
        </Tooltip>
      </Group>
    </Group>
  );
}
