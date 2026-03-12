import { Card, Stack, Group, Text, Button } from '@mantine/core';

/**
 * TuningConfigCard
 * A light wrapper to reduce repeated Card/Stack boilerplate for tuning config sections.
 */
export default function TuningConfigCard({
  title = 'Configuration',
  actionLabel,
  actionLabelLoading,
  onAction,
  loading = false,
  disabled = false,
  loadingHint,
  children,
}) {
  return (
    <Card withBorder shadow="sm" padding="lg">
      <Stack gap="md">
        <Group justify="space-between" wrap="nowrap">
          <Text fw={500}>{title}</Text>
          {onAction && actionLabel && (
            <Button size="xs" onClick={onAction} loading={loading} disabled={disabled}>
              {loading ? (actionLabelLoading || 'Working…') : actionLabel}
            </Button>
          )}
        </Group>
        {children}

        {loading && loadingHint ? (
          <Text size="xs" c="dimmed">
            {loadingHint}
          </Text>
        ) : null}
      </Stack>
    </Card>
  );
}
