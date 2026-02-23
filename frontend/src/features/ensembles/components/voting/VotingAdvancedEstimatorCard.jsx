import { Card, Stack, Group, Text, ActionIcon, TextInput, Box } from '@mantine/core';
import { IconTrash } from '@tabler/icons-react';

import ModelSelectionCard from '../../../training/components/ModelSelectionCard.jsx';

import ParamNumber from '../common/ParamNumber.jsx';

export default function VotingAdvancedEstimatorCard({
  idx,
  estimator,
  onUpdate,
  onRemove,
  disableRemove,
  models,
  enums,
}) {
  return (
    <Card withBorder radius="md" p="md">
      <Stack gap="sm">
        <Group justify="space-between" align="center">
          <Text fw={600}>Estimator {idx + 1}</Text>

          <ActionIcon
            variant="subtle"
            color="red"
            onClick={onRemove}
            disabled={disableRemove}
            title="Remove estimator"
          >
            <IconTrash size={18} />
          </ActionIcon>
        </Group>

        <Group grow align="flex-end" wrap="wrap">
          <TextInput
            label="Name (optional)"
            placeholder="auto"
            value={estimator?.name ?? ''}
            onChange={(e) => onUpdate({ name: e.currentTarget.value })}
          />

          <ParamNumber
            label="Weight (optional)"
            value={estimator?.weight ?? ''}
            onChange={(v) => onUpdate({ weight: v })}
            step={0.5}
            min={0}
            emptyToUndefined
          />
        </Group>

        <Box>
          <ModelSelectionCard
            model={estimator?.model}
            onChange={(next) => onUpdate({ model: next })}
            schema={models?.schema}
            enums={enums}
            models={models}
            showHelp={false}
          />
        </Box>
      </Stack>
    </Card>
  );
}
