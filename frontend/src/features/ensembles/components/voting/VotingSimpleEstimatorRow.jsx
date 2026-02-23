import { Group, ActionIcon } from '@mantine/core';
import { IconTrash } from '@tabler/icons-react';

import ParamSelect from '../common/ParamSelect.jsx';

export default function VotingSimpleEstimatorRow({
  idx,
  algo,
  algoOptions,
  onAlgoChange,
  onRemove,
  disableRemove,
}) {
  return (
    <Group align="flex-end" wrap="nowrap">
      <ParamSelect
        style={{ flex: 1, minWidth: 180, maxWidth: 360 }}
        label={`Estimator ${idx + 1}`}
        value={algo || null}
        onChange={(v) => onAlgoChange(v || algo)}
        data={algoOptions}
      />

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
  );
}
