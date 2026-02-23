import { Stack, Text } from '@mantine/core';

import VotingAdvancedEstimatorCard from './VotingAdvancedEstimatorCard.jsx';

export default function VotingAdvancedEstimatorList({
  estimators,
  onUpdateAt,
  onRemoveAt,
  models,
  enums,
}) {
  return (
    <Stack gap="md">
      <Text size="sm" c="dimmed">
        Advanced mode lets you tune each estimator and optionally assign weights.
      </Text>

      {estimators.map((s, idx) => (
        <VotingAdvancedEstimatorCard
          key={idx}
          idx={idx}
          estimator={s}
          onUpdate={(patch) => onUpdateAt(idx, patch)}
          onRemove={() => onRemoveAt(idx)}
          disableRemove={estimators.length <= 2}
          models={models}
          enums={enums}
        />
      ))}
    </Stack>
  );
}
