import { Stack, Text, Group } from '@mantine/core';

import VotingSimpleEstimatorRow from './VotingSimpleEstimatorRow.jsx';

export default function VotingSimpleEstimatorList({
  estimators,
  algoOptions,
  onAlgoChangeAt,
  onRemoveAt,
}) {
  return (
    <Stack gap="sm">
      <Text size="sm" c="dimmed">
        Simple mode uses each model’s default hyperparameters. Switch to Advanced to edit parameters
        and set weights.
      </Text>

      <Group align="flex-start" wrap="nowrap" gap="md">
        <Stack style={{ flex: 1 }} gap="sm">
          {estimators
            .map((s, idx) => ({ s, idx }))
            .filter((x) => x.idx % 2 === 0)
            .map(({ s, idx }) => (
              <VotingSimpleEstimatorRow
                key={idx}
                idx={idx}
                algo={s?.model?.algo}
                algoOptions={algoOptions}
                onAlgoChange={(nextAlgo) => onAlgoChangeAt(idx, nextAlgo)}
                onRemove={() => onRemoveAt(idx)}
                disableRemove={estimators.length <= 2}
              />
            ))}
        </Stack>

        <Stack style={{ flex: 1 }} gap="sm">
          {estimators
            .map((s, idx) => ({ s, idx }))
            .filter((x) => x.idx % 2 === 1)
            .map(({ s, idx }) => (
              <VotingSimpleEstimatorRow
                key={idx}
                idx={idx}
                algo={s?.model?.algo}
                algoOptions={algoOptions}
                onAlgoChange={(nextAlgo) => onAlgoChangeAt(idx, nextAlgo)}
                onRemove={() => onRemoveAt(idx)}
                disableRemove={estimators.length <= 2}
              />
            ))}
        </Stack>
      </Group>
    </Stack>
  );
}
