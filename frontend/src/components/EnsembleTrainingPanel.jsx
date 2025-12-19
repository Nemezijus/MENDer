// frontend/src/components/EnsembleTrainingPanel.jsx
import { Tabs, Stack, Card, Text } from '@mantine/core';

import VotingEnsemblePanel from './ensembles/VotingEnsemblePanel.jsx';

export default function EnsembleTrainingPanel() {
  return (
    <Stack gap="md">
      <Tabs defaultValue="voting" keepMounted={true}>
        <Tabs.List grow>
          <Tabs.Tab value="voting">Voting</Tabs.Tab>
          <Tabs.Tab value="bagging">Bagging</Tabs.Tab>
          <Tabs.Tab value="adaboost">AdaBoost</Tabs.Tab>
          <Tabs.Tab value="xgboost">XGBoost</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="voting" pt="md">
          <VotingEnsemblePanel />
        </Tabs.Panel>

        <Tabs.Panel value="bagging" pt="md">
          <Card withBorder radius="md" p="md">
            <Text size="sm">
              Bagging ensemble training will be added after Voting is complete.
            </Text>
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="adaboost" pt="md">
          <Card withBorder radius="md" p="md">
            <Text size="sm">
              AdaBoost ensemble training will be added after Voting is complete.
            </Text>
          </Card>
        </Tabs.Panel>

        <Tabs.Panel value="xgboost" pt="md">
          <Card withBorder radius="md" p="md">
            <Text size="sm">
              XGBoost ensemble training will be added after Voting is complete.
            </Text>
          </Card>
        </Tabs.Panel>
      </Tabs>
    </Stack>
  );
}
