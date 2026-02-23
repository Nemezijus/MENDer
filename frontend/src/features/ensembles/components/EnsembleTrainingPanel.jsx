// frontend/src/features/ensembles/components/EnsembleTrainingPanel.jsx
import { Tabs, Stack } from '@mantine/core';

import VotingEnsemblePanel from './VotingEnsemblePanel.jsx';
import BaggingEnsemblePanel from './BaggingEnsemblePanel.jsx';
import AdaBoostEnsemblePanel from './AdaBoostEnsemblePanel.jsx';
import XGBoostEnsemblePanel from './XGBoostEnsemblePanel.jsx';

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
          <BaggingEnsemblePanel />
        </Tabs.Panel>

        <Tabs.Panel value="adaboost" pt="md">
          <AdaBoostEnsemblePanel />
        </Tabs.Panel>

        <Tabs.Panel value="xgboost" pt="md">
          <XGBoostEnsemblePanel />
        </Tabs.Panel>
      </Tabs>
    </Stack>
  );
}
