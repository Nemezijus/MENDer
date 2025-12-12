
import { Card, Tabs } from '@mantine/core';

import TrainingDataUploadCard from './TrainingDataUploadCard.jsx';
import SavedModelUploadCard from './SavedModelUploadCard.jsx';
import ProductionDataUploadCard from './ProductionDataUploadCard.jsx';

export default function UploadPanel() {
  return (
    <Card withBorder shadow="sm" radius="md" padding="lg">
      <Tabs defaultValue="training" keepMounted={false}>
        <Tabs.List grow>
          <Tabs.Tab value="training">Training data</Tabs.Tab>
          <Tabs.Tab value="saved-model">Saved model</Tabs.Tab>
          <Tabs.Tab value="production">Production data</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="training" pt="md">
          <TrainingDataUploadCard />
        </Tabs.Panel>

        <Tabs.Panel value="saved-model" pt="md">
          <SavedModelUploadCard />
        </Tabs.Panel>

        <Tabs.Panel value="production" pt="md">
          <ProductionDataUploadCard />
        </Tabs.Panel>
      </Tabs>
    </Card>
  );
}
