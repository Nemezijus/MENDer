import { Tabs } from '@mantine/core';

import '../styles/dataFilesPanel.css';

import TrainingDataUploadCard from './TrainingDataUploadCard.jsx';
import SavedModelUploadCard from '../../modelArtifacts/components/SavedModelUploadCard.jsx';
import ProductionDataUploadCard from './ProductionDataUploadCard.jsx';

export default function DataFilesPanel() {
  return (
      <Tabs defaultValue="training" keepMounted={false} className="appPanelTabs dataFilesTabs">
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
  );
}
