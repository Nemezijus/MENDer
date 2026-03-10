import { Card, Stack, Text, Group, Badge, Tabs, Box } from '@mantine/core';

import { useFilesConstraintsQuery } from '../../../shared/schema/useFilesConstraintsQuery.js';
import { useProductionDataStore } from '../state/useProductionDataStore.js';
import { useModelArtifactStore } from '../../modelArtifacts/state/useModelArtifactStore.js';

import DataSummaryCard from './helpers/DataSummaryCard.jsx';
import ProductionIndividualFilesTab from './production/IndividualFilesTab.jsx';
import ProductionCompoundFileTab from './production/CompoundFileTab.jsx';

import { ProductionDataIntroText } from '../../../shared/content/help/DataFilesHelpTexts.jsx';

export default function ProductionDataUploadCard() {
  const { data: filesConstraints } = useFilesConstraintsQuery();
  const acceptExts = Array.isArray(filesConstraints?.allowed_exts)
    ? filesConstraints.allowed_exts.join(',')
    : undefined;
  const defaultXKey = filesConstraints?.data_default_keys?.x_key ?? 'X';
  const defaultYKey = filesConstraints?.data_default_keys?.y_key ?? 'y';

  const inspectReport = useProductionDataStore((s) => s.inspectReport);

  const xPath = useProductionDataStore((s) => s.xPath);
  const yPath = useProductionDataStore((s) => s.yPath);
  const npzPath = useProductionDataStore((s) => s.npzPath);

  const modelArtifact = useModelArtifactStore(
    (s) => s?.artifact || s?.activeArtifact || s?.modelArtifact || null,
  );

  // Task shown can be inferred from the inspected production data
  const effectiveTask = inspectReport?.task_inferred || null;

  // Persisted display
  const xDisplay = useProductionDataStore((s) => s.xDisplay);
  const yDisplay = useProductionDataStore((s) => s.yDisplay);
  const npzDisplay = useProductionDataStore((s) => s.npzDisplay);
  const setXDisplay = useProductionDataStore((s) => s.setXDisplay);
  const setYDisplay = useProductionDataStore((s) => s.setYDisplay);
  const setNpzDisplay = useProductionDataStore((s) => s.setNpzDisplay);

  return (
    <Stack gap="md">
      <Card className="specialCard" withBorder shadow="sm" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <Box className="dataFilesHeaderSpacer" />
            <Text fw={700} size="lg" align="center" className="dataFilesHeaderTitle">
              Production data
            </Text>
            {inspectReport ? <Badge color="green">Ready</Badge> : <Badge color="gray">Not loaded</Badge>}
          </Group>

          <ProductionDataIntroText />

          <Tabs defaultValue="individual" keepMounted={false}>
            <Tabs.List grow>
              <Tabs.Tab value="individual">Individual files</Tabs.Tab>
              <Tabs.Tab value="compound">Compound file</Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="individual" pt="md">
              <ProductionIndividualFilesTab
                acceptExts={acceptExts}
                setXDisplayGlobal={setXDisplay}
                setYDisplayGlobal={setYDisplay}
                modelArtifact={modelArtifact}
                initialXPath={xPath}
                initialYPath={yPath}
                initialXDisplay={xDisplay}
                initialYDisplay={yDisplay}
              />
            </Tabs.Panel>

            <Tabs.Panel value="compound" pt="md">
              <ProductionCompoundFileTab
                acceptExts={acceptExts}
                defaultXKey={defaultXKey}
                defaultYKey={defaultYKey}
                setNpzDisplayGlobal={setNpzDisplay}
                modelArtifact={modelArtifact}
                initialNpzPath={npzPath}
                initialNpzDisplay={npzDisplay}
              />
            </Tabs.Panel>
          </Tabs>
        </Stack>
      </Card>

      <DataSummaryCard
        inspectReport={inspectReport}
        effectiveTask={effectiveTask}
        xPath={xPath}
        yPath={yPath}
        npzPath={npzPath}
        xDisplay={xDisplay}
        yDisplay={yDisplay}
        npzDisplay={npzDisplay}
        modelArtifact={modelArtifact}
        showSuggestion={false}
      />
    </Stack>
  );
}
