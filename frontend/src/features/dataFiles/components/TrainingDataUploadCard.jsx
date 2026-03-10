import { Card, Stack, Text, Alert, Group, Badge, Tabs, Box, Select } from '@mantine/core';

import { useDataStore } from '../state/useDataStore.js';
import { useFilesConstraintsQuery } from '../../../shared/schema/useFilesConstraintsQuery.js';

import DataSummaryCard from './helpers/DataSummaryCard.jsx';
import TrainingIndividualFilesTab from './training/IndividualFilesTab.jsx';
import TrainingCompoundFileTab from './training/CompoundFileTab.jsx';

import { TrainingDataIntroText } from '../../../shared/content/help/DataFilesHelpTexts.jsx';

export default function TrainingDataUploadCard() {
  const { data: filesConstraints } = useFilesConstraintsQuery();
  const acceptExts = Array.isArray(filesConstraints?.allowed_exts)
    ? filesConstraints.allowed_exts.join(',')
    : undefined;
  const defaultXKey = filesConstraints?.data_default_keys?.x_key ?? 'X';
  const defaultYKey = filesConstraints?.data_default_keys?.y_key ?? 'y';

  const inspectReport = useDataStore((s) => s.inspectReport);
  const setInspectReport = useDataStore((s) => s.setInspectReport);

  const taskSelected = useDataStore((s) => s.taskSelected);
  const setTaskSelected = useDataStore((s) => s.setTaskSelected);

  const xKey = useDataStore((s) => s.xKey);
  const setXKey = useDataStore((s) => s.setXKey);
  const yKey = useDataStore((s) => s.yKey);
  const setYKey = useDataStore((s) => s.setYKey);

  const setXPath = useDataStore((s) => s.setXPath);
  const setYPath = useDataStore((s) => s.setYPath);
  const setNpzPath = useDataStore((s) => s.setNpzPath);

  const xPath = useDataStore((s) => s.xPath);
  const yPath = useDataStore((s) => s.yPath);
  const npzPath = useDataStore((s) => s.npzPath);

  // Persisted display
  const xDisplay = useDataStore((s) => s.xDisplay);
  const yDisplay = useDataStore((s) => s.yDisplay);
  const npzDisplay = useDataStore((s) => s.npzDisplay);
  const setXDisplay = useDataStore((s) => s.setXDisplay);
  const setYDisplay = useDataStore((s) => s.setYDisplay);
  const setNpzDisplay = useDataStore((s) => s.setNpzDisplay);

  const dataReady = !!inspectReport && (inspectReport?.n_samples ?? 0) > 0;
  const taskInferredRaw = inspectReport?.task_inferred || null;
  // Backwards compatibility: older backend/meta may still report "clustering".
  const taskInferred = taskInferredRaw === 'clustering' ? 'unsupervised' : taskInferredRaw;
  const effectiveTask = taskSelected || taskInferred || null;

  return (
    <Stack gap="md">
      <Card  className="specialCard" withBorder shadow="sm" padding="lg">
        <Stack gap="md">
          <Group justify="space-between" align="center">
            <Box className="dataFilesHeaderSpacer" />
            <Text fw={700} size="lg" align="center" className="dataFilesHeaderTitle">
              Training data
            </Text>
            {dataReady ? <Badge color="green">Ready</Badge> : <Badge color="gray">Not loaded</Badge>}
          </Group>

          <TrainingDataIntroText />

          <Tabs defaultValue="individual" keepMounted={false}>
            <Tabs.List grow>
              <Tabs.Tab value="individual">Individual files</Tabs.Tab>
              <Tabs.Tab value="compound">Compound file</Tabs.Tab>
            </Tabs.List>

            <Tabs.Panel value="individual" pt="md">
              <TrainingIndividualFilesTab
                acceptExts={acceptExts}
                setInspectReportGlobal={setInspectReport}
                setXPathGlobal={setXPath}
                setYPathGlobal={setYPath}
                setNpzPathGlobal={setNpzPath}
                xKeyGlobal={xKey}
                yKeyGlobal={yKey}
                setXDisplayGlobal={setXDisplay}
                setYDisplayGlobal={setYDisplay}
                initialXPath={xPath}
                initialYPath={yPath}
                initialXDisplay={xDisplay}
                initialYDisplay={yDisplay}
              />
            </Tabs.Panel>

            <Tabs.Panel value="compound" pt="md">
              <TrainingCompoundFileTab
                acceptExts={acceptExts}
                defaultXKey={defaultXKey}
                defaultYKey={defaultYKey}
                setInspectReportGlobal={setInspectReport}
                setXPathGlobal={setXPath}
                setYPathGlobal={setYPath}
                setNpzPathGlobal={setNpzPath}
                xKey={xKey}
                yKey={yKey}
                setXKey={setXKey}
                setYKey={setYKey}
                setNpzDisplayGlobal={setNpzDisplay}
                initialNpzPath={npzPath}
                initialNpzDisplay={npzDisplay}
              />
            </Tabs.Panel>
          </Tabs>

          <Select
            label="Task (suggestion)"
            description="Overrides inferred task. You can leave it empty."
            data={[
              { value: 'classification', label: 'classification' },
              { value: 'regression', label: 'regression' },
              { value: 'unsupervised', label: 'unsupervised' },
            ]}
            value={taskSelected || null}
            placeholder={taskInferred ? `inferred: ${taskInferred}` : 'leave empty'}
            onChange={(v) => setTaskSelected(v)}
            clearable
          />

          {effectiveTask === 'unsupervised' && (
            <Alert color="blue" variant="light" title="Unsupervised mode">
              <Text size="sm">
                In unsupervised mode, <b>y</b> (labels/targets) are ignored even if provided.
              </Text>
            </Alert>
          )}
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
      />
    </Stack>
  );
}
