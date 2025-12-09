// src/components/SettingsPanel.jsx
import { Stack, Tabs, Text } from '@mantine/core';

import ScalingCard from './ScalingCard.jsx';
import FeatureCard from './FeatureCard.jsx';
import MetricCard from './MetricCard.jsx';
import { useSettingsStore } from '../state/useSettingsStore.js';

export default function SettingsPanel() {
  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const setScaleMethod = useSettingsStore((s) => s.setScaleMethod);
  const metric = useSettingsStore((s) => s.metric);
  const setMetric = useSettingsStore((s) => s.setMetric);

  return (
    <Stack gap="md">
      <Text fw={600} size="lg">
        Global modelling settings
      </Text>

      <Tabs defaultValue="scaling" keepMounted={false}>
        <Tabs.List grow>
          <Tabs.Tab value="scaling">Scaling</Tabs.Tab>
          <Tabs.Tab value="metric">Metric</Tabs.Tab>
          <Tabs.Tab value="features">Features</Tabs.Tab>
        </Tabs.List>

        <Tabs.Panel value="scaling" pt="md">
          <ScalingCard value={scaleMethod} onChange={setScaleMethod} />
        </Tabs.Panel>

        <Tabs.Panel value="metric" pt="md">
          <MetricCard value={metric} onChange={setMetric} />
        </Tabs.Panel>

        <Tabs.Panel value="features" pt="md">
          <FeatureCard />
        </Tabs.Panel>
      </Tabs>
    </Stack>
  );
}
