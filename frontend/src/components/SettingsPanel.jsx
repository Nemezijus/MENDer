// src/components/SettingsPanel.jsx
import { Stack, Tabs, Text } from '@mantine/core';

import ScalingCard from './ScalingCard.jsx';
import FeatureCard from './FeatureCard.jsx';
import MetricCard from './MetricCard.jsx';
import { useSettingsStore } from '../state/useSettingsStore.js';
import { useSchemaDefaults } from '../state/SchemaDefaultsContext.jsx';

export default function SettingsPanel() {
  const { scale } = useSchemaDefaults();

  const scaleMethod = useSettingsStore((s) => s.scaleMethod);
  const setScaleMethod = useSettingsStore((s) => s.setScaleMethod);
  const metric = useSettingsStore((s) => s.metric);
  const setMetric = useSettingsStore((s) => s.setMetric);

  // Store is overrides-only. Display schema defaults when unset.
  const scaleDefault = scale?.defaults?.method;
  const effectiveScaleMethod = scaleMethod ?? scaleDefault ?? null;

  const handleScaleChange = (v) => {
    const next = v || undefined;
    // Keep store as an override: clear if user selects the schema default.
    if (scaleDefault != null && next === scaleDefault) {
      setScaleMethod(undefined);
      return;
    }
    setScaleMethod(next);
  };

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
          <ScalingCard value={effectiveScaleMethod} onChange={handleScaleChange} />
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
