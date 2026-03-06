import { Tabs } from '@mantine/core';

import ScalingCard from '../../../shared/ui/config/ScalingCard.jsx';
import FeatureCard from '../../../shared/ui/config/FeatureCard.jsx';
import MetricCard from '../../../shared/ui/config/MetricCard.jsx';

export default function SettingsTabs({
  initialTab = 'scaling',
  scaleValue,
  onScaleChange,
  metricValue,
  onMetricChange,
}) {
  return (
    <Tabs defaultValue={initialTab} keepMounted={false} className="settingsTabs">
      <Tabs.List grow>
        <Tabs.Tab value="scaling">Scaling</Tabs.Tab>
        <Tabs.Tab value="metric">Metric</Tabs.Tab>
        <Tabs.Tab value="features">Features</Tabs.Tab>
      </Tabs.List>

      <Tabs.Panel value="scaling" pt="md">
        <ScalingCard value={scaleValue} onChange={onScaleChange} />
      </Tabs.Panel>

      <Tabs.Panel value="metric" pt="md">
        <MetricCard value={metricValue} onChange={onMetricChange} />
      </Tabs.Panel>

      <Tabs.Panel value="features" pt="md">
        <FeatureCard />
      </Tabs.Panel>
    </Tabs>
  );
}
