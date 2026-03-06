import { Stack, Text } from '@mantine/core';

import '../styles/settingsPanel.css';

import { useSettingsStore } from '../state/useSettingsStore.js';
import { useGlobalScaleSetting } from '../hooks/useGlobalScaleSetting.js';
import SettingsTabs from './SettingsTabs.jsx';

export default function SettingsPanel({ initialTab = 'scaling' }) {
  const { effectiveScaleMethod, setScaleMethod } = useGlobalScaleSetting();

  const metric = useSettingsStore((s) => s.metric);
  const setMetric = useSettingsStore((s) => s.setMetric);

  return (
    <Stack gap="md" className="settingsPanel">
      <Text fw={600} size="lg" className="settingsTitle">
        Global modelling settings
      </Text>

      <SettingsTabs
        initialTab={initialTab}
        scaleValue={effectiveScaleMethod}
        onScaleChange={setScaleMethod}
        metricValue={metric}
        onMetricChange={setMetric}
      />
    </Stack>
  );
}
