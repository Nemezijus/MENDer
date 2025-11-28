import { Box, Stack } from '@mantine/core';

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
    <Box
      style={{
        display: 'flex',
        gap: 'var(--mantine-spacing-lg)',
        alignItems: 'flex-start', // <- prevents vertical stretching
        flexWrap: 'wrap', // nicer on small screens
      }}
    >
      {/* Left column: Scaling + Features stacked */}
      <Box style={{ flex: 1, minWidth: 0 }}>
        <Stack gap="lg">
          <ScalingCard value={scaleMethod} onChange={setScaleMethod} />
          <FeatureCard />
        </Stack>
      </Box>

      {/* Right column: Metric, natural height */}
      <Box
        style={{
          flex: 1,
          minWidth: 260, // small guard so it doesn't get too narrow
          alignSelf: 'flex-start',
        }}
      >
        <MetricCard value={metric} onChange={setMetric} />
      </Box>
    </Box>
  );
}
