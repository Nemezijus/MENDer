import { Box, Text } from '@mantine/core';

import { fmtAny as fmt } from '../../../../shared/utils/valueFormat.js';

export default function ValidationCurveAnalyticsResults({
  analytics,
  withinPct,
  metricLabel,
}) {
  if (!analytics) return null;

  const { best, minimal } = analytics;
  const metricText = metricLabel || 'score';

  // In unsupervised mode, validation can be unavailable (no predict()).
  // Show N/A rather than crashing.
  if (!best || !minimal) {
    return (
      <Box mt="sm">
        <Text size="sm">
          <Text span fw={600}>Peak validation</Text>: N/A
        </Text>
        <Text size="sm">
          <Text span fw={600}>
            Recommended (≥ {(withinPct * 100).toFixed(0)}% of peak)
          </Text>
          : N/A
        </Text>
      </Box>
    );
  }

  return (
    <Box mt="sm">
      <Text size="sm">
        <Text span fw={600}>Peak validation</Text>: value{' '}
        <Text span fw={600}>{fmt(best.value)}</Text>, val {metricText} ={' '}
        <Text span fw={600}>{fmt(best.val)}</Text>, train ={' '}
        {fmt(best.train)}
      </Text>
      <Text size="sm">
        <Text span fw={600}>
          Recommended (≥ {(withinPct * 100).toFixed(0)}% of peak)
        </Text>
        : value <Text span fw={600}>{fmt(minimal.value)}</Text>, val {metricText} ={' '}
        <Text span fw={600}>{fmt(minimal.val)}</Text>, train ={' '}
        {fmt(minimal.train)}
      </Text>
    </Box>
  );
}
