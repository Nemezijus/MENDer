import { Box, Text } from '@mantine/core';

export default function LearningCurveAnalyticsResults({ analytics, withinPct }) {
  if (!analytics) return null;

  const { best, minimal } = analytics;

  // In unsupervised mode, validation scores can be unavailable (no predict()).
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
        <Text span fw={600}>Peak validation</Text>: size{' '}
        <Text span fw={600}>{best.size}</Text>, val ={' '}
        <Text span fw={600}>{best.val.toFixed(3)}</Text>, train ={' '}
        {best.train.toFixed(3)}
      </Text>
      <Text size="sm">
        <Text span fw={600}>
          Recommended (≥ {(withinPct * 100).toFixed(0)}% of peak)
        </Text>
        : size <Text span fw={600}>{minimal.size}</Text>, val ={' '}
        <Text span fw={600}>{minimal.val.toFixed(3)}</Text>, train ={' '}
        {minimal.train.toFixed(3)}
      </Text>
    </Box>
  );
}
