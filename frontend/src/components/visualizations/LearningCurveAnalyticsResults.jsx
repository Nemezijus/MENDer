import { Box, Text } from '@mantine/core';

export default function LearningCurveAnalyticsResults({ analytics, withinPct }) {
  if (!analytics) return null;

  const { best, minimal } = analytics;

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
