import { Box, Text } from '@mantine/core';

function fmt(x) {
  if (x == null || Number.isNaN(x)) return String(x);
  if (typeof x === 'number') return x.toFixed(3); // or 4 if you prefer
  return String(x);
}

export default function ValidationCurveAnalyticsResults({
  analytics,
  withinPct,
  metricLabel,
}) {
  if (!analytics) return null;

  const { best, minimal } = analytics;
  const metricText = metricLabel || 'score';

  return (
    <Box mt="sm">
      <Text size="sm">
        <Text span fw={600}>Peak validation</Text>: value{' '}
        <Text span fw={600}>{String(fmt(best.value))}</Text>, val {metricText} ={' '}
        <Text span fw={600}>{fmt(best.val)}</Text>, train ={' '}
        {fmt(best.train)}
      </Text>
      <Text size="sm">
        <Text span fw={600}>
          Recommended (≥ {(withinPct * 100).toFixed(0)}% of peak)
        </Text>
        : value <Text span fw={600}>{String(fmt(minimal.value))}</Text>, val {metricText} ={' '}
        <Text span fw={600}>{fmt(minimal.val)}</Text>, train ={' '}
        {fmt(minimal.train)}
      </Text>
    </Box>
  );
}
