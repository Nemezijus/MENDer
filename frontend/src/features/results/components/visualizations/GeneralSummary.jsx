import { Stack, Text, Card } from '@mantine/core';

export default function GeneralSummary({
  isCV,
  metricName,
  metricValue,
  meanScore,
  stdScore,
  nTrain,
  nTest,
}) {
  const formatNumber = (v) =>
    typeof v === 'number' ? v.toFixed(3) : v;

  return (
    <Card
      withBorder
      radius="md"
      padding="sm"
      style={{ borderStyle: 'solid', borderWidth: 1, borderColor: '#acc7e0ff' }}
    >
      <Stack gap="xs">
        <Text size="md">
          <Text span fw={500}>Metric: </Text>
          <Text span fw={700}>{metricName}</Text>
        </Text>

        {isCV ? (
          <Text size="md">
            <Text span fw={500}>Score (CV): </Text>
            <Text span fw={700}>
              mean {formatNumber(meanScore)} ± {formatNumber(stdScore)}
            </Text>
          </Text>
        ) : (
          <Text size="md">
            <Text span fw={500}>Score: </Text>
            <Text span fw={700}>{formatNumber(metricValue)}</Text>
          </Text>
        )}

        {!isCV && (nTrain != null && nTest != null) && (
          <Text size="md">
            <Text span fw={500}>Train / Test: </Text>
            <Text span fw={700}>
              {nTrain} / {nTest}
            </Text>
          </Text>
        )}
      </Stack>
    </Card>
  );
}
