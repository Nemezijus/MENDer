import { Card, Stack, Text, Divider } from '@mantine/core';
import ConfusionMatrixResults from './visualizations/ConfusionMatrixResults.jsx';
import ClassificationMetricResults from './visualizations/ClassificationMetricResults.jsx';
import RocResults from './visualizations/RocResults.jsx';

export default function ClassificationResultsPanel({ trainResult }) {
  if (!trainResult) return null;

  const confusion = trainResult.confusion;
  const roc = trainResult.roc;
  const metricName = trainResult.metric_name;

  const hasConfusion =
    confusion &&
    Array.isArray(confusion.matrix) &&
    confusion.matrix.length > 0 &&
    Array.isArray(confusion.labels) &&
    confusion.labels.length > 0;

  const hasRoc = !!roc && Array.isArray(roc.curves) && roc.curves.length > 0;

  if (!hasConfusion && !hasRoc) {
    return null;
  }

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500}>Classification diagnostics</Text>

        {hasConfusion && (
          <>
            <ConfusionMatrixResults confusion={confusion} />
            <ClassificationMetricResults
              confusion={confusion}
              metricName={metricName}
            />
          </>
        )}

        {hasRoc && (
          <>
            {hasConfusion && <Divider />}
            <RocResults roc={roc} />
          </>
        )}
      </Stack>
    </Card>
  );
}
