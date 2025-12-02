// frontend/src/components/visualizations/GridSearchAnalyticsResults.jsx
import { Box, Text } from '@mantine/core';

function fmt(x) {
  if (x == null || Number.isNaN(x)) return String(x);
  if (typeof x === 'number') return x.toFixed(3);
  return String(x);
}

export default function GridSearchAnalyticsResults({
  result,
  metricLabel,
  pipelineName1,
  pipelineName2,
  shortName1,
  shortName2,
}) {
  if (!result) return null;

  const metricText = metricLabel || 'score';
  const bestScore = result.best_score;
  const bestParams = result.best_params || {};

  if (bestScore == null || !bestParams) {
    return (
      <Box mt="sm">
        <Text size="sm">
          Could not infer a best combination from the grid search results.
        </Text>
      </Box>
    );
  }

  // Use the pipeline param names we derived in the panel to pull best values
  const p1NamePipeline = pipelineName1 || Object.keys(bestParams)[0];
  const p2NamePipeline = pipelineName2 || Object.keys(bestParams)[1];

  const label1 = shortName1 || p1NamePipeline;
  const label2 = shortName2 || p2NamePipeline;

  const p1Val = bestParams[p1NamePipeline];
  const p2Val = bestParams[p2NamePipeline];

  return (
    <Box mt="sm">
      <Text size="sm">
        <Text span fw={600}>Best score</Text>: {metricText} ={' '}
        <Text span fw={600}>{fmt(bestScore)}</Text>
      </Text>
      <Text size="sm">
        at {label1} = <Text span fw={600}>{fmt(p1Val)}</Text>,{' '}
        {label2} = <Text span fw={600}>{fmt(p2Val)}</Text>
      </Text>
    </Box>
  );
}
