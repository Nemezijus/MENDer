import { Card, Stack, Text } from '@mantine/core';
import { useRunModelResultsCtx } from '../state/RunModelResultsContext.jsx';
import GeneralSummary from './visualizations/GeneralSummary.jsx';
import KFoldResults from './visualizations/KFoldResults.jsx';
import ConfusionMatrixResults from './visualizations/ConfusionMatrixResults.jsx';
import BaselineShufflingResults from './visualizations/BaselineShufflingResults.jsx';

export default function ModelTrainingResultsPanel() {
  const { result } = useRunModelResultsCtx();

  const isCV = result && Array.isArray(result.fold_scores);

  if (!result) {
    return (
      <Card withBorder radius="md" shadow="sm" padding="md">
        <Text fw={500} mb="xs">Results</Text>
        <Text size="sm" c="dimmed">
          Run a model to see results here.
        </Text>
      </Card>
    );
  }

  const hasBaseline =
    Array.isArray(result.shuffled_scores) && result.shuffled_scores.length > 0;

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500}>Results</Text>

        <GeneralSummary
          isCV={isCV}
          metricName={result.metric_name}
          metricValue={result.metric_value}
          meanScore={result.mean_score}
          stdScore={result.std_score}
          nTrain={result.n_train}
          nTest={result.n_test}
        />

        {isCV ? (
          <KFoldResults
            title={'K-fold data splitting results'}
            foldScores={result.fold_scores}
            metricName={result.metric_name}
            meanScore={result.mean_score}
            stdScore={result.std_score}
          />
        ) : (
          <ConfusionMatrixResults
            confusion={result.confusion}
          />
        )}

        {hasBaseline && (
          <BaselineShufflingResults
            title={isCV ? 'Shuffle-label baseline (CV mean)' : 'Shuffle-label baseline'}
            metricName={result.metric_name}
            referenceLabel={isCV ? 'real mean' : 'real'}
            referenceValue={isCV ? result.mean_score : result.metric_value}
            shuffledScores={result.shuffled_scores}
            pValue={result.p_value}
          />
        )}

        {Array.isArray(result.notes) && result.notes.length > 0 && (
          <>
            <Text fw={500} size="sm" mt="sm">Notes</Text>
            <ul style={{ marginTop: 4 }}>
              {result.notes.map((n, i) => (
                <li key={i}>
                  <Text size="sm">{n}</Text>
                </li>
              ))}
            </ul>
          </>
        )}
      </Stack>
    </Card>
  );
}
