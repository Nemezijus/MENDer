import { Card, Stack, Text } from '@mantine/core';
import { useResultsStore } from '../state/useResultsStore.js';
import GeneralSummary from './visualizations/GeneralSummary.jsx';
import KFoldResults from './visualizations/KFoldResults.jsx';
import ConfusionMatrixResults from './visualizations/ConfusionMatrixResults.jsx';
import BaselineShufflingResults from './visualizations/BaselineShufflingResults.jsx';

export default function ModelTrainingResultsPanel() {
  const trainResult = useResultsStore((s) => s.trainResult);

  const isCV = trainResult && Array.isArray(trainResult.fold_scores);

  if (!trainResult) {
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
    Array.isArray(trainResult.shuffled_scores) && trainResult.shuffled_scores.length > 0;

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500}>Results</Text>

        <GeneralSummary
          isCV={isCV}
          metricName={trainResult.metric_name}
          metricValue={trainResult.metric_value}
          meanScore={trainResult.mean_score}
          stdScore={trainResult.std_score}
          nTrain={trainResult.n_train}
          nTest={trainResult.n_test}
        />

        {isCV ? (
          <KFoldResults
            title={'K-fold data splitting results'}
            foldScores={trainResult.fold_scores}
            metricName={trainResult.metric_name}
            meanScore={trainResult.mean_score}
            stdScore={trainResult.std_score}
          />
        ) : (
          <ConfusionMatrixResults
            confusion={trainResult.confusion}
          />
        )}

        {hasBaseline && (
          <BaselineShufflingResults
            title={isCV ? 'Shuffle-label baseline (CV mean)' : 'Shuffle-label baseline'}
            metricName={trainResult.metric_name}
            referenceLabel={isCV ? 'real mean' : 'real'}
            referenceValue={isCV ? trainResult.mean_score : trainResult.metric_value}
            shuffledScores={trainResult.shuffled_scores}
            pValue={trainResult.p_value}
          />
        )}

        {Array.isArray(trainResult.notes) && trainResult.notes.length > 0 && (
          <>
            <Text fw={500} size="sm" mt="sm">Notes</Text>
            <ul style={{ marginTop: 4 }}>
              {trainResult.notes.map((n, i) => (
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
