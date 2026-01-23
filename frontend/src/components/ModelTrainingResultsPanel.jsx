import { Card, Stack, Text } from '@mantine/core';
import { useResultsStore } from '../state/useResultsStore.js';
import GeneralSummary from './visualizations/GeneralSummary.jsx';
import KFoldResults from './visualizations/KFoldResults.jsx';
import BaselineShufflingResults from './visualizations/BaselineShufflingResults.jsx';
import ClassificationResultsPanel from './ClassificationResultsPanel.jsx';
import RegressionResultsPanel from './RegressionResultsPanel.jsx';
import DecoderOutputsResults from './visualizations/DecoderOutputsResults.jsx';

export default function ModelTrainingResultsPanel() {
  const trainResult = useResultsStore((s) => s.trainResult);

  const isCV = trainResult && Array.isArray(trainResult.fold_scores);

  if (!trainResult) {
    return (
      <Card withBorder radius="md" shadow="sm" padding="md">
        <Text fw={500} mb="xs">
          Results
        </Text>
        <Text size="sm" c="dimmed">
          No results yet. Run a model from the &quot;Run a model&quot; tab.
        </Text>
      </Card>
    );
  }

  const isClassification =
    trainResult?.artifact?.kind === 'classification' ||
    // fallback heuristic: confusion matrix present & non-empty
    (trainResult.confusion &&
      Array.isArray(trainResult.confusion.matrix) &&
      trainResult.confusion.matrix.length > 0);

  const isRegression =
    trainResult?.artifact?.kind === 'regression' ||
    (trainResult.regression && typeof trainResult.regression === 'object');

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500}>Results</Text>

        <GeneralSummary
          isCV={!!isCV}
          metricName={trainResult.metric_name}
          metricValue={trainResult.metric_value}
          meanScore={trainResult.mean_score}
          stdScore={trainResult.std_score}
          nTrain={trainResult.n_train}
          nTest={trainResult.n_test}
        />

        {isCV && (
          <KFoldResults
            title="K-fold CV scores"
            foldScores={trainResult.fold_scores}
            metricName={trainResult.metric_name}
            meanScore={trainResult.mean_score}
            stdScore={trainResult.std_score}
          />
        )}

        {isClassification && (
          <>
            <ClassificationResultsPanel trainResult={trainResult} />
            <DecoderOutputsResults trainResult={trainResult} />
          </>
        )}

        {isRegression && !isClassification && (
          <>
            <RegressionResultsPanel trainResult={trainResult} />
            <DecoderOutputsResults trainResult={trainResult} />
          </>
        )}

        {Array.isArray(trainResult.shuffled_scores) &&
          trainResult.shuffled_scores.length > 0 && (
            <BaselineShufflingResults
              metricName={trainResult.metric_name}
              referenceLabel={isCV ? 'real mean' : 'real'}
              referenceValue={
                isCV ? trainResult.mean_score : trainResult.metric_value
              }
              shuffledScores={trainResult.shuffled_scores}
              pValue={trainResult.p_value}
            />
          )}

        {Array.isArray(trainResult.notes) && trainResult.notes.length > 0 && (
          <>
            <Text fw={500} size="sm" mt="sm">
              Notes
            </Text>
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
