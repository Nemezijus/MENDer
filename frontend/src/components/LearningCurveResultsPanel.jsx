import { useMemo } from 'react';
import { Stack, Text, Card, useMantineTheme } from '@mantine/core';
import { useLearningCurveResultsCtx } from '../state/LearningCurveResultsContext.jsx';
import LearningCurveResults from './visualizations/LearningCurveResults.jsx';
import LearningCurveAnalyticsResults from './visualizations/LearningCurveAnalyticsResults.jsx';

export default function LearningCurveResultsPanel() {
  const theme = useMantineTheme();
  const isDark = theme.colorScheme === 'dark';
  const textColor = isDark ? theme.colors.gray[2] : theme.black;
  const gridColor = isDark ? theme.colors.dark[4] : '#e0e0e0';
  const axisColor = isDark ? theme.colors.dark[2] : '#222';

  const { result, nSplits, withinPct } = useLearningCurveResultsCtx();

  const analytics = useMemo(() => {
    if (!result) return null;
    const xs = result.train_sizes;
    const trainMean = result.train_scores_mean;
    const trainStd  = result.train_scores_std;
    const valMean   = result.val_scores_mean;
    const valStd    = result.val_scores_std;

    const n = Math.max(1, Number(nSplits));
    const trainSEM = trainStd.map(s => s / Math.sqrt(n));
    const valSEM   = valStd.map(s => s / Math.sqrt(n));

    let bestIdx = 0;
    for (let i = 1; i < valMean.length; i++) {
      if (valMean[i] > valMean[bestIdx]) bestIdx = i;
    }
    const best = {
      size: xs[bestIdx],
      val: valMean[bestIdx],
      train: trainMean[bestIdx],
      idx: bestIdx,
    };

    const cutoff = withinPct * best.val;
    let minimalIdx = bestIdx;
    for (let i = 0; i < valMean.length; i++) {
      if (valMean[i] >= cutoff) { minimalIdx = i; break; }
    }
    const minimal = {
      size: xs[minimalIdx],
      val: valMean[minimalIdx],
      train: trainMean[minimalIdx],
      idx: minimalIdx,
      cutoff,
    };

    return { xs, trainMean, valMean, trainSEM, valSEM, best, minimal };
  }, [result, nSplits, withinPct]);

  const plotTraces = useMemo(() => {
    if (!analytics) return [];

    const { xs, trainMean, valMean, trainSEM, valSEM, minimal } = analytics;

    const lower = (arr, sem) => arr.map((v, i) => v - sem[i]);
    const upper = (arr, sem) => arr.map((v, i) => v + sem[i]);

    const trainLower = {
      x: xs, y: lower(trainMean, trainSEM),
      name: 'Train (−SEM)',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const trainUpper = {
      x: xs, y: upper(trainMean, trainSEM),
      name: 'Train (SEM area)',
      fill: 'tonexty',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const trainLine = {
      x: xs, y: trainMean,
      name: 'Train (mean)',
      type: 'scatter',
      mode: 'lines+markers',
      hovertemplate: 'Train size: %{x}<br>Train acc: %{y:.3f}<extra></extra>',
    };

    const valLower = {
      x: xs, y: lower(valMean, valSEM),
      name: 'Validation (−SEM)',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const valUpper = {
      x: xs, y: upper(valMean, valSEM),
      name: 'Validation (SEM area)',
      fill: 'tonexty',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const valLine = {
      x: xs, y: valMean,
      name: 'Validation (mean)',
      type: 'scatter',
      mode: 'lines+markers',
      hovertemplate: 'Train size: %{x}<br>Val acc: %{y:.3f}<extra></extra>',
    };

    const vLine = {
      x: [minimal.size, minimal.size],
      y: [0, 1],
      name: 'Recommended size',
      mode: 'lines',
      line: { dash: 'dash' },
      hoverinfo: 'skip',
      showlegend: true,
    };

    return [trainLower, trainUpper, trainLine, valLower, valUpper, valLine, vLine];
  }, [analytics]);

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500}>Learning Curve Results</Text>

        {!result && (
          <Text size="sm" c="dimmed">
            Compute a learning curve to see results here.
          </Text>
        )}

        {result && (
          <>
            <LearningCurveResults
              plotTraces={plotTraces}
              textColor={textColor}
              gridColor={gridColor}
              axisColor={axisColor}
            />
            <LearningCurveAnalyticsResults
              analytics={analytics}
              withinPct={withinPct}
            />
          </>
        )}
      </Stack>
    </Card>
  );
}
