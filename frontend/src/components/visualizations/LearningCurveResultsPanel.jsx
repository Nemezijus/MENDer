import { useMemo } from 'react';
import { Stack, Text, Card, useMantineTheme } from '@mantine/core';
import { useResultsStore } from '../../state/useResultsStore.js';
import { useSettingsStore } from '../../state/useSettingsStore';
import LearningCurveResults from './LearningCurveResults.jsx';
import LearningCurveAnalyticsResults from './LearningCurveAnalyticsResults.jsx';

function hasFiniteNumbers(arr) {
  return Array.isArray(arr) && arr.some((v) => typeof v === 'number' && Number.isFinite(v));
}

export default function LearningCurveResultsPanel() {
  const theme = useMantineTheme();
  const isDark = theme.colorScheme === 'dark';
  const textColor = isDark ? theme.colors.gray[2] : theme.black;
  const gridColor = isDark ? theme.colors.dark[4] : '#e0e0e0';
  const axisColor = isDark ? theme.colors.dark[2] : '#222';

  const learningCurveResult = useResultsStore((s) => s.learningCurveResult);
  const learningCurveNSplits = useResultsStore((s) => s.learningCurveNSplits);
  const learningCurveWithinPct = useResultsStore((s) => s.learningCurveWithinPct);

  const metricFromSettings = useSettingsStore((s) => s.metric);
  const metricFromSettingsScalar = Array.isArray(metricFromSettings)
    ? metricFromSettings[0]
    : metricFromSettings;
  const metricLabel =
    (learningCurveResult && learningCurveResult.metric_used) ||
    metricFromSettingsScalar ||
    'Accuracy';

  const analytics = useMemo(() => {
    if (!learningCurveResult) return null;
    const xs = learningCurveResult.train_sizes;
    const trainMean = learningCurveResult.train_scores_mean;
    const trainStd  = learningCurveResult.train_scores_std;
    const valMean   = learningCurveResult.val_scores_mean;
    const valStd    = learningCurveResult.val_scores_std;

    const hasValidation = hasFiniteNumbers(valMean) && hasFiniteNumbers(valStd);

    const n = Math.max(1, Number(learningCurveNSplits));
    const trainSEM = trainStd.map((s) => s / Math.sqrt(n));
    const valSEM   = hasValidation ? valStd.map((s) => s / Math.sqrt(n)) : null;

    // If validation is unavailable (common for clustering models without predict()),
    // we still return train-side analytics so the plot can render.
    if (!hasValidation) {
      return {
        xs,
        trainMean,
        valMean: null,
        trainSEM,
        valSEM: null,
        best: null,
        minimal: null,
        hasValidation: false,
      };
    }

    // Find best validation index among finite values
    let bestIdx = -1;
    for (let i = 0; i < valMean.length; i++) {
      const v = valMean[i];
      if (typeof v !== 'number' || !Number.isFinite(v)) continue;
      if (bestIdx === -1 || v > valMean[bestIdx]) bestIdx = i;
    }

    if (bestIdx === -1) {
      return {
        xs,
        trainMean,
        valMean,
        trainSEM,
        valSEM,
        best: null,
        minimal: null,
        hasValidation: false,
      };
    }

    const best = {
      size: xs[bestIdx],
      val: valMean[bestIdx],
      train: trainMean[bestIdx],
      idx: bestIdx,
    };

    const cutoff = learningCurveWithinPct * best.val;
    let minimalIdx = bestIdx;
    for (let i = 0; i < valMean.length; i++) {
      const v = valMean[i];
      if (typeof v !== 'number' || !Number.isFinite(v)) continue;
      if (v >= cutoff) { minimalIdx = i; break; }
    }

    const minimal = {
      size: xs[minimalIdx],
      val: valMean[minimalIdx],
      train: trainMean[minimalIdx],
      idx: minimalIdx,
      cutoff,
    };

    return { xs, trainMean, valMean, trainSEM, valSEM, best, minimal, hasValidation: true };
  }, [learningCurveResult, learningCurveNSplits, learningCurveWithinPct]);

  const plotTraces = useMemo(() => {
    if (!analytics) return [];

    const { xs, trainMean, valMean, trainSEM, valSEM, minimal, hasValidation } = analytics;

    const lower = (arr, sem) => arr.map((v, i) => v - sem[i]);
    const upper = (arr, sem) => arr.map((v, i) => v + sem[i]);

    const yCandidates = [];
    for (let i = 0; i < trainMean.length; i++) {
      if (typeof trainMean[i] === 'number' && Number.isFinite(trainMean[i])) {
        yCandidates.push(trainMean[i] - trainSEM[i]);
        yCandidates.push(trainMean[i] + trainSEM[i]);
      }
    }

    const trainLower = {
      x: xs,
      y: lower(trainMean, trainSEM),
      name: 'Train (−SEM)',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const trainUpper = {
      x: xs,
      y: upper(trainMean, trainSEM),
      name: 'Train (SEM area)',
      fill: 'tonexty',
      line: { width: 0 },
      hoverinfo: 'skip',
      showlegend: false,
      type: 'scatter',
      mode: 'lines',
    };
    const trainLine = {
      x: xs,
      y: trainMean,
      name: 'Train (mean)',
      type: 'scatter',
      mode: 'lines+markers',
      hovertemplate: `Train size: %{x}<br>Train ${metricLabel}: %{y:.3f}<extra></extra>`,
    };

    const traces = [trainLower, trainUpper, trainLine];

    if (hasValidation && Array.isArray(valMean) && Array.isArray(valSEM)) {
      for (let i = 0; i < valMean.length; i++) {
        const v = valMean[i];
        const s = valSEM[i];
        if (typeof v === 'number' && Number.isFinite(v) && typeof s === 'number' && Number.isFinite(s)) {
          yCandidates.push(v - s);
          yCandidates.push(v + s);
        }
      }

      const valLower = {
        x: xs,
        y: lower(valMean, valSEM),
        name: 'Validation (−SEM)',
        line: { width: 0 },
        hoverinfo: 'skip',
        showlegend: false,
        type: 'scatter',
        mode: 'lines',
      };
      const valUpper = {
        x: xs,
        y: upper(valMean, valSEM),
        name: 'Validation (SEM area)',
        fill: 'tonexty',
        line: { width: 0 },
        hoverinfo: 'skip',
        showlegend: false,
        type: 'scatter',
        mode: 'lines',
      };
      const valLine = {
        x: xs,
        y: valMean,
        name: 'Validation (mean)',
        type: 'scatter',
        mode: 'lines+markers',
        hovertemplate: `Train size: %{x}<br>Val ${metricLabel}: %{y:.3f}<extra></extra>`,
      };

      traces.push(valLower, valUpper, valLine);
    }

    if (minimal && typeof minimal.size !== 'undefined') {
      const yMin = yCandidates.length ? Math.min(...yCandidates) : 0;
      const yMax = yCandidates.length ? Math.max(...yCandidates) : 1;
      const vLine = {
        x: [minimal.size, minimal.size],
        y: [yMin, yMax],
        name: 'Recommended size',
        mode: 'lines',
        line: { dash: 'dash' },
        hoverinfo: 'skip',
        showlegend: true,
      };
      traces.push(vLine);
    }

    return traces;
  }, [analytics, metricLabel]);

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500}>Learning Curve Results</Text>

        {!learningCurveResult && (
          <Text size="sm" c="dimmed">
            Compute a learning curve to see results here.
          </Text>
        )}

        {learningCurveResult && (
          <>
            {learningCurveResult.note && (
              <Text size="sm" c="dimmed">
                {learningCurveResult.note}
              </Text>
            )}
            <LearningCurveResults
              plotTraces={plotTraces}
              textColor={textColor}
              gridColor={gridColor}
              axisColor={axisColor}
              metricLabel={metricLabel}
            />
            <LearningCurveAnalyticsResults
              analytics={analytics}
              withinPct={learningCurveWithinPct}
            />
          </>
        )}
      </Stack>
    </Card>
  );
}

