import { useMemo } from 'react';
import { Card, Stack, Text, useMantineTheme } from '@mantine/core';
import { useSettingsStore } from '../../state/useSettingsStore';
import ValidationCurveResults from './ValidationCurveResults.jsx';
import ValidationCurveAnalyticsResults from './ValidationCurveAnalyticsResults.jsx';

export default function ValidationCurveResultsPanel({
  result,
  nSplits,
  withinPct = 0.99,
}) {
  const theme = useMantineTheme();
  const isDark = theme.colorScheme === 'dark';
  const textColor = isDark ? theme.colors.gray[2] : theme.black;
  const gridColor = isDark ? theme.colors.dark[4] : '#e0e0e0';
  const axisColor = isDark ? theme.colors.dark[2] : '#222';

  const metricFromSettings = useSettingsStore((s) => s.metric);
  const metricFromSettingsScalar = Array.isArray(metricFromSettings)
    ? metricFromSettings[0]
    : metricFromSettings;
  const metricLabel =
    (result && result.metric_used) || metricFromSettingsScalar || 'Metric';

  const analytics = useMemo(() => {
    if (!result) return null;

    const xs = result.param_range || [];
    const trainMean = result.train_scores_mean || [];
    const trainStd = result.train_scores_std || [];
    const valMeanRaw = result.val_scores_mean || null;
    const valStdRaw = result.val_scores_std || null;

    const hasValidation =
      Array.isArray(valMeanRaw) &&
      valMeanRaw.some((v) => typeof v === 'number' && Number.isFinite(v)) &&
      Array.isArray(valStdRaw) &&
      valStdRaw.some((v) => typeof v === 'number' && Number.isFinite(v));

    const n = Math.max(1, Number(nSplits || 1));
    const trainSEM = trainStd.map((s) =>
      s == null ? 0 : s / Math.sqrt(n)
    );
    const valMean = hasValidation ? valMeanRaw : null;
    const valSEM = hasValidation
      ? valStdRaw.map((s) => (s == null ? 0 : s / Math.sqrt(n)))
      : null;

    // If validation is unavailable (common for clustering models without predict()),
    // return train-only analytics.
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

    // Find best validation index among non-null / non-NaN values
    let bestIdx = -1;
    for (let i = 0; i < valMean.length; i++) {
      const v = valMean[i];
      if (v == null || Number.isNaN(v)) continue;
      if (bestIdx === -1 || v > valMean[bestIdx]) {
        bestIdx = i;
      }
    }

    // Fallback: if we never found a "best", but we have values, use index 0
    if (bestIdx === -1) {
      if (valMean.length === 0) return null;
      bestIdx = 0;
    }

    const best = {
      value: xs[bestIdx],
      val: valMean[bestIdx],
      train: trainMean[bestIdx],
      idx: bestIdx,
    };

    const cutoff = withinPct * best.val;
    let minimalIdx = bestIdx;
    for (let i = 0; i < valMean.length; i++) {
      const v = valMean[i];
      if (v == null || Number.isNaN(v)) continue;
      if (v >= cutoff) {
        minimalIdx = i;
        break;
      }
    }

    const minimal = {
      value: xs[minimalIdx],
      val: valMean[minimalIdx],
      train: trainMean[minimalIdx],
      idx: minimalIdx,
      cutoff,
    };

    return { xs, trainMean, valMean, trainSEM, valSEM, best, minimal, hasValidation: true };
  }, [result, nSplits, withinPct]);

  const plotTraces = useMemo(() => {
    if (!analytics) return [];

    const { xs, trainMean, valMean, trainSEM, valSEM, minimal, hasValidation } = analytics;

    const lower = (arr, sem) =>
      arr.map((v, i) =>
        v == null || sem[i] == null ? null : v - sem[i]
      );
    const upper = (arr, sem) =>
      arr.map((v, i) =>
        v == null || sem[i] == null ? null : v + sem[i]
      );

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
      hovertemplate:
        'Param: %{x}<br>Train ' + metricLabel + ': %{y:.3f}<extra></extra>',
    };

    const traces = [trainLower, trainUpper, trainLine];

    // Track a y-range for the recommendation line so it spans the visible plot.
    const yCandidates = [];
    for (let i = 0; i < trainMean.length; i++) {
      const v = trainMean[i];
      const s = trainSEM[i];
      if (typeof v === 'number' && Number.isFinite(v) && typeof s === 'number' && Number.isFinite(s)) {
        yCandidates.push(v - s, v + s);
      }
    }

    if (hasValidation && Array.isArray(valMean) && Array.isArray(valSEM)) {
      for (let i = 0; i < valMean.length; i++) {
        const v = valMean[i];
        const s = valSEM[i];
        if (typeof v === 'number' && Number.isFinite(v) && typeof s === 'number' && Number.isFinite(s)) {
          yCandidates.push(v - s, v + s);
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
      hovertemplate:
        'Param: %{x}<br>Val ' + metricLabel + ': %{y:.3f}<extra></extra>',
      };

      traces.push(valLower, valUpper, valLine);
    }

    if (minimal && typeof minimal.value !== 'undefined') {
      const yMin = yCandidates.length ? Math.min(...yCandidates) : 0;
      const yMax = yCandidates.length ? Math.max(...yCandidates) : 1;
      const vLine = {
        x: [minimal.value, minimal.value],
        y: [yMin, yMax],
        name: 'Recommended value',
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
        <Text fw={500}>Validation Curve Results</Text>

        {!result && (
          <Text size="sm" c="dimmed">
            Compute a validation curve to see results here.
          </Text>
        )}

        {result && analytics && (
          <>
            {result.note && (
              <Text size="sm" c="dimmed">
                {result.note}
              </Text>
            )}
            <ValidationCurveResults
              plotTraces={plotTraces}
              textColor={textColor}
              gridColor={gridColor}
              axisColor={axisColor}
              metricLabel={metricLabel}
              paramName={result.param_name}
            />
            <ValidationCurveAnalyticsResults
              analytics={analytics}
              withinPct={withinPct}
              metricLabel={metricLabel}
            />
          </>
        )}
      </Stack>
    </Card>
  );
}