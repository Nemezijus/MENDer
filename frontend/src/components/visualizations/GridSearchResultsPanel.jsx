// frontend/src/components/visualizations/GridSearchResultsPanel.jsx
import { useMemo } from 'react';
import { Card, Stack, Text, useMantineTheme } from '@mantine/core';
import { useSettingsStore } from '../../state/useSettingsStore.js';
import GridSearchResults from './GridSearchResults.jsx';
import GridSearchAnalyticsResults from './GridSearchAnalyticsResults.jsx';

// helper to make a stable key for mixed types (bool, str, number, null)
function makeKey(v) {
  if (v === null) return 'null';
  return `${typeof v}|${String(v)}`;
}

// unique values preserving order
function uniquePreserve(values) {
  const seen = new Set();
  const out = [];
  for (const v of values) {
    const k = makeKey(v);
    if (!seen.has(k)) {
      seen.add(k);
      out.push(v);
    }
  }
  return out;
}

export default function GridSearchResultsPanel({ result }) {
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

  function pickScoreArray(cv) {
    const candidates = [
      { key: 'mean_test_score', label: 'validation' },
      { key: 'mean_score', label: 'validation' },
      { key: 'mean_train_score', label: 'train' },
      { key: 'mean_train_scores', label: 'train' },
      { key: 'mean_test_scores', label: 'validation' },
    ];
    for (const c of candidates) {
      const arr = cv[c.key];
      if (Array.isArray(arr) && arr.some((v) => typeof v === 'number' && Number.isFinite(v))) {
        return { scores: arr, source: c.label };
      }
    }
    // Last resort: accept numeric-or-null arrays (still plottable)
    const loose = cv.mean_test_score || cv.mean_score || cv.mean_train_score || cv.mean_train_scores || null;
    if (Array.isArray(loose)) return { scores: loose, source: 'unknown' };
    return { scores: null, source: null };
  }

  const gridData = useMemo(() => {
    if (!result) return null;

    const cv = result.cv_results;
    if (!cv) return null;

    // Find the two hyperparameter keys from cv_results (param_...)
    const paramKeys = Object.keys(cv).filter((k) => k.startsWith('param_'));
    if (paramKeys.length < 2) return null; // nothing to plot

    const p1Key = paramKeys[0]; // e.g. "param_clf__C"
    const p2Key = paramKeys[1]; // e.g. "param_clf__max_depth"

    const pipelineName1 = p1Key.slice('param_'.length); // "clf__C"
    const pipelineName2 = p2Key.slice('param_'.length); // "clf__max_depth"

    const shortName1 = pipelineName1.split('__').slice(-1)[0] || pipelineName1;
    const shortName2 = pipelineName2.split('__').slice(-1)[0] || pipelineName2;

    const vals1 = cv[p1Key] || [];
    const vals2 = cv[p2Key] || [];
    const { scores, source } = pickScoreArray(cv);
    if (!scores) return null;

    if (
      !Array.isArray(vals1) ||
      !Array.isArray(vals2) ||
      !Array.isArray(scores) ||
      vals1.length !== vals2.length ||
      vals1.length !== scores.length
    ) {
      return null;
    }

    const param1Values = uniquePreserve(vals1);
    const param2Values = uniquePreserve(vals2);

    // Create index maps for quick lookups
    const idx1 = new Map(param1Values.map((v, i) => [makeKey(v), i]));
    const idx2 = new Map(param2Values.map((v, i) => [makeKey(v), i]));

    // Build z as [len(param2Values)][len(param1Values)]
    const z = param2Values.map(() => param1Values.map(() => null));

    for (let i = 0; i < scores.length; i += 1) {
      const v1 = vals1[i];
      const v2 = vals2[i];
      const s = scores[i];
      const k1 = makeKey(v1);
      const k2 = makeKey(v2);
      const j = idx1.get(k1); // x index
      const k = idx2.get(k2); // y index
      if (j != null && k != null) {
        z[k][j] = s;
      }
    }

    // Best point position (for marker)
    let bestPoint = null;
    if (typeof result.best_index === 'number') {
      const bi = result.best_index;
      if (bi >= 0 && bi < vals1.length) {
        const v1 = vals1[bi];
        const v2 = vals2[bi];
        bestPoint = { x: v1, y: v2 };
      }
    }

    return {
      param1Name: shortName1,
      param2Name: shortName2,
      param1Values,
      param2Values,
      meanScores: z,
      bestPoint,
      pipelineName1,
      pipelineName2,
      scoreSource: source,
    };
  }, [result]);

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500}>Grid search results</Text>

        {!result && (
          <Text size="sm" c="dimmed">
            Run a grid search to see results here.
          </Text>
        )}

        {result && !gridData && (
          <Text size="sm" c="red">
            Grid search result does not contain a 2D parameter grid that can be
            plotted.
          </Text>
        )}

        {result && gridData && (
          <>
            {result.note && (
              <Text size="sm" c="dimmed">
                {result.note}
              </Text>
            )}
            {!result.note && gridData.scoreSource === 'train' && (
              <Text size="sm" c="dimmed">
                Validation scores were not available; this plot reflects train-side scores.
              </Text>
            )}
            <GridSearchResults
              param1Name={gridData.param1Name}
              param2Name={gridData.param2Name}
              param1Values={gridData.param1Values}
              param2Values={gridData.param2Values}
              meanScores={gridData.meanScores}
              bestPoint={gridData.bestPoint}
              textColor={textColor}
              gridColor={gridColor}
              axisColor={axisColor}
              metricLabel={metricLabel}
            />
            <GridSearchAnalyticsResults
              result={result}
              metricLabel={metricLabel}
              pipelineName1={gridData.pipelineName1}
              pipelineName2={gridData.pipelineName2}
              shortName1={gridData.param1Name}
              shortName2={gridData.param2Name}
            />
          </>
        )}
      </Stack>
    </Card>
  );
}
