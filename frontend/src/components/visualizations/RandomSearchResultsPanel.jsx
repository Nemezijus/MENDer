// frontend/src/components/visualizations/RandomSearchResultsPanel.jsx
import { useMemo } from 'react';
import { Card, Stack, Text, useMantineTheme } from '@mantine/core';
import { useSettingsStore } from '../../state/useSettingsStore';
import RandomSearchResults from './RandomSearchResults.jsx';
import RandomSearchAnalyticsResults from './RandomSearchAnalyticsResults.jsx';

export default function RandomSearchResultsPanel({ result }) {
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
    const loose = cv.mean_test_score || cv.mean_score || cv.mean_train_score || cv.mean_train_scores || null;
    if (Array.isArray(loose)) return { scores: loose, source: 'unknown' };
    return { scores: null, source: null };
  }

  const rsData = useMemo(() => {
    if (!result) return null;
    const cv = result.cv_results;
    if (!cv) return null;

    const paramKeys = Object.keys(cv).filter((k) => k.startsWith('param_'));
    if (paramKeys.length < 2) return null;

    const p1Key = paramKeys[0];
    const p2Key = paramKeys[1];

    const pipelineName1 = p1Key.slice('param_'.length);
    const pipelineName2 = p2Key.slice('param_'.length);

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

    const bestIndex =
      typeof result.best_index === 'number' ? result.best_index : null;

    const bestPoint =
      bestIndex != null && bestIndex >= 0 && bestIndex < vals1.length
        ? { index: bestIndex }
        : null;

    return {
      param1Name: shortName1,
      param2Name: shortName2,
      param1Samples: vals1,
      param2Samples: vals2,
      scores,
      bestPoint,
      pipelineName1,
      pipelineName2,
      shortName1,
      shortName2,
      scoreSource: source,
    };
  }, [result]);

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={500}>Randomized search results</Text>

        {!result && (
          <Text size="sm" c="dimmed">
            Run a randomized search to see results here.
          </Text>
        )}

        {result && !rsData && (
          <Text size="sm" c="red">
            Randomized search result does not contain parameter information that
            can be plotted.
          </Text>
        )}

        {result && rsData && (
          <>
            {result.note && (
              <Text size="sm" c="dimmed">
                {result.note}
              </Text>
            )}
            {!result.note && rsData.scoreSource === 'train' && (
              <Text size="sm" c="dimmed">
                Validation scores were not available; this plot reflects train-side scores.
              </Text>
            )}
            <RandomSearchResults
              param1Name={rsData.param1Name}
              param2Name={rsData.param2Name}
              param1Samples={rsData.param1Samples}
              param2Samples={rsData.param2Samples}
              scores={rsData.scores}
              bestPoint={rsData.bestPoint}
              textColor={textColor}
              gridColor={gridColor}
              axisColor={axisColor}
              metricLabel={metricLabel}
            />
            <RandomSearchAnalyticsResults
              result={result}
              metricLabel={metricLabel}
              pipelineName1={rsData.pipelineName1}
              pipelineName2={rsData.pipelineName2}
              shortName1={rsData.shortName1}
              shortName2={rsData.shortName2}
            />
          </>
        )}
      </Stack>
    </Card>
  );
}
