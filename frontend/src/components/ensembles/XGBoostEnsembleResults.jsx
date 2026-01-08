import {
  Box,
  Card,
  Divider,
  Group,
  Stack,
  Table,
  Text,
  Tooltip,
} from '@mantine/core';
import Plot from 'react-plotly.js';

// Confusion-matrix-inspired blue ramp
function cmBlue(t) {
  const tt = Math.max(0, Math.min(1, Number(t) || 0));
  const lightness = 100 - 55 * tt; // 100% -> 45%
  return `hsl(210, 80%, ${lightness}%)`;
}

function safeNum(x) {
  if (x === null || x === undefined || x === '') return null;
  const n = Number(x);
  return Number.isFinite(n) ? n : null;
}

function fmt(x, digits = 3) {
  const n = safeNum(x);
  if (n == null) return 'N/A';
  return Number.isInteger(n) ? String(n) : n.toFixed(digits);
}

function titleCase(s) {
  return String(s || '')
    .replace(/_/g, ' ')
    .trim()
    .split(/\s+/)
    .map((w) => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ');
}

function pickTopKeys(obj, maxKeys = 3) {
  if (!obj || typeof obj !== 'object') return [];
  const keys = Object.keys(obj);
  // Prefer validation metrics over training if both exist
  const score = (k) => {
    const kk = k.toLowerCase();
    if (kk.includes('validation_1')) return 3;
    if (kk.includes('validation_0')) return 2;
    if (kk.includes('train')) return 1;
    return 0;
  };
  return keys
    .sort((a, b) => score(b) - score(a) || a.localeCompare(b))
    .slice(0, maxKeys);
}

function prettyCurveName(key) {
  const k = String(key || '');

  // Expected patterns from backend aggregation (e.g. "validation_0:mlogloss")
  const m = k.match(/^validation_(\d+):(.+)$/);
  if (!m) return k;

  const idx = Number(m[1]);
  const metric = m[2];

  let setLabel = `Eval set ${idx}`;
  if (idx === 0) setLabel = 'Train (internal)';
  if (idx === 1) setLabel = 'Validation (internal)';

  return `${setLabel} • ${metric}`;
}

export default function XGBoostEnsembleResults({ report }) {
  if (!report || report.kind !== 'xgboost') return null;

  const metricName = report.metric_name || '';
  const trainEvalMetric = report.train_eval_metric || '';

  const xgb = report.xgboost || {};
  const params = xgb.params || {};
  const curves = report.learning_curves || null;
  const feat = report.feature_importance || null;

  const bestIterMean = safeNum(xgb.best_iteration_mean);
  const bestIterStd = safeNum(xgb.best_iteration_std);
  const bestScoreMean = safeNum(xgb.best_score_mean);
  const bestScoreStd = safeNum(xgb.best_score_std);

  const summaryItems = [
    {
      label: 'Best iteration (mean)',
      value: bestIterMean == null ? 'N/A' : fmt(bestIterMean, 1),
      tooltip:
        'Average best boosting round (only available when early stopping / eval sets are used).',
    },
    {
      label: 'Best iteration (std)',
      value: bestIterStd == null ? 'N/A' : fmt(bestIterStd, 1),
      tooltip: 'Standard deviation of best iteration across folds.',
    },
    {
      label: 'Best score (mean)',
      value: bestScoreMean == null ? 'N/A' : fmt(bestScoreMean, 5),
      tooltip:
        'Average best evaluation score reported by XGBoost during training (depends on eval metric).',
    },
    {
      label: 'Best score (std)',
      value: bestScoreStd == null ? 'N/A' : fmt(bestScoreStd, 5),
      tooltip: 'Standard deviation of best score across folds.',
    },
  ];

  const rows = [
    [summaryItems[0], summaryItems[1]],
    [summaryItems[2], summaryItems[3]],
  ];

  // Learning curves: show up to 3 series
  const curveKeys = pickTopKeys(curves, 3);

  const curveTraces = [];
  if (curves && curveKeys.length > 0) {
    curveKeys.forEach((k, idx) => {
      const item = curves[k];
      const mean = Array.isArray(item?.mean) ? item.mean : null;
      const std = Array.isArray(item?.std) ? item.std : null;
      if (!mean || mean.length === 0) return;

      const x = Array.from({ length: mean.length }, (_, i) => i + 1);
    const prettyName = prettyCurveName(k);
      curveTraces.push({
        type: 'scatter',
        mode: 'lines',
        name: prettyName,
        x,
        y: mean,
        line: { width: 2 },
        hovertemplate: `round: %{x}<br>${prettyName}: %{y:.5f}<extra></extra>`,
      });

      // Optional std band
      if (std && std.length === mean.length) {
        const upper = mean.map((v, i) => v + std[i]);
        const lower = mean.map((v, i) => v - std[i]);

        curveTraces.push({
          type: 'scatter',
          mode: 'lines',
          name: `${prettyName} +std`,
          x,
          y: upper,
          line: { width: 0 },
          showlegend: false,
          hoverinfo: 'skip',
        });
        curveTraces.push({
          type: 'scatter',
          mode: 'lines',
          name: `${prettyName} -std`,
          x,
          y: lower,
          line: { width: 0 },
          fill: 'tonexty',
          fillcolor: 'rgba(0,0,0,0.08)',
          showlegend: false,
          hoverinfo: 'skip',
        });
      }
    });
  }

  // Feature importance (top-K)
  const topFeatures = Array.isArray(feat?.top_features) ? feat.top_features : [];
  const featNames = topFeatures.map((d) => String(d.name));
  const featVals = topFeatures.map((d) => Number(d.importance) || 0);

  const featTrace =
    topFeatures.length > 0
      ? {
          type: 'bar',
          orientation: 'h',
          y: featNames.slice().reverse(),
          x: featVals.slice().reverse(),
          marker: {
            color: featVals.slice().reverse().map((_, i) => cmBlue(0.55 + 0.35 * (i / Math.max(1, featVals.length - 1)))),
          },
          hovertemplate: '%{y}<br>importance: %{x:.5f}<extra></extra>',
        }
      : null;

  const usedParams = [
    'n_estimators',
    'max_depth',
    'learning_rate',
    'subsample',
    'colsample_bytree',
    'reg_alpha',
    'reg_lambda',
    'gamma',
    'min_child_weight',
  ]
    .filter((k) => params && Object.prototype.hasOwnProperty.call(params, k))
    .map((k) => `${k}=${params[k]}`)
    .slice(0, 6);

  return (
    <Card withBorder radius="md" p="md">
      <Stack gap="xs">
        <Text fw={650} size="xl" align="center">
          XGBoost ensemble insights
        </Text>

        <Box>
          <Text fw={500} size="lg" align="center">
            Summary metrics
          </Text>

          <Table
            withTableBorder={false}
            withColumnBorders={false}
            horizontalSpacing="xs"
            verticalSpacing="xs"
            mt="xs"
          >
            <Table.Tbody>
              {rows.map((pair, i) => (
                <Table.Tr
                  key={i}
                  style={{
                    backgroundColor:
                      i % 2 === 1 ? 'var(--mantine-color-gray-0)' : 'white',
                  }}
                >
                  <Table.Td style={{ width: '25%' }}>
                    <Tooltip label={pair[0].tooltip} multiline maw={360} withArrow>
                      <Text size="sm" fw={600}>
                        {pair[0].label}
                      </Text>
                    </Tooltip>
                  </Table.Td>
                  <Table.Td style={{ width: '25%' }}>
                    <Text size="sm" fw={700}>
                      {pair[0].value}
                    </Text>
                  </Table.Td>

                  <Table.Td style={{ width: '25%' }}>
                    <Tooltip label={pair[1].tooltip} multiline maw={360} withArrow>
                      <Text size="sm" fw={600}>
                        {pair[1].label}
                      </Text>
                    </Tooltip>
                  </Table.Td>
                  <Table.Td style={{ width: '25%' }}>
                    <Text size="sm" fw={700}>
                      {pair[1].value}
                    </Text>
                  </Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>

          <Text size="sm" c="dimmed" mt="xs" align="center">
            <b>Final metric:</b> {metricName || 'N/A'} •{' '}
            <b>Training eval metric:</b> {trainEvalMetric || 'N/A'}
            {usedParams.length > 0 ? (
                <>
                {' '}
                • <b>Params:</b> {usedParams.join(' • ')}
                </>
            ) : null}
            </Text>
        </Box>

        <Divider my="xs" />

        <Box style={{ overflowX: 'auto' }}>
        <Group align="stretch" wrap="nowrap" gap="md" style={{ flexWrap: 'nowrap' }}>
          <Box style={{ flex: '1 1 0', minWidth: 250 }}>
            <Tooltip
              label="Mean learning curves aggregated across folds (aligned to shortest curve). Shaded area indicates ±1 std."
              multiline
              maw={420}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Learning curves
              </Text>
            </Tooltip>

            {curveTraces.length > 0 ? (
              <Plot
                data={curveTraces}
                layout={{
                  autosize: true,
                  height: 360,
                  margin: { l: 60, r: 12, t: 10, b: 60 },
                  xaxis: {
                    title: { text: 'Boosting round' },
                    automargin: true,
                    showgrid: false,
                    zeroline: false,
                  },
                  yaxis: {
                    title: { text: 'Metric' },
                    automargin: true,
                    showgrid: true,
                    zeroline: false,
                  },
                  legend: { orientation: 'h', x: 0.5, xanchor: 'center', y: -0.2 },
                  plot_bgcolor: '#ffffff',
                  paper_bgcolor: '#ffffff',
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            ) : (
              <Text size="sm" c="dimmed" align="center">
                No learning curves available (evals_result not provided by backend).
              </Text>
            )}
          </Box>
          

          <Box style={{ flex: '1 1 0', minWidth: 250 }}>
            <Tooltip
              label="Top feature importances aggregated across folds (normalized)."
              multiline
              maw={420}
              withArrow
            >
              <Text size="lg" fw={500} align="center" mb={6}>
                Feature importance
              </Text>
            </Tooltip>

            {featTrace ? (
              <Plot
                data={[featTrace]}
                layout={{
                  autosize: true,
                  height: 360,
                  margin: { l: 60, r: 12, t: 10, b: 50 },
                  xaxis: {
                    title: { text: 'Importance (normalized)' },
                    automargin: true,
                    showgrid: true,
                    zeroline: false,
                  },
                  yaxis: {
                    automargin: true,
                    showgrid: false,
                    zeroline: false,
                  },
                  plot_bgcolor: '#ffffff',
                  paper_bgcolor: '#ffffff',
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: '100%' }}
              />
            ) : (
              <Text size="sm" c="dimmed" align="center">
                No feature importance available (backend not provided yet).
              </Text>
            )}
          </Box>
        </Group>
        </Box>
      </Stack>
    </Card>
  );
}
