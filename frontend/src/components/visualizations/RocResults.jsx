import { Card, Stack, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

export default function RocResults({ roc }) {
  if (!roc || !Array.isArray(roc.curves) || roc.curves.length === 0) {
    return null;
  }

  const { kind, curves, macro_auc } = roc;

  const traces = curves.map((curve, idx) => {
    const label = curve.label != null ? String(curve.label) : 'ROC';
    const auc = typeof curve.auc === 'number'
      ? curve.auc.toFixed(3)
      : '—';

    return {
      x: curve.fpr,
      y: curve.tpr,
      type: 'scatter',
      mode: 'lines',
      name: `${label} (AUC=${auc})`,
      line: {
        width: 2,
      },
    };
  });

  // Diagonal baseline
  traces.push({
    x: [0, 1],
    y: [0, 1],
    type: 'scatter',
    mode: 'lines',
    name: 'Chance',
    line: {
      dash: 'dash',
      width: 1,
    },
    hoverinfo: 'none',
  });

  const titleSuffix =
    kind === 'multiclass' && typeof macro_auc === 'number'
      ? ` (macro AUC = ${macro_auc.toFixed(3)})`
      : '';

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="xs">
        <Text fw={500} size="sm">
          ROC curve{titleSuffix}
        </Text>
        <Plot
          data={traces}
          layout={{
            margin: { l: 50, r: 10, t: 30, b: 40 },
            xaxis: {
              title: 'False positive rate',
              range: [0, 1],
            },
            yaxis: {
              title: 'True positive rate',
              range: [0, 1],
            },
            legend: {
              x: 1,
              y: 0,
              xanchor: 'right',
              yanchor: 'bottom',
            },
            hovermode: 'closest',
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: '100%', maxWidth: 520, height: 360 }}
        />
      </Stack>
    </Card>
  );
}
