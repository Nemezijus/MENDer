import { Stack, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

export default function RocResults({ roc }) {
  if (!roc || !Array.isArray(roc.curves) || roc.curves.length === 0) {
    return null;
  }

  const { kind, curves, macro_auc } = roc;

  // Yosemite-ish earthy palette (olive, slate, green, peach, brown, mustard)
  const baseColors = [
    '#b3a369', // olive
    '#5c6f82', // slate blue/grey
    '#637939', // deep green
    '#f2a65a', // peach / warm orange
    '#8c4c2e', // brown
    '#d2b04c', // mustard
  ];

  const traces = curves.map((curve, idx) => {
    const rawLabel = curve.label != null ? String(curve.label) : 'ROC';
    const label = rawLabel.toString();
    const auc =
      typeof curve.auc === 'number' ? curve.auc.toFixed(3) : '—';

    const lower = label.toLowerCase();
    const isMicro =
      lower === 'micro' || lower === 'micro avg' || lower === 'micro_avg';
    const isMacro =
      lower === 'macro' || lower === 'macro avg' || lower === 'macro_avg';

    const color =
      isMicro || isMacro
        ? '#000000'
        : baseColors[idx % baseColors.length];

    const lineStyle = {
      color,
      width: isMicro || isMacro ? 3 : 2,
      dash: isMicro || isMacro ? 'dot' : 'solid',
    };

    return {
      x: curve.fpr,
      y: curve.tpr,
      type: 'scatter',
      mode: 'lines',
      name: `${label} (AUC=${auc})`,
      line: lineStyle,
    };
  });

  // Diagonal baseline
  traces.push({
    x: [0, 1.05],
    y: [0, 1.05],
    type: 'scatter',
    mode: 'lines',
    name: 'Chance',
    line: {
      dash: 'dash',
      width: 1,
      color: 'rgba(150,150,150,0.8)',
    },
    hoverinfo: 'none',
  });

  const macroLabel = kind === 'multiclass' ? 'Macro AUC' : 'AUC';
  const macroText =
    typeof macro_auc === 'number'
      ? `${macroLabel}: `
      : null;
  const macroVal =
    typeof macro_auc === 'number'
      ? macro_auc.toFixed(3)
      : null;

  return (
    <Stack gap="xs">
      <Text fw={500} size="xl" align="center">
        ROC Curve
      </Text>

      <div
        style={{
          width: '100%',
          maxWidth: 520,
          aspectRatio: '1 / 1',
          margin: '0 auto',
        }}
      >
        <Plot
          data={traces}
          layout={{
            margin: { l: 70, r: 40, t: 20, b: 70 },
            xaxis: {
              title: {
                text: 'False positive rate (fall-out)',
                font: { size: 16, weight: 'bold' },
              },
              range: [0, 1.05],
              tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
              ticktext: ['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.4)',
              zeroline: false,
              showline: true,
              linewidth: 2,
              linecolor: '#000',
              mirror: false,
              constrain: 'domain',
              scaleanchor: 'y',
              scaleratio: 1,
            },
            yaxis: {
              title: {
                text: 'True positive rate (sensitivity)',
                font: { size: 16, weight: 'bold' },
              },
              range: [0, 1.05],
              tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1.0],
              ticktext: ['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.4)',
              zeroline: false,
              showline: true,
              linewidth: 2,
              linecolor: '#000',
              mirror: false,
              constrain: 'domain',
            },
            // Thin bounding box around ROC area
            shapes: [
              {
                type: 'rect',
                xref: 'x',
                yref: 'y',
                x0: 0,
                x1: 1.05,
                y0: 0,
                y1: 1.05,
                line: {
                  color: '#555',
                  width: 1,
                },
                fillcolor: 'rgba(0,0,0,0)',
              },
            ],
            legend: {
              x: 0.97,
              y: 0.03,
              xanchor: 'right',
              yanchor: 'bottom',
              borderwidth: 0,
              bgcolor: 'rgba(255,255,255,0.7)',
            },
            hovermode: 'closest',
            plot_bgcolor: '#ffffff',
            paper_bgcolor: 'rgba(0,0,0,0)',
          }}
          config={{
            displayModeBar: false,
            responsive: true,
            useResizeHandler: true,
          }}
          style={{ width: '100%', height: '100%' }}
        />
      </div>

      {macroText && macroVal && (
        <Text size="sm">
          {macroText}
          <Text span fw={700}>
            {macroVal}
          </Text>
        </Text>
      )}
    </Stack>
  );
}
