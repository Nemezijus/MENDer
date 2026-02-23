import { Stack, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

export default function RocResults({ roc }) {
  if (!roc || !Array.isArray(roc.curves) || roc.curves.length === 0) {
    return null;
  }
  const { kind, curves, macro_auc, micro_auc } = roc;

  // New palette
const baseColors = [
  '#2a9d8f',
  '#f4a261',
  '#9576c9',
  '#e36040',
  '#287271',
  '#8ab17d',
  '#bc6b85',
  '#264653',
  '#e9c46a',
  '#ec8151',
];

  const MICRO_COLOR = '#7209b7';
  const MACRO_COLOR = '#4361ee';

  // Slightly darken the color for subsequent cycles over the palette
  const adjustColor = (hex, repeatIndex) => {
    if (!repeatIndex) return hex;

    const factor = Math.pow(0.9, repeatIndex); // darker each time
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);

    const rf = Math.max(0, Math.min(255, Math.round(r * factor)));
    const gf = Math.max(0, Math.min(255, Math.round(g * factor)));
    const bf = Math.max(0, Math.min(255, Math.round(b * factor)));

    const toHex = (v) => v.toString(16).padStart(2, '0');
    return `#${toHex(rf)}${toHex(gf)}${toHex(bf)}`;
  };

  const getPaletteColor = (idx) => {
    const baseIndex = idx % baseColors.length;
    const repeatIndex = Math.floor(idx / baseColors.length);
    const base = baseColors[baseIndex];
    return adjustColor(base, repeatIndex);
  };

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

    const color = isMicro
      ? MICRO_COLOR
      : isMacro
      ? MACRO_COLOR
      : getPaletteColor(idx);

    const lineStyle = {
      color,
      width: isMicro || isMacro ? 3 : 2, // one size thicker than normal curves
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

  const microVal =
    typeof micro_auc === 'number' ? micro_auc.toFixed(3) : null;

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
              tickfont: { size: 16 },
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.4)',
              zeroline: false,
              showline: true,
              linewidth: 1, // normal thickness
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
              tickfont: { size: 16 },
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.4)',
              zeroline: false,
              showline: true,
              linewidth: 1, // normal thickness
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

    {roc.kind === "binary" && roc.positive_label !== undefined && (
      <Text size="xs" c="dimmed">
        This ROC curve treats class <Text span fw={600}>{roc.positive_label} </Text> 
        as the reference (positive) class. The curve shows how well the model 
        distinguishes that class from the other one across thresholds.
      </Text>
    )}
    {(macroText && macroVal) || microVal ? (
      <Stack gap={2}>
        {macroText && macroVal && (
          <Text size="sm">
            {macroText}
            <Text span fw={700}>
              {macroVal}
            </Text>
          </Text>
        )}

        {microVal && (
          <Text size="sm">
            Micro AUC:{' '}
            <Text span fw={700}>
              {microVal}
            </Text>
          </Text>
        )}

        {/* Show explanation only when Micro AUC is present (i.e. multiclass) */}
        {microVal && (
          <Text span size="xs" c="dimmed">
            Macro AUC treats all classes equally, so it’s more informative when you
            want to compare performance across classes—especially if some classes
            are rare. Micro AUC looks at every prediction together and reflects
            overall performance weighted by how often each class appears.
          </Text>
        )}
      </Stack>
    ) : null}
    </Stack>
  );
}
