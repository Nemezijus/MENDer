import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';

import { hexToRgba, toFiniteNumbers } from '../../../utils/stats.js';
import {
  AXIS_TICK,
  AXIS_TITLE,
  PLOT_BG,
  PLOTLY_COLORS,
  PLOT_MARGIN_STD,
} from '../../../utils/plotly.js';

import { PLOT_HEIGHT } from '../common/styles.js';

export default function SilhouetteSection({ silhouette, clusterLabel }) {
  if (!silhouette) return null;
  const clusterIds = silhouette?.cluster_ids;
  const groups = silhouette?.values;
  if (!Array.isArray(clusterIds) || !Array.isArray(groups) || clusterIds.length !== groups.length) return null;

  const avg = Number(silhouette?.avg);

  let yLower = 10;
  const traces = [];
  const hoverPoints = [];
  let globalMin = 1;

  clusterIds.forEach((cid, idx) => {
    const vals = toFiniteNumbers(groups[idx]).sort((a, b) => a - b);
    if (!vals.length) return;

    globalMin = Math.min(globalMin, vals[0]);

    const yUpper = yLower + vals.length;
    const ySeq = Array.from({ length: vals.length }, (_, i) => yLower + i);

    const color = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
    const fillcolor = hexToRgba(color, 0.25);

    // Filled polygon (mirrors sklearn's fill_betweenx between x=0 and silhouette values)
    // Build a simple, non-self-intersecting polygon even when values are negative.
    const zeros = new Array(vals.length).fill(0);
    const xPoly = [...zeros, ...vals.slice().reverse(), 0];
    const yPoly = [...ySeq, ...ySeq.slice().reverse(), ySeq[0]];

    traces.push({
      type: 'scatter',
      mode: 'lines',
      x: xPoly,
      y: yPoly,
      fill: 'toself',
      fillcolor,
      line: { width: 0, color },
      hoverinfo: 'skip',
      showlegend: false,
    });

    // Invisible markers to enable per-sample hover
    hoverPoints.push({
      type: 'scatter',
      mode: 'markers',
      x: vals,
      y: ySeq,
      marker: { size: 8, opacity: 0 },
      hovertemplate: `Cluster=${clusterLabel(cid)}<br>Silhouette=%{x:.3f}<extra></extra>`,
      showlegend: false,
    });

    // Cluster label (centered)
    traces.push({
      type: 'scatter',
      mode: 'text',
      x: [Math.max(0.02, Math.min(0.2, (Math.max(...vals) + Math.min(...vals)) / 2))],
      y: [(yLower + yUpper) / 2],
      text: [clusterLabel(cid)],
      textfont: { size: 11, color },
      hoverinfo: 'skip',
      showlegend: false,
    });

    yLower = yUpper + 10;
  });

  if (!traces.length) return null;

  const xMin = Math.max(-1, Math.min(-0.1, globalMin - 0.05));

  const yTop = Math.max(0, yLower - 10);

  const shapes = [];
  const annotations = [];
  if (Number.isFinite(avg)) {
    shapes.push({
      type: 'line',
      x0: avg,
      x1: avg,
      y0: 0,
      y1: yTop,
      line: { dash: 'dash', width: 2, color: '#000' },
    });

    annotations.push({
      x: avg,
      y: yTop,
      xref: 'x',
      yref: 'y',
      text: `Average = ${avg.toFixed(3)}`,
      showarrow: false,
      xanchor: 'left',
      yanchor: 'top',
      font: { size: 12, color: '#000' },
      bgcolor: 'rgba(255,255,255,0.75)',
      bordercolor: 'rgba(0,0,0,0.15)',
      borderwidth: 1,
      borderpad: 2,
    });
  }

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader
        title="Silhouette scores"
        help="Silhouette coefficients can be negative when some samples are closer to a neighboring cluster than to their assigned cluster."
      />
      <Plot
        data={[...traces, ...hoverPoints]}
        layout={{
          margin: { ...PLOT_MARGIN_STD, t: 18, l: 55 },
          xaxis: {
            title: AXIS_TITLE('Silhouette coefficient'),
            tickfont: AXIS_TICK,
            range: [xMin, 1.0],
            showgrid: true,
            gridcolor: 'rgba(200,200,200,0.35)',
            zeroline: true,
            zerolinecolor: 'rgba(0,0,0,0.35)',
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          yaxis: {
            title: { text: '' },
            tickfont: { size: 10 },
            showgrid: false,
            showticklabels: false,
            zeroline: false,
            showline: false,
          },
          shapes,
          annotations,
          ...PLOT_BG,
        }}
        config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
        className="unsupPlotMd"
      />
    </Stack>
  );
}
