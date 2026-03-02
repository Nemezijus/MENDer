import { useMemo } from 'react';
import { Stack, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';

import { ellipsePoints, toFiniteNumbers } from '../../../utils/stats.js';
import {
  AXIS_TICK,
  AXIS_TITLE,
  PLOT_BG,
  PLOTLY_COLORS,
  PLOT_MARGIN_STD,
} from '../../../utils/plotly.js';

import { PLOT_HEIGHT } from '../common/styles.js';

function ClusterLegend({ entries }) {
  if (!entries?.length) return null;

  return (
    <div className="unsupLegendWrap">
      {entries.map((e) => (
        <div
          key={e.label}
          className="unsupLegendChip"
        >
          <span
            className="unsupLegendDot"
            style={{ background: e.color }}
          />
          <Text size="sm" fw={500}>
            {e.label}
          </Text>
        </div>
      ))}
    </div>
  );
}

export default function EmbeddingScatterSection({ embedding, gmmEllipses, clusterLabel }) {
  const embX = useMemo(() => toFiniteNumbers(embedding?.x), [embedding]);
  const embY = useMemo(() => toFiniteNumbers(embedding?.y), [embedding]);
  const embLabel = useMemo(
    () => (Array.isArray(embedding?.label) ? embedding.label : null),
    [embedding],
  );

  if (!embedding || !embX.length || !embY.length) return null;

  const traces = [];
  const labels = embLabel && embLabel.length === embX.length ? embLabel : null;

  const legendEntries = [];

  if (labels) {
    const unique = Array.from(
      new Set(labels.map((v) => Number(v)).filter((v) => Number.isFinite(v))),
    ).sort((a, b) => a - b);

    unique.forEach((lab, idx) => {
      const xs = [];
      const ys = [];
      for (let i = 0; i < labels.length; i += 1) {
        if (Number(labels[i]) === lab) {
          xs.push(embX[i]);
          ys.push(embY[i]);
        }
      }
      if (!xs.length) return;

      const color = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
      legendEntries.push({ label: clusterLabel(lab), color });

      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: clusterLabel(lab),
        x: xs,
        y: ys,
        marker: { size: 6, opacity: 0.75, color },
        hovertemplate: `Cluster=${clusterLabel(lab)}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>`,
        showlegend: false,
      });
    });
  } else {
    traces.push({
      type: 'scattergl',
      mode: 'markers',
      name: 'Samples',
      x: embX,
      y: embY,
      marker: { size: 6, opacity: 0.75 },
      hovertemplate: 'x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
      showlegend: false,
    });
  }

  // Optional: overlay GMM covariance ellipses in embedding space
  if (gmmEllipses && Array.isArray(gmmEllipses?.components)) {
    gmmEllipses.components.forEach((c, idx) => {
      const pts = ellipsePoints(c.mean, c.cov, 120, 2);
      if (!pts) return;
      const color = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
      traces.push({
        type: 'scatter',
        mode: 'lines',
        name: `Component ${idx + 1}`,
        x: pts.x,
        y: pts.y,
        line: { width: 2, color },
        hoverinfo: 'skip',
        showlegend: false,
      });
    });
  }

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader
        title="2D embedding scatter"
        help="A 2D projection of your (preprocessed) features. Points are colored by cluster label when available."
      />
      {legendEntries.length ? <ClusterLegend entries={legendEntries} /> : null}

      <div className="unsupPlotBox">
        <Plot
          data={traces}
          layout={{
            margin: { ...PLOT_MARGIN_STD, t: 14, b: 48 },
            xaxis: {
              title: AXIS_TITLE('Embedding 1'),
              tickfont: AXIS_TICK,
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.35)',
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
              automargin: true,
            },
            yaxis: {
              title: AXIS_TITLE('Embedding 2'),
              tickfont: AXIS_TICK,
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.35)',
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
              scaleanchor: 'x',
              scaleratio: 1,
              automargin: true,
            },
            hovermode: 'closest',
            height: PLOT_HEIGHT,
            ...PLOT_BG,
          }}
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          className="unsupPlotFill"
        />
      </div>
    </Stack>
  );
}
