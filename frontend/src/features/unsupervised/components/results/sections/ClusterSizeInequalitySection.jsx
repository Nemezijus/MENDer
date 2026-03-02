import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';
import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_STD } from '../../../utils/plotly.js';

import { LEGEND_TOP_TIGHT } from '../common/styles.js';

export default function ClusterSizeInequalitySection({ lorenz }) {
  if (!lorenz) return null;

  const giniText = lorenz.gini == null ? '' : ` (Gini ≈ ${lorenz.gini.toFixed(3)})`;

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader
        title={`Cluster size inequality${giniText}`}
        help="Lorenz curve of cluster size distribution. Curves closer to the diagonal indicate more even cluster sizes."
      />
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines',
            x: lorenz.x,
            y: lorenz.y,
            name: 'Lorenz curve',
            hovertemplate: 'x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
          },
          {
            type: 'scatter',
            mode: 'lines',
            x: [0, 1],
            y: [0, 1],
            name: 'Equality',
            line: { dash: 'dash', width: 1 },
            hoverinfo: 'skip',
          },
        ]}
        layout={{
          margin: { ...PLOT_MARGIN_STD, t: 24 },
          xaxis: {
            title: AXIS_TITLE('Cumulative share of clusters'),
            tickfont: AXIS_TICK,
            range: [0, 1],
            showgrid: true,
            gridcolor: 'rgba(200,200,200,0.4)',
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          yaxis: {
            title: AXIS_TITLE('Cumulative share of samples'),
            tickfont: AXIS_TICK,
            range: [0, 1],
            showgrid: true,
            gridcolor: 'rgba(200,200,200,0.4)',
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          legend: LEGEND_TOP_TIGHT,
          ...PLOT_BG,
        }}
        config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
        className="unsupPlotSm"
      />
    </Stack>
  );
}
