import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';
import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_STD } from '../../../utils/plotly.js';



export default function ClusterSizeDistributionSection({ sizes }) {
  const rows = Array.isArray(sizes) ? sizes : [];
  if (!rows.length) return null;

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader title="Cluster size distribution" help="Bar chart of samples per cluster id." />
      <Plot
        data={[
          {
            type: 'bar',
            x: rows.map((r) => r.cluster_id),
            y: rows.map((r) => r.size),
            hovertemplate: 'Cluster=%{x}<br>Size=%{y}<extra></extra>',
            showlegend: false,
          },
        ]}
        layout={{
          margin: { ...PLOT_MARGIN_STD, t: 24 },
          xaxis: {
            title: AXIS_TITLE('Cluster id'),
            tickfont: AXIS_TICK,
            showgrid: false,
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          yaxis: {
            title: AXIS_TITLE('Size'),
            tickfont: AXIS_TICK,
            showgrid: true,
            gridcolor: 'rgba(200,200,200,0.4)',
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          ...PLOT_BG,
        }}
        config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
        className="unsupPlotSm"
      />
    </Stack>
  );
}
