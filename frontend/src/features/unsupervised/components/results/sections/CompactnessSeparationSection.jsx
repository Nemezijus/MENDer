import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';

import { toFiniteNumbers } from '../../../utils/stats.js';
import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_STD } from '../../../utils/plotly.js';



export default function CompactnessSeparationSection({ compactSep }) {
  if (!compactSep || !Array.isArray(compactSep?.cluster_ids)) return null;
  const x = toFiniteNumbers(compactSep.compactness);
  const y = toFiniteNumbers(compactSep.separation);
  const ids = compactSep.cluster_ids;
  if (x.length === 0 || y.length === 0 || ids.length !== x.length || ids.length !== y.length) return null;

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader
        title="Cluster compactness vs separation"
        help="Each point is a cluster: compactness is within-cluster spread, separation is distance to the nearest other cluster centroid."
      />
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'markers+text',
            x,
            y,
            text: ids.map((c) => String(c)),
            textposition: 'top center',
            marker: { size: 10, opacity: 0.75 },
            hovertemplate: 'Cluster=%{text}<br>Compact=%{x:.3f}<br>Sep=%{y:.3f}<extra></extra>',
            showlegend: false,
          },
        ]}
        layout={{
          margin: PLOT_MARGIN_STD,
          xaxis: {
            title: AXIS_TITLE('Compactness'),
            tickfont: AXIS_TICK,
            showgrid: true,
            gridcolor: 'rgba(200,200,200,0.4)',
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          yaxis: {
            title: AXIS_TITLE('Separation'),
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
        className="unsupPlotMd"
      />
    </Stack>
  );
}
