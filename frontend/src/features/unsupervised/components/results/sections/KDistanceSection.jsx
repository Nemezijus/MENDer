import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';

import { toFiniteNumbers } from '../../../utils/stats.js';
import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_STD } from '../../../utils/plotly.js';

import { TILE_STACK_STYLE } from '../common/styles.js';

export default function KDistanceSection({ kdist }) {
  if (!kdist || !Array.isArray(kdist?.y) || kdist.y.length === 0) return null;
  const y = toFiniteNumbers(kdist.y);
  if (!y.length) return null;
  const x = Array.from({ length: y.length }, (_, i) => i + 1);
  const kText = typeof kdist?.k === 'number' ? ` (k=${kdist.k})` : '';

  return (
    <Stack gap={4} style={TILE_STACK_STYLE}>
      <PlotHeader
        title={`k-distance plot${kText}`}
        help="Sorted distance to the k-th nearest neighbor. Often used to guide DBSCAN eps selection."
      />
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines',
            x,
            y,
            hovertemplate: 'Rank=%{x}<br>Dist=%{y:.3f}<extra></extra>',
            showlegend: false,
          },
        ]}
        layout={{
          margin: PLOT_MARGIN_STD,
          xaxis: {
            title: AXIS_TITLE('Rank'),
            tickfont: AXIS_TICK,
            showgrid: false,
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          yaxis: {
            title: AXIS_TITLE('k-distance'),
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
        style={{ width: '100%', height: 320, marginTop: 'auto' }}
      />
    </Stack>
  );
}
