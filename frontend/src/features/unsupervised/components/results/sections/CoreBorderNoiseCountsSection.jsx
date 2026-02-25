import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';
import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_STD } from '../../../utils/plotly.js';

import { TILE_STACK_STYLE } from '../common/styles.js';

export default function CoreBorderNoiseCountsSection({ coreCounts }) {
  if (!coreCounts) return null;
  const core = Number(coreCounts?.core);
  const border = Number(coreCounts?.border);
  const noise = Number(coreCounts?.noise);
  if (![core, border, noise].every((v) => Number.isFinite(v))) return null;

  return (
    <Stack gap={4} style={TILE_STACK_STYLE}>
      <PlotHeader title="Core vs border vs noise counts" help="Available for density-based clustering (e.g., DBSCAN)." />
      <Plot
        data={[
          {
            type: 'bar',
            x: ['Core', 'Border', 'Noise'],
            y: [core, border, noise],
            hovertemplate: '%{x}: %{y}<extra></extra>',
            showlegend: false,
          },
        ]}
        layout={{
          margin: PLOT_MARGIN_STD,
          xaxis: {
            tickfont: AXIS_TICK,
            showgrid: false,
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          yaxis: {
            title: AXIS_TITLE('Count'),
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
