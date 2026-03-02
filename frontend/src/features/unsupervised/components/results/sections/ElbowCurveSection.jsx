import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';
import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_STD } from '../../../utils/plotly.js';



export default function ElbowCurveSection({ elbow }) {
  if (!elbow || !Array.isArray(elbow?.x) || !Array.isArray(elbow?.y)) return null;
  if (!elbow.x.length || !elbow.y.length) return null;

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader title="Elbow curve" help="Objective vs number of clusters/components (when computed)." />
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines+markers',
            x: elbow.x,
            y: elbow.y,
            hovertemplate: 'k=%{x}<br>value=%{y:.3f}<extra></extra>',
            showlegend: false,
          },
        ]}
        layout={{
          margin: PLOT_MARGIN_STD,
          xaxis: {
            title: AXIS_TITLE('k'),
            tickfont: AXIS_TICK,
            showgrid: false,
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          yaxis: {
            title: AXIS_TITLE('Objective'),
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
