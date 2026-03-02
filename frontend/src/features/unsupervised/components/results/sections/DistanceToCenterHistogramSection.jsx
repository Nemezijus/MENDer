import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';
import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_STD } from '../../../utils/plotly.js';



export default function DistanceToCenterHistogramSection({ distanceHist }) {
  const xs = distanceHist?.x;
  const ys = distanceHist?.y;
  if (!Array.isArray(xs) || !Array.isArray(ys) || xs.length === 0 || ys.length === 0) return null;

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader
        title="Distance-to-center distribution"
        help="Histogram of per-sample distances to the nearest cluster center (when the estimator supports it, e.g., KMeans)."
      />
      <Plot
        data={[
          {
            type: 'bar',
            x: xs,
            y: ys,
            hovertemplate: 'Distance≈%{x:.3f}<br>Count=%{y}<extra></extra>',
            showlegend: false,
          },
        ]}
        layout={{
          margin: { ...PLOT_MARGIN_STD, t: 24 },
          xaxis: {
            title: AXIS_TITLE('Distance to nearest center'),
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
        className="unsupPlotSm"
      />
    </Stack>
  );
}
