import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../../common/PlotHeader.jsx';

import { toFiniteNumbers } from '../../../utils/stats.js';
import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_STD } from '../../../utils/plotly.js';



export default function SpectralEigenvaluesSection({ spectral }) {
  if (!spectral || !Array.isArray(spectral?.values) || spectral.values.length === 0) return null;
  const y = toFiniteNumbers(spectral.values);
  if (!y.length) return null;
  const x = Array.from({ length: y.length }, (_, i) => i + 1);

  return (
    <Stack gap={4} className="unsupTileStack">
      <PlotHeader
        title="Spectral eigenvalues"
        help="Eigenvalue spectrum (when available). A large gap may indicate a good cluster count."
      />
      <Plot
        data={[
          {
            type: 'scatter',
            mode: 'lines+markers',
            x,
            y,
            hovertemplate: 'Index=%{x}<br>λ=%{y:.4f}<extra></extra>',
            showlegend: false,
          },
        ]}
        layout={{
          margin: PLOT_MARGIN_STD,
          xaxis: {
            title: AXIS_TITLE('Index'),
            tickfont: AXIS_TICK,
            showgrid: false,
            zeroline: false,
            showline: true,
            linecolor: '#000',
            linewidth: 1,
          },
          yaxis: {
            title: AXIS_TITLE('Eigenvalue'),
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
