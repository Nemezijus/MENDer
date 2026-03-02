import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';
import PlotTitle from '../common/PlotTitle.jsx';
import {
  LEGEND_INSIDE,
  makeBaseLayout,
  PLOT_CONFIG,
  PLOT_CONTAINER_CLASS,
  PLOT_INNER_CLASS,
} from '../../utils/plotly.js';

export default function ResidualHistogram({ x, counts, widths }) {
  const histX = Array.isArray(x) ? x : [];
  const histCounts = Array.isArray(counts) ? counts : [];

  if (histX.length === 0 || histCounts.length === 0 || histCounts.length !== histX.length) return null;

  const layout = makeBaseLayout({
    legend: LEGEND_INSIDE,
    xaxis: {
      title: {
        text: 'Residuals (predicted − true)',
        font: { size: 16, weight: 'bold' },
      },
      tickfont: { size: 14 },
      showgrid: true,
      gridcolor: 'rgba(200,200,200,0.4)',
      zeroline: false,
      showline: true,
      linecolor: '#000',
      linewidth: 1,
    },
    yaxis: {
      title: { text: 'Count', font: { size: 16, weight: 'bold' } },
      tickfont: { size: 14 },
      showgrid: true,
      gridcolor: 'rgba(200,200,200,0.4)',
      zeroline: false,
      showline: true,
      linecolor: '#000',
      linewidth: 1,
    },
  });

  return (
    <Stack gap="xs">
      <PlotTitle
        label="Residual distribution"
        tip="Histogram of residuals (predicted − true). Centered near zero suggests low bias; wider spread indicates larger errors."
      />

      <div className={PLOT_CONTAINER_CLASS}>
        <div className={PLOT_INNER_CLASS}>
          <Plot
            data={[
              {
                x: histX,
                y: histCounts,
                type: 'bar',
                name: 'Residuals',
                marker: { color: '#9576c9', opacity: 0.85 },
                width: Array.isArray(widths) && widths.length === histX.length ? widths : undefined,
                hovertemplate: 'Residual≈%{x}<br>Count=%{y}<extra></extra>',
                showlegend: false,
              },
            ]}
            layout={layout}
            config={PLOT_CONFIG}
            className="plotSizeStandard"
          />
        </div>
      </div>
    </Stack>
  );
}
