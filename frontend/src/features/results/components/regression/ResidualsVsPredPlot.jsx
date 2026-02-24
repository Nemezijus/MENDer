import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';
import PlotTitle from '../common/PlotTitle.jsx';
import {
  LEGEND_INSIDE,
  makeBaseLayout,
  PLOT_CONFIG,
  PLOT_CONTAINER_STYLE,
  PLOT_INNER_STYLE,
} from '../../utils/plotly.js';

export default function ResidualsVsPredPlot({ points }) {
  const residualScatter =
    points && Array.isArray(points.x) && Array.isArray(points.y) ? points : null;
  if (!residualScatter) return null;

  const xs = residualScatter.x;
  const hasX = Array.isArray(xs) && xs.length > 0;
  const xMin = hasX ? Math.min(...xs) : 0;
  const xMax = hasX ? Math.max(...xs) : 0;

  const layout = makeBaseLayout({
    legend: LEGEND_INSIDE,
    xaxis: {
      title: { text: 'Predicted values', font: { size: 16, weight: 'bold' } },
      tickfont: { size: 14 },
      showgrid: true,
      gridcolor: 'rgba(200,200,200,0.4)',
      zeroline: false,
      showline: true,
      linecolor: '#000',
      linewidth: 1,
    },
    yaxis: {
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
  });

  const data = [
    {
      x: residualScatter.x,
      y: residualScatter.y,
      type: 'scattergl',
      mode: 'markers',
      name: 'Samples',
      marker: { size: 4, color: '#9576c9', opacity: 0.55 },
      hovertemplate: 'Predicted=%{x}<br>Residual=%{y}<extra></extra>',
      showlegend: true,
    },
  ];

  // Zero residual reference line.
  if (hasX) {
    data.push({
      x: [xMin, xMax],
      y: [0, 0],
      type: 'scatter',
      mode: 'lines',
      name: 'Zero residual',
      line: { color: 'rgba(120,120,120,0.9)', width: 2, dash: 'dash' },
      hoverinfo: 'none',
      showlegend: true,
    });
  }

  return (
    <Stack gap="xs">
      <PlotTitle
        label="Residuals vs predicted values"
        tip="Residuals (predicted − true) plotted against predicted values. Patterns can indicate nonlinearity or heteroscedasticity."
      />

      <div style={PLOT_CONTAINER_STYLE}>
        <div style={PLOT_INNER_STYLE}>
          <Plot data={data} layout={layout} config={PLOT_CONFIG} style={{ width: '100%', height: 380 }} />
        </div>
      </div>
    </Stack>
  );
}
