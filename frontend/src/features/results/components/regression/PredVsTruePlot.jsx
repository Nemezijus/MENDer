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

export default function PredVsTruePlot({ points, idealLine }) {
  const scatterPoints =
    points && Array.isArray(points.x) && Array.isArray(points.y) ? points : null;
  if (!scatterPoints) return null;

  const ideal =
    idealLine && Array.isArray(idealLine.x) && Array.isArray(idealLine.y) ? idealLine : null;

  const layout = makeBaseLayout({
    legend: LEGEND_INSIDE,
    xaxis: {
      title: { text: 'True values', font: { size: 16, weight: 'bold' } },
      tickfont: { size: 14 },
      showgrid: true,
      gridcolor: 'rgba(200,200,200,0.4)',
      zeroline: false,
      showline: true,
      linecolor: '#000',
      linewidth: 1,
    },
    yaxis: {
      title: { text: 'Predicted values', font: { size: 16, weight: 'bold' } },
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
      x: scatterPoints.x,
      y: scatterPoints.y,
      type: 'scattergl',
      mode: 'markers',
      name: 'Samples',
      marker: { size: 4, color: '#2a9d8f', opacity: 0.55 },
      hovertemplate: 'True=%{x}<br>Predicted=%{y}<extra></extra>',
      showlegend: true,
    },
  ];

  if (ideal) {
    data.push({
      x: ideal.x,
      y: ideal.y,
      type: 'scatter',
      mode: 'lines',
      name: 'Ideal (y=x)',
      line: { color: 'rgba(120,120,120,0.9)', width: 2, dash: 'dash' },
      hoverinfo: 'none',
      showlegend: true,
    });
  }

  return (
    <Stack gap="xs">
      <PlotTitle
        label="Predicted vs true values"
        tip="Scatter plot of predicted values against true values. Closer to the diagonal means better predictions."
      />

      <div className={PLOT_CONTAINER_CLASS}>
        <div className={PLOT_INNER_CLASS}>
          <Plot data={data} layout={layout} config={PLOT_CONFIG} className="plotSizeTall" />
        </div>
      </div>
    </Stack>
  );
}
