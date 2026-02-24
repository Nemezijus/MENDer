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

export default function ErrorByTrueBinPlot({ x, mae, rmse }) {
  const xs = Array.isArray(x) ? x : [];
  const ysMae = Array.isArray(mae) ? mae : [];
  const ysRmse = Array.isArray(rmse) ? rmse : [];

  if (xs.length === 0 || ysMae.length !== xs.length || ysRmse.length !== xs.length) return null;

  const layout = makeBaseLayout({
    legend: LEGEND_INSIDE,
    xaxis: {
      title: { text: 'True values (bin centers)', font: { size: 16, weight: 'bold' } },
      tickfont: { size: 14 },
      showgrid: true,
      gridcolor: 'rgba(200,200,200,0.4)',
      zeroline: false,
      showline: true,
      linecolor: '#000',
      linewidth: 1,
    },
    yaxis: {
      title: { text: 'Error', font: { size: 16, weight: 'bold' } },
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
        label="Error by target magnitude (quantile bins)"
        tip="MAE and RMSE computed within quantile bins of the true target value to show where errors are largest (e.g., at extremes)."
      />

      <div style={PLOT_CONTAINER_STYLE}>
        <div style={PLOT_INNER_STYLE}>
          <Plot
            data={[
              {
                x: xs,
                y: ysMae,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'MAE',
                line: { color: '#2a9d8f', width: 2 },
                marker: { size: 6, color: '#2a9d8f' },
                hovertemplate: 'Bin center=%{x}<br>MAE=%{y}<extra></extra>',
                showlegend: true,
              },
              {
                x: xs,
                y: ysRmse,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'RMSE',
                line: { color: '#e36040', width: 2 },
                marker: { size: 6, color: '#e36040' },
                hovertemplate: 'Bin center=%{x}<br>RMSE=%{y}<extra></extra>',
                showlegend: true,
              },
            ]}
            layout={layout}
            config={PLOT_CONFIG}
            style={{ width: '100%', height: 360 }}
          />
        </div>
      </div>
    </Stack>
  );
}
