import { Stack } from '@mantine/core';
import Plot from 'react-plotly.js';
import PlotTitle from '../common/PlotTitle.jsx';
import { makeBaseLayout, PLOT_CONFIG, PLOT_CONTAINER_STYLE, PLOT_INNER_STYLE } from '../../utils/plotly.js';

export default function FoldScoreDistribution({ foldScores, metricName, metricNameRaw }) {
  const scores = Array.isArray(foldScores)
    ? foldScores.filter((v) => typeof v === 'number' && Number.isFinite(v))
    : [];

  if (scores.length < 2) return null;

  const layout = makeBaseLayout({
    xaxis: {
      visible: false,
      showgrid: false,
      zeroline: false,
      showline: false,
      showticklabels: false,
      ticks: '',
    },
    yaxis: {
      title: { text: metricName || 'score', font: { size: 16, weight: 'bold' } },
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
    <Stack gap="xs" mt="xs">
      <PlotTitle
        label={`Per-fold score distribution (${metricName || 'score'})`}
        tip={`Distribution of fold evaluation scores for metric "${metricNameRaw}". Each point corresponds to one fold's held-out score.`}
      />

      <div style={PLOT_CONTAINER_STYLE}>
        <div style={PLOT_INNER_STYLE}>
          <Plot
            data={[
              {
                type: 'violin',
                y: scores,
                name: 'Fold scores',
                box: { visible: true },
                meanline: { visible: true },
                points: 'all',
                jitter: 0.25,
                pointpos: 0,
                marker: { size: 6, opacity: 0.65 },
                showlegend: false,
                hovertemplate: 'Score=%{y}<extra></extra>',
              },
            ]}
            layout={layout}
            config={PLOT_CONFIG}
            style={{ width: '100%', height: 320 }}
          />
        </div>
      </div>
    </Stack>
  );
}
