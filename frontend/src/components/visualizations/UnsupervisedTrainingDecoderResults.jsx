import { useMemo } from 'react';
import { Card, Divider, SimpleGrid, Stack, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

const PLOT_MARGIN = { l: 55, r: 20, t: 20, b: 55 };
const AXIS_TITLE = (text) => ({ text, font: { size: 16, weight: 'bold' } });
const AXIS_TICK = { size: 14 };
const PLOT_BG = {
  plot_bgcolor: '#ffffff',
  paper_bgcolor: 'rgba(0,0,0,0)',
};

function toFiniteNumbers(arr) {
  if (!Array.isArray(arr)) return [];
  return arr
    .map((v) => (typeof v === 'number' ? v : Number(v)))
    .filter((v) => Number.isFinite(v));
}

function plotTitle(title, help) {
  return (
    <Stack gap={2}>
      <Text fw={600} size="sm">
        {title}
      </Text>
      {help ? (
        <Text size="xs" c="dimmed">
          {help}
        </Text>
      ) : null}
    </Stack>
  );
}

export default function UnsupervisedTrainingDecoderResults({ trainResult }) {
  const diag = trainResult?.diagnostics || {};
  const plotData = diag?.plot_data || {};

  const confidenceHist = plotData?.confidence_hist || null;
  const likelihoodHist = plotData?.log_likelihood_hist || null;
  const noiseTrend = plotData?.noise_trend || null;

  const hasConfidence = Array.isArray(confidenceHist?.x) && Array.isArray(confidenceHist?.y) && confidenceHist.x.length;
  const hasLikelihood = Array.isArray(likelihoodHist?.x) && Array.isArray(likelihoodHist?.y) && likelihoodHist.x.length;
  const hasNoiseTrend = Array.isArray(noiseTrend?.x) && Array.isArray(noiseTrend?.y) && noiseTrend.x.length;

  const noiseX = useMemo(() => toFiniteNumbers(noiseTrend?.x), [noiseTrend]);
  const noiseY = useMemo(() => toFiniteNumbers(noiseTrend?.y), [noiseTrend]);

  const cards = [];

  if (hasConfidence) {
    cards.push(
      <Card key="conf" withBorder radius="md" shadow="sm" padding="md">
        {plotTitle(
          'Assignment confidence distribution',
          'Histogram of the maximum cluster membership probability (when available, e.g., GaussianMixture).',
        )}
        <Divider my="sm" />
        <Plot
          data={[
            {
              type: 'bar',
              x: confidenceHist.x,
              y: confidenceHist.y,
              hovertemplate: 'Confidence≈%{x:.3f}<br>Count=%{y}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={{
            margin: PLOT_MARGIN,
            xaxis: {
              title: AXIS_TITLE('Confidence'),
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
          style={{ width: '100%', height: 300 }}
        />
      </Card>,
    );
  }

  if (hasLikelihood) {
    cards.push(
      <Card key="lik" withBorder radius="md" shadow="sm" padding="md">
        {plotTitle(
          'Outlier score / likelihood distribution',
          'Histogram of per-sample log-likelihood (when available). Lower likelihood typically indicates more outlier-like samples.',
        )}
        <Divider my="sm" />
        <Plot
          data={[
            {
              type: 'bar',
              x: likelihoodHist.x,
              y: likelihoodHist.y,
              hovertemplate: 'LogL≈%{x:.3f}<br>Count=%{y}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={{
            margin: PLOT_MARGIN,
            xaxis: {
              title: AXIS_TITLE('Log-likelihood'),
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
          style={{ width: '100%', height: 300 }}
        />
      </Card>,
    );
  }

  if (hasNoiseTrend) {
    if (noiseX.length && noiseY.length && noiseX.length === noiseY.length) {
      cards.push(
        <Card key="noise" withBorder radius="md" shadow="sm" padding="md">
          {plotTitle(
            'Noise fraction trend',
            'Cumulative fraction of samples marked as noise (label=-1) across sample index order.',
          )}
          <Divider my="sm" />
          <Plot
            data={[
              {
                type: 'scatter',
                mode: 'lines',
                x: noiseX,
                y: noiseY,
                hovertemplate: 'Index=%{x}<br>Noise frac=%{y:.3f}<extra></extra>',
                showlegend: false,
              },
            ]}
            layout={{
              margin: PLOT_MARGIN,
              xaxis: {
                title: AXIS_TITLE('Sample index'),
                tickfont: AXIS_TICK,
                showgrid: false,
                zeroline: false,
                showline: true,
                linecolor: '#000',
                linewidth: 1,
              },
              yaxis: {
                title: AXIS_TITLE('Noise fraction'),
                tickfont: AXIS_TICK,
                range: [0, 1],
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
            style={{ width: '100%', height: 300 }}
          />
        </Card>,
      );
    }
  }

  if (!cards.length) return null;

  return (
    <Stack gap="md">
      <Text fw={600} size="sm">
        Decoder visualizations
      </Text>
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
        {cards}
      </SimpleGrid>
    </Stack>
  );
}
