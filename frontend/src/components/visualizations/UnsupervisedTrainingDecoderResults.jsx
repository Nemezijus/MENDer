import { useMemo } from 'react';
import { ActionIcon, Group, SimpleGrid, Stack, Text, Tooltip } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import Plot from 'react-plotly.js';

const PLOT_MARGIN = { l: 55, r: 20, t: 16, b: 55 };
const AXIS_TITLE = (text) => ({ text, font: { size: 13, weight: 'bold' } });
const AXIS_TICK = { size: 11 };
const LEGEND_TOP = { orientation: 'h', y: 1.12, x: 0, xanchor: 'left', yanchor: 'bottom', font: { size: 11 } };
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

function PlotHeader({ title, help }) {
  return (
    <Group justify="space-between" align="center" wrap="nowrap" gap="xs">
      <Text fw={600} size="sm">
        {title}
      </Text>
      {help ? (
        <Tooltip label={help} withArrow position="top-end">
          <ActionIcon variant="subtle" size="sm" aria-label={`${title} help`}>
            <IconInfoCircle size={16} />
          </ActionIcon>
        </Tooltip>
      ) : null}
    </Group>
  );
}

function SectionHeader({ title, help }) {
  return (
    <Group justify="space-between" align="center" wrap="nowrap" gap="xs">
      <Text fw={600} size="sm">
        {title}
      </Text>
      {help ? (
        <Tooltip label={help} withArrow position="top-end">
          <ActionIcon variant="subtle" size="sm" aria-label={`${title} help`}>
            <IconInfoCircle size={16} />
          </ActionIcon>
        </Tooltip>
      ) : null}
    </Group>
  );
}

export default function UnsupervisedTrainingDecoderResults({ trainResult }) {
  const diag = trainResult?.diagnostics || {};
  const plotDataRoot = diag?.plot_data || {};
  const decoderData = plotDataRoot?.decoder || plotDataRoot;

  const confidenceHist = decoderData?.confidence_hist || null;
  const likelihoodHist = decoderData?.log_likelihood_hist || null;
  const noiseTrend = decoderData?.noise_trend || null;

  const hasConfidence = Array.isArray(confidenceHist?.x) && Array.isArray(confidenceHist?.y) && confidenceHist.x.length;
  const hasLikelihood = Array.isArray(likelihoodHist?.x) && Array.isArray(likelihoodHist?.y) && likelihoodHist.x.length;
  const hasNoiseTrend = Array.isArray(noiseTrend?.x) && Array.isArray(noiseTrend?.y) && noiseTrend.x.length;

  const noiseX = useMemo(() => toFiniteNumbers(noiseTrend?.x), [noiseTrend]);
  const noiseY = useMemo(() => toFiniteNumbers(noiseTrend?.y), [noiseTrend]);

  const plots = [];

  if (hasConfidence) {
    plots.push(
      <Stack key="conf" gap={6}>
        <PlotHeader
          title="Assignment confidence distribution"
          help="Histogram of the maximum cluster membership probability (when available, e.g., GaussianMixture)."
        />
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
      </Stack>,
    );
  }

  if (hasLikelihood) {
    plots.push(
      <Stack key="lik" gap={6}>
        <PlotHeader
          title="Outlier score / likelihood distribution"
          help="Histogram of per-sample log-likelihood (when available). Lower likelihood typically indicates more outlier-like samples."
        />
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
      </Stack>,
    );
  }

  if (hasNoiseTrend) {
    if (noiseX.length && noiseY.length && noiseX.length === noiseY.length) {
      plots.push(
        <Stack key="noise" gap={6}>
          <PlotHeader
            title="Noise fraction trend"
            help="Cumulative fraction of samples marked as noise (label=-1) across sample index order."
          />
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
        </Stack>,
      );
    }
  }

  if (!plots.length) return null;

  return (
    <Stack gap="md">
      <SectionHeader
        title="Decoder visualizations"
        help="Per-sample diagnostics produced by the unsupervised model (when available)."
      />
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
        {plots}
      </SimpleGrid>
    </Stack>
  );
}
