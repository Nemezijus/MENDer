import { useMemo } from 'react';
import { SimpleGrid, Stack, Text } from '@mantine/core';
import Plot from 'react-plotly.js';

import PlotHeader from '../common/PlotHeader.jsx';

import { histogram, toFiniteNumbers } from '../../utils/stats.js';
import { AXIS_TICK, AXIS_TITLE, PLOT_BG, PLOT_MARGIN_TIGHT } from '../../utils/plotly.js';

export default function UnsupervisedTrainingDecoderResults({ trainResult }) {
  const diag = trainResult?.diagnostics || {};
  const algo = trainResult?.model?.algo || null;
  const plotDataRoot = diag?.plot_data || {};
  const decoderData = plotDataRoot?.decoder || plotDataRoot;

  const confidenceHist = decoderData?.confidence_hist || null;
  const likelihoodHist = decoderData?.log_likelihood_hist || null;
  const noiseTrend = decoderData?.noise_trend || null;

  // Backward/forward compatibility: if backend does not emit histogram-ready arrays,
  // compute a small histogram locally from the raw per-sample values.
  const fallbackConfidenceHist = useMemo(() => {
    if (Array.isArray(confidenceHist?.x) && Array.isArray(confidenceHist?.y)) return confidenceHist;
    return histogram(decoderData?.confidence?.values);
  }, [confidenceHist, decoderData]);

  const fallbackLikelihoodHist = useMemo(() => {
    if (Array.isArray(likelihoodHist?.x) && Array.isArray(likelihoodHist?.y)) return likelihoodHist;
    return histogram(decoderData?.log_likelihood?.values);
  }, [likelihoodHist, decoderData]);

  const hasConfidence =
    Array.isArray(fallbackConfidenceHist?.x) &&
    Array.isArray(fallbackConfidenceHist?.y) &&
    fallbackConfidenceHist.x.length;

  const hasLikelihood =
    Array.isArray(fallbackLikelihoodHist?.x) &&
    Array.isArray(fallbackLikelihoodHist?.y) &&
    fallbackLikelihoodHist.x.length;

  const hasNoiseTrend =
    Array.isArray(noiseTrend?.x) && Array.isArray(noiseTrend?.y) && noiseTrend.x.length;

  const noiseX = useMemo(() => toFiniteNumbers(noiseTrend?.x), [noiseTrend]);
  const noiseY = useMemo(() => toFiniteNumbers(noiseTrend?.y), [noiseTrend]);
  const noiseAllZero = useMemo(
    () => (noiseY.length ? noiseY.every((v) => Math.abs(v) < 1e-12) : false),
    [noiseY],
  );

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
              x: fallbackConfidenceHist.x,
              y: fallbackConfidenceHist.y,
              hovertemplate: 'Confidence≈%{x:.3f}<br>Count=%{y}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={
            {
              margin: PLOT_MARGIN_TIGHT,
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
            }
          }
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          className="unsupDecoderPlot"
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
              x: fallbackLikelihoodHist.x,
              y: fallbackLikelihoodHist.y,
              hovertemplate: 'LogL≈%{x:.3f}<br>Count=%{y}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={
            {
              margin: PLOT_MARGIN_TIGHT,
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
            }
          }
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          className="unsupDecoderPlot"
        />
      </Stack>,
    );
  }

  if (hasNoiseTrend) {
    if (noiseAllZero) {
      plots.push(
        <Stack key="noise_zero" gap={6}>
          <PlotHeader
            title="Noise fraction trend"
            help="Cumulative fraction of samples marked as noise (label=-1) across sample index order."
          />
          <Text size="sm" c="dimmed" ta="center">
            {algo === 'dbscan'
              ? 'Noise trend is flat at 0.0 — this run produced no noise labels.'
              : 'This model does not produce noise labels — a flat 0.0 noise trend is expected.'}
          </Text>
        </Stack>,
      );
    } else if (noiseX.length && noiseY.length && noiseX.length === noiseY.length) {
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
            layout={
              {
                margin: PLOT_MARGIN_TIGHT,
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
              }
            }
            config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
            className="unsupDecoderPlot"
          />
        </Stack>,
      );
    }
  }

  if (!plots.length) return null;

  return (
    <Stack gap="md">
      <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md" className="unsupGridStretch">
        {plots}
      </SimpleGrid>
    </Stack>
  );
}
