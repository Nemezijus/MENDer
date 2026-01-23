import { Fragment } from 'react';
import { Card, Stack, Text, Table, Divider, Group, Tooltip } from '@mantine/core';
import Plot from 'react-plotly.js';

function parseNumber(v) {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string' && v.trim() !== '') {
    const x = Number(v);
    if (Number.isFinite(x)) return x;
  }
  return null;
}

function fmt3(v) {
  const num = parseNumber(v);
  if (num === null) return '—';
  if (Number.isInteger(num)) return String(num);
  return num.toFixed(3);
}

function rangeText(minV, maxV) {
  const a = parseNumber(minV);
  const b = parseNumber(maxV);
  if (a === null || b === null) return '—';
  return `${fmt3(a)} … ${fmt3(b)}`;
}

function binCenters(edges) {
  if (!Array.isArray(edges) || edges.length < 2) return [];
  const out = [];
  for (let i = 0; i < edges.length - 1; i++) {
    const a = parseNumber(edges[i]);
    const b = parseNumber(edges[i + 1]);
    if (a === null || b === null) continue;
    out.push((a + b) / 2);
  }
  return out;
}

function chunk(arr, n) {
  const out = [];
  for (let i = 0; i < arr.length; i += n) out.push(arr.slice(i, i + n));
  return out;
}

export default function RegressionResultsPanel({ trainResult }) {
  if (!trainResult) return null;

  const nSplits = trainResult?.n_splits;
  const isKFold = typeof nSplits === 'number' && Number.isFinite(nSplits) && nSplits > 1;

  const diag = trainResult.regression || null;
  const summary = diag?.summary || null;

  if (!summary || typeof summary !== 'object') return null;

  const predVsTrue = diag?.pred_vs_true || null;
  const residualHist = diag?.residual_hist || null;
  const residualsVsPred = diag?.residuals_vs_pred || null;
  const errorByBin = diag?.error_by_true_bin || null;
  const idealLine = diag?.ideal_line || null;

  const headerStyle = {
    backgroundColor: 'var(--mantine-color-gray-8)',
    textAlign: 'center',
    whiteSpace: 'nowrap',
  };

  const headerTextStyle = { lineHeight: 1.1, whiteSpace: 'nowrap' };

  const metricPairs = [
    { k: 'RMSE', v: summary.rmse, tip: 'Root mean squared error. Smaller is better.' },
    { k: 'MAE', v: summary.mae, tip: 'Mean absolute error. Smaller is better.' },
    {
      k: 'Median AE',
      v: summary.median_ae,
      tip: 'Median absolute error (robust to outliers). Smaller is better.',
    },
    { k: 'R²', v: summary.r2, tip: 'Coefficient of determination. Larger is better (1 is perfect).' },
    {
      k: 'Explained variance',
      v: summary.explained_variance,
      tip: 'Explained variance score. Larger is better (1 is perfect).',
    },
    {
      k: 'Bias (pred − true)',
      v: summary.bias,
      tip: 'Mean(predicted − true). Values > 0 indicate overestimation on average.',
    },
    {
      k: 'Residual std',
      v: summary.residual_std,
      tip: 'Standard deviation of residuals (predicted − true). Smaller is better.',
    },
    {
      k: 'Pearson r',
      v: summary.pearson_r,
      tip: 'Linear correlation between predictions and ground truth.',
    },
    {
      k: 'Spearman ρ',
      v: summary.spearman_r,
      tip: 'Rank correlation between predictions and ground truth.',
    },
    {
      k: 'NRMSE',
      v: summary.nrmse,
      tip: 'Normalized RMSE (e.g., divided by std(y_true)). Smaller is better.',
    },
    { k: 'N samples', v: summary.n, tip: 'Number of evaluation samples used for these metrics.' },
    {
      k: 'True range',
      v: rangeText(summary.y_true_min, summary.y_true_max),
      tip: 'Range of ground-truth values in the evaluation set.',
      isText: true,
    },
    {
      k: 'Pred range',
      v: rangeText(summary.y_pred_min, summary.y_pred_max),
      tip: 'Range of predicted values in the evaluation set.',
      isText: true,
    },
  ];

  const metricRows = chunk(metricPairs, 3);

  const scatterPoints =
    predVsTrue && Array.isArray(predVsTrue.x) && Array.isArray(predVsTrue.y) ? predVsTrue : null;

  const residualScatter =
    residualsVsPred && Array.isArray(residualsVsPred.x) && Array.isArray(residualsVsPred.y)
      ? residualsVsPred
      : null;

  const ideal =
    idealLine && Array.isArray(idealLine.x) && Array.isArray(idealLine.y) ? idealLine : null;

  const histEdges = residualHist?.edges;
  const histCounts = residualHist?.counts;
  const hasHist =
    Array.isArray(histEdges) &&
    Array.isArray(histCounts) &&
    histEdges.length >= 2 &&
    histCounts.length === histEdges.length - 1;

  const histX = hasHist ? binCenters(histEdges) : [];
  const histW = hasHist
    ? histEdges.slice(0, -1).map((e, i) => {
        const a = parseNumber(e);
        const b = parseNumber(histEdges[i + 1]);
        return a !== null && b !== null ? Math.abs(b - a) : 0;
      })
    : [];

  const hasBinned =
    errorByBin &&
    Array.isArray(errorByBin.edges) &&
    Array.isArray(errorByBin.mae) &&
    Array.isArray(errorByBin.rmse) &&
    errorByBin.edges.length >= 2 &&
    errorByBin.mae.length === errorByBin.edges.length - 1 &&
    errorByBin.rmse.length === errorByBin.edges.length - 1;

  const binnedX = hasBinned ? binCenters(errorByBin.edges) : [];

  const PLOT_CONTAINER = {
    width: '100%',
    display: 'flex',
    justifyContent: 'center',
  };
  const PLOT_INNER = { width: '70%', maxWidth: 860 };
  const PLOT_MARGIN = { l: 70, r: 35, t: 20, b: 70 };
  const RESIDUAL_COLOR = '#9576c9';

  const legendInside = {
    x: 0.98,
    y: 0.02,
    xanchor: 'right',
    yanchor: 'bottom',
    bgcolor: 'rgba(255,255,255,0.7)',
    borderwidth: 0,
  };

  const plotTitle = (label, tip) => (
    <Group justify="center" gap={0}>
      <Tooltip label={tip} multiline maw={360} withArrow>
        <Text fw={500} size="xl" align="center" style={{ cursor: 'help' }}>
          {label}
        </Text>
      </Tooltip>
    </Group>
  );

  return (
    <Card withBorder radius="md" shadow="sm" padding="md">
      <Stack gap="sm">
        <Text fw={600} size="xl" align="center">
          Regression diagnostics
        </Text>

        <Stack gap={6}>
          <Text size="sm" c="dimmed">
            {isKFold
              ? `Summary metrics computed on pooled out-of-fold predictions (concatenated across ${nSplits} folds).`
              : 'Summary metrics computed on the evaluation (test) set.'}
          </Text>
        </Stack>

        <Table
          withTableBorder={false}
          withColumnBorders={false}
          horizontalSpacing="xs"
          verticalSpacing="xs"
        >
          <Table.Thead>
            <Table.Tr style={headerStyle}>
              {Array.from({ length: 3 }).map((_, i) => (
                <Fragment key={`hdr-${i}`}>
                  <Table.Th style={{ textAlign: 'center' }}>
                    <Text size="xs" fw={600} c="white" style={headerTextStyle}>
                      Metric
                    </Text>
                  </Table.Th>
                  <Table.Th style={{ textAlign: 'center' }}>
                    <Text size="xs" fw={600} c="white" style={headerTextStyle}>
                      Value
                    </Text>
                  </Table.Th>
                </Fragment>
              ))}
            </Table.Tr>
          </Table.Thead>

          <Table.Tbody>
            {metricRows.map((row, idx) => (
              <Table.Tr
                key={idx}
                style={{
                  backgroundColor: idx % 2 === 1 ? 'var(--mantine-color-gray-1)' : 'white',
                }}
              >
                {row.map((r) => (
                  <Fragment key={r.k}>
                    <Table.Td style={{ textAlign: 'left' }}>
                      <Tooltip label={r.tip} multiline maw={320} withArrow>
                        <Text size="sm" fw={500} style={{ cursor: 'help' }}>
                          {r.k}
                        </Text>
                      </Tooltip>
                    </Table.Td>
                    <Table.Td style={{ textAlign: 'center' }}>
                      <Text size="sm" fw={700}>
                        {r.isText ? String(r.v ?? '—') : fmt3(r.v)}
                      </Text>
                    </Table.Td>
                  </Fragment>
                ))}

                {row.length < 3 &&
                  Array.from({ length: 3 - row.length }).map((_, p) => (
                    <Fragment key={`pad-${p}`}>
                      <Table.Td />
                      <Table.Td />
                    </Fragment>
                  ))}
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>

        <Divider />

        {scatterPoints && (
          <Stack gap="xs">
            {plotTitle(
              'Predicted vs true values',
              'Scatter plot of predicted values against true values. Closer to the diagonal means better predictions.',
            )}

            <div style={PLOT_CONTAINER}>
              <div style={PLOT_INNER}>
              <Plot
                data={[
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
                  ...(ideal
                    ? [
                        {
                          x: ideal.x,
                          y: ideal.y,
                          type: 'scatter',
                          mode: 'lines',
                          name: 'Ideal (y=x)',
                          line: { color: 'rgba(120,120,120,0.9)', width: 2, dash: 'dash' },
                          hoverinfo: 'none',
                          showlegend: true,
                        },
                      ]
                    : []),
                ]}
                layout={{
                  margin: PLOT_MARGIN,
                  legend: legendInside,
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
                  hovermode: 'closest',
                  plot_bgcolor: '#ffffff',
                  paper_bgcolor: 'rgba(0,0,0,0)',
                }}
                config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
                style={{ width: '100%', height: 420 }}
              />
              </div>
            </div>
          </Stack>
        )}

        {hasHist && (
          <>
            <Divider />
            <Stack gap="xs">
              {plotTitle(
                'Residual distribution',
                'Histogram of residuals (predicted − true). Centered near zero suggests low bias; wider spread indicates larger errors.',
              )}

              <div style={PLOT_CONTAINER}>
                <div style={PLOT_INNER}>
                <Plot
                  data={[
                    {
                      x: histX,
                      y: histCounts,
                      type: 'bar',
                      name: 'Residuals',
                      marker: { color: RESIDUAL_COLOR, opacity: 0.85 },
                      width: histW,
                      hovertemplate: 'Residual≈%{x}<br>Count=%{y}<extra></extra>',
                      showlegend: false,
                    },
                  ]}
                  layout={{
                    margin: PLOT_MARGIN,
                    legend: legendInside,
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
                    hovermode: 'closest',
                    plot_bgcolor: '#ffffff',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                  }}
                  config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
                  style={{ width: '100%', height: 360 }}
                />
                </div>
              </div>
            </Stack>
          </>
        )}

        {residualScatter && (
          <>
            <Divider />
            <Stack gap="xs">
              {plotTitle(
                'Residuals vs predicted values',
                'Residuals (predicted − true) plotted against predicted values. Patterns can indicate nonlinearity or heteroscedasticity.',
              )}

              <div style={PLOT_CONTAINER}>
                <div style={PLOT_INNER}>
                <Plot
                  data={[
                    {
                      x: residualScatter.x,
                      y: residualScatter.y,
                      type: 'scattergl',
                      mode: 'markers',
                      name: 'Samples',
                      marker: { size: 4, color: RESIDUAL_COLOR, opacity: 0.55 },
                      hovertemplate: 'Predicted=%{x}<br>Residual=%{y}<extra></extra>',
                      showlegend: true,
                    },
                    {
                      x: [Math.min(...residualScatter.x), Math.max(...residualScatter.x)],
                      y: [0, 0],
                      type: 'scatter',
                      mode: 'lines',
                      name: 'Zero residual',
                      line: { color: 'rgba(120,120,120,0.9)', width: 2, dash: 'dash' },
                      hoverinfo: 'none',
                      showlegend: true,
                    },
                  ]}
                  layout={{
                    margin: PLOT_MARGIN,
                    legend: legendInside,
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
                    hovermode: 'closest',
                    plot_bgcolor: '#ffffff',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                  }}
                  config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
                  style={{ width: '100%', height: 380 }}
                />
                </div>
              </div>
            </Stack>
          </>
        )}

        {hasBinned && (
          <>
            <Divider />
            <Stack gap="xs">
              {plotTitle(
                'Error by target magnitude (quantile bins)',
                'MAE and RMSE computed within quantile bins of the true target value to show where errors are largest (e.g., at extremes).',
              )}

              <div style={PLOT_CONTAINER}>
                <div style={PLOT_INNER}>
                <Plot
                  data={[
                    {
                      x: binnedX,
                      y: errorByBin.mae,
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'MAE',
                      line: { color: '#2a9d8f', width: 2 },
                      marker: { size: 6, color: '#2a9d8f' },
                      hovertemplate: 'Bin center=%{x}<br>MAE=%{y}<extra></extra>',
                      showlegend: true,
                    },
                    {
                      x: binnedX,
                      y: errorByBin.rmse,
                      type: 'scatter',
                      mode: 'lines+markers',
                      name: 'RMSE',
                      line: { color: '#e36040', width: 2 },
                      marker: { size: 6, color: '#e36040' },
                      hovertemplate: 'Bin center=%{x}<br>RMSE=%{y}<extra></extra>',
                      showlegend: true,
                    },
                  ]}
                  layout={{
                    margin: PLOT_MARGIN,
                    legend: legendInside,
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
                    hovermode: 'closest',
                    plot_bgcolor: '#ffffff',
                    paper_bgcolor: 'rgba(0,0,0,0)',
                  }}
                  config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
                  style={{ width: '100%', height: 360 }}
                />
                </div>
              </div>
            </Stack>
          </>
        )}
      </Stack>
    </Card>
  );
}
