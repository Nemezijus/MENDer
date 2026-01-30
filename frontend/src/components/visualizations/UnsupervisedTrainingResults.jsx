import { useMemo } from 'react';
import { ActionIcon, Group, SimpleGrid, Stack, Text, Tooltip } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import Plot from 'react-plotly.js';

const PLOT_MARGIN = { l: 55, r: 20, t: 20, b: 55 };

const AXIS_TITLE = (text) => ({ text, font: { size: 13, weight: 'bold' } });
const AXIS_TICK = { size: 11 };

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

function histogram(values, nBins = 30) {
  const xs = toFiniteNumbers(values);
  if (!xs.length) return null;
  const min = Math.min(...xs);
  const max = Math.max(...xs);
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return null;

  const bins = Math.max(5, Math.min(nBins, Math.floor(Math.sqrt(xs.length) * 2)));
  const width = (max - min) / bins;
  const counts = new Array(bins).fill(0);

  xs.forEach((v) => {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor((v - min) / width)));
    counts[idx] += 1;
  });

  const centers = Array.from({ length: bins }, (_, i) => min + (i + 0.5) * width);
  return { x: centers, y: counts };
}

function safeClusterSizes(clusterSummary) {
  const sizes = clusterSummary?.cluster_sizes;
  if (!sizes || typeof sizes !== 'object') return [];
  return Object.entries(sizes)
    .map(([k, v]) => ({ cluster_id: Number(k), size: Number(v) }))
    .filter((r) => Number.isFinite(r.cluster_id) && Number.isFinite(r.size))
    .sort((a, b) => a.cluster_id - b.cluster_id);
}

function lorenzFromSizes(sizes) {
  const vals = sizes
    .map((r) => Number(r.size))
    .filter((v) => Number.isFinite(v) && v >= 0)
    .sort((a, b) => a - b);
  const n = vals.length;
  if (n < 1) return null;
  const total = vals.reduce((s, v) => s + v, 0);
  if (!Number.isFinite(total) || total <= 0) return null;

  const x = [0];
  const y = [0];
  let cum = 0;
  for (let i = 0; i < n; i += 1) {
    cum += vals[i];
    x.push((i + 1) / n);
    y.push(cum / total);
  }

  // Gini from Lorenz via trapezoid area
  let area = 0;
  for (let i = 1; i < x.length; i += 1) {
    const dx = x[i] - x[i - 1];
    const yAvg = (y[i] + y[i - 1]) / 2;
    area += dx * yAvg;
  }
  const gini = 1 - 2 * area;
  return { x, y, gini: Number.isFinite(gini) ? gini : null };
}

function ellipsePoints(mean, cov, n = 100, scale = 2) {
  // Returns a (rough) ellipse outline in 2D from mean + covariance.
  // scale=2 roughly corresponds to ~95% for Gaussian if cov is well-behaved.
  const mx = mean?.[0];
  const my = mean?.[1];
  const a = cov?.[0]?.[0];
  const b = cov?.[0]?.[1];
  const c = cov?.[1]?.[0];
  const d = cov?.[1]?.[1];
  if (![mx, my, a, b, c, d].every((v) => typeof v === 'number' && Number.isFinite(v))) return null;

  // eigen-decomposition for 2x2
  const tr = a + d;
  const det = a * d - b * c;
  const disc = Math.max(tr * tr - 4 * det, 0);
  const s = Math.sqrt(disc);
  const l1 = (tr + s) / 2;
  const l2 = (tr - s) / 2;
  if (!(l1 > 0) || !(l2 > 0)) return null;

  // eigenvector for l1
  let vx = b;
  let vy = l1 - a;
  if (Math.abs(vx) + Math.abs(vy) < 1e-12) {
    vx = l1 - d;
    vy = c;
  }
  const norm = Math.hypot(vx, vy) || 1;
  vx /= norm;
  vy /= norm;

  const wx = -vy;
  const wy = vx;

  const rx = Math.sqrt(l1) * scale;
  const ry = Math.sqrt(l2) * scale;

  const xs = [];
  const ys = [];
  for (let i = 0; i <= n; i += 1) {
    const t = (i / n) * 2 * Math.PI;
    const ct = Math.cos(t);
    const st = Math.sin(t);
    const px = rx * ct;
    const py = ry * st;
    const x = mx + px * vx + py * wx;
    const y = my + px * vy + py * wy;
    xs.push(x);
    ys.push(y);
  }
  return { x: xs, y: ys };
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

const LEGEND_TOP = {
  orientation: 'h',
  yanchor: 'bottom',
  y: 1.12,
  xanchor: 'left',
  x: 0,
  font: { size: 11 },
};

export default function UnsupervisedTrainingResults({ trainResult }) {
  const clusterSummary = trainResult?.cluster_summary || {};
  const diag = trainResult?.diagnostics || {};
  const plotData = diag?.plot_data || {};

  const sizes = useMemo(() => safeClusterSizes(clusterSummary), [clusterSummary]);
  const lorenz = useMemo(() => lorenzFromSizes(sizes), [sizes]);

  const embedding = diag?.embedding_2d || null;
  const embX = useMemo(() => toFiniteNumbers(embedding?.x), [embedding]);
  const embY = useMemo(() => toFiniteNumbers(embedding?.y), [embedding]);
  const embLabel = useMemo(() => (Array.isArray(embedding?.label) ? embedding.label : null), [embedding]);

  const distanceToCenter = plotData?.distance_to_center || null;
  const distanceHist = useMemo(() => histogram(distanceToCenter?.values), [distanceToCenter]);

  const centroids = plotData?.centroids || null;
  const sepMatrix = plotData?.separation_matrix || null;
  const silhouette = plotData?.silhouette || null;
  const compactSep = plotData?.compactness_separation || null;
  const elbow = plotData?.elbow_curve || null;
  const kdist = plotData?.k_distance || null;
  const coreCounts = plotData?.core_border_noise_counts || null;
  const spectral = plotData?.spectral_eigenvalues || null;
  const gmmEllipses = plotData?.gmm_ellipses || null;

  const hasAnyGlobal = Boolean(
    (embedding && embX.length && embY.length) ||
      sizes.length ||
      (distanceHist && Array.isArray(distanceHist?.x) && Array.isArray(distanceHist?.y)) ||
      featureProfiles ||
      sepMatrix ||
      silhouette,
  );

  const hasAnyModelSpecific = Boolean(elbow || compactSep || kdist || coreCounts || spectral || gmmEllipses);

  const renderEmbedding = () => {
    if (!embedding || embX.length === 0 || embY.length === 0) return null;

    const traces = [];
    if (Array.isArray(embLabel) && embLabel.length === embX.length) {
      const by = new Map();
      for (let i = 0; i < embLabel.length; i += 1) {
        const lab = embLabel[i];
        if (!by.has(lab)) by.set(lab, { x: [], y: [] });
        by.get(lab).x.push(embX[i]);
        by.get(lab).y.push(embY[i]);
      }
      [...by.entries()].sort((a, b) => Number(a[0]) - Number(b[0])).forEach(([lab, pts]) => {
        traces.push({
          type: 'scattergl',
          mode: 'markers',
          name: `Cluster ${lab}`,
          x: pts.x,
          y: pts.y,
          marker: { size: 6, opacity: 0.75 },
          hovertemplate: 'x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
        });
      });
    } else {
      traces.push({
        type: 'scattergl',
        mode: 'markers',
        name: 'Samples',
        x: embX,
        y: embY,
        marker: { size: 6, opacity: 0.75 },
        hovertemplate: 'x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
        showlegend: false,
      });
    }

    // Optional: overlay GMM covariance ellipses in embedding space
    if (gmmEllipses && Array.isArray(gmmEllipses?.components)) {
      gmmEllipses.components.forEach((c, idx) => {
        const pts = ellipsePoints(c.mean, c.cov, 120, 2);
        if (!pts) return;
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: `Component ${idx}`,
          x: pts.x,
          y: pts.y,
          line: { width: 2 },
          hoverinfo: 'skip',
        });
      });
    }

    return (
      <Stack gap={6}>
        <PlotHeader
          title="2D embedding scatter"
          help="A 2D projection of your (preprocessed) features. Points are colored by cluster id when available."
        />
        <Plot
          data={traces}
          layout={{
            margin: { ...PLOT_MARGIN, t: 30 },
            xaxis: {
              title: AXIS_TITLE('Embedding 1'),
              tickfont: AXIS_TICK,
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.35)',
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: AXIS_TITLE('Embedding 2'),
              tickfont: AXIS_TICK,
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.35)',
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            hovermode: 'closest',
            ...PLOT_BG,
            legend: LEGEND_TOP,
          }}
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          style={{ width: '100%', height: 320 }}
        />
      </Stack>
    );
  };

  const renderClusterSizeBar = () => {
    if (!sizes.length) return null;
    return (
      <Stack gap={6}>
        <PlotHeader title="Cluster size distribution" help="Bar chart of samples per cluster id." />
        <Plot
          data={[
            {
              type: 'bar',
              x: sizes.map((r) => r.cluster_id),
              y: sizes.map((r) => r.size),
              hovertemplate: 'Cluster=%{x}<br>Size=%{y}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={{
            margin: { ...PLOT_MARGIN, t: 30 },
            xaxis: {
              title: AXIS_TITLE('Cluster id'),
              tickfont: AXIS_TICK,
              showgrid: false,
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: AXIS_TITLE('Size'),
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
      </Stack>
    );
  };

  const renderLorenz = () => {
    if (!lorenz) return null;
    const giniText = lorenz.gini == null ? '' : ` (Gini ≈ ${lorenz.gini.toFixed(3)})`;
    return (
      <Stack gap={6}>
        <PlotHeader
          title={`Cluster size inequality${giniText}`}
          help="Lorenz curve of cluster size distribution. Curves closer to the diagonal indicate more even cluster sizes."
        />
        <Plot
          data={[
            {
              type: 'scatter',
              mode: 'lines',
              x: lorenz.x,
              y: lorenz.y,
              name: 'Lorenz curve',
              hovertemplate: 'x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
            },
            {
              type: 'scatter',
              mode: 'lines',
              x: [0, 1],
              y: [0, 1],
              name: 'Equality',
              line: { dash: 'dash', width: 1 },
              hoverinfo: 'skip',
            },
          ]}
          layout={{
            margin: { ...PLOT_MARGIN, t: 30 },
            xaxis: {
              title: AXIS_TITLE('Cumulative share of clusters'),
              tickfont: AXIS_TICK,
              range: [0, 1],
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.4)',
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: AXIS_TITLE('Cumulative share of samples'),
              tickfont: AXIS_TICK,
              range: [0, 1],
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.4)',
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            legend: LEGEND_TOP,
            ...PLOT_BG,
          }}
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          style={{ width: '100%', height: 300 }}
        />
      </Stack>
    );
  };

  const renderDistanceHist = () => {
    const xs = distanceHist?.x;
    const ys = distanceHist?.y;
    if (!Array.isArray(xs) || !Array.isArray(ys) || xs.length === 0 || ys.length === 0) return null;
    return (
      <Stack gap={6}>
        <PlotHeader
          title="Distance-to-center distribution"
          help="Histogram of per-sample distances to the nearest cluster center (when the estimator supports it, e.g., KMeans)."
        />
        <Plot
          data={[
            {
              type: 'bar',
              x: xs,
              y: ys,
              hovertemplate: 'Distance≈%{x:.3f}<br>Count=%{y}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={{
            margin: { ...PLOT_MARGIN, t: 30 },
            xaxis: {
              title: AXIS_TITLE('Distance to nearest center'),
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
      </Stack>
    );
  };

  const renderFeatureProfiles = () => {
    if (!centroids) return null;
    const clusterIds = centroids?.cluster_ids;
    const featIdx = centroids?.feature_idx;
    const z = centroids?.values;
    if (!Array.isArray(clusterIds) || !Array.isArray(featIdx) || !Array.isArray(z)) return null;
    if (!z.length || !Array.isArray(z[0])) return null;

    return (
      <Stack gap={6}>
        <PlotHeader
          title="Per-cluster feature profiles"
          help="Heatmap of cluster centroids in the preprocessed feature space."
        />
        <Plot
          data={[
            {
              type: 'heatmap',
              z,
              x: featIdx.map((i) => `f${i}`),
              y: clusterIds.map((c) => `c${c}`),
              hovertemplate: 'Cluster=%{y}<br>Feature=%{x}<br>Value=%{z:.3f}<extra></extra>',
            },
          ]}
          layout={{
            margin: { ...PLOT_MARGIN, l: 70, t: 30 },
            xaxis: {
              title: AXIS_TITLE('Features'),
              tickfont: { size: 10 },
              showgrid: false,
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: AXIS_TITLE('Clusters'),
              tickfont: { size: 10 },
              showgrid: false,
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            ...PLOT_BG,
          }}
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          style={{ width: '100%', height: 320 }}
        />
      </Stack>
    );
  };

  const renderSeparationMatrix = () => {
    if (!sepMatrix) return null;
    const clusterIds = sepMatrix?.cluster_ids;
    const z = sepMatrix?.values;
    if (!Array.isArray(clusterIds) || !Array.isArray(z) || !z.length) return null;

    return (
      <Stack gap={6}>
        <PlotHeader
          title="Pairwise cluster separation matrix"
          help="Heatmap of centroid-to-centroid distances between clusters."
        />
        <Plot
          data={[
            {
              type: 'heatmap',
              z,
              x: clusterIds.map((c) => `c${c}`),
              y: clusterIds.map((c) => `c${c}`),
              hovertemplate: 'From=%{y}<br>To=%{x}<br>Distance=%{z:.3f}<extra></extra>',
            },
          ]}
          layout={{
            margin: { ...PLOT_MARGIN, l: 70, t: 30 },
            xaxis: {
              title: AXIS_TITLE('Cluster'),
              tickfont: { size: 10 },
              showgrid: false,
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: AXIS_TITLE('Cluster'),
              tickfont: { size: 10 },
              showgrid: false,
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            ...PLOT_BG,
          }}
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          style={{ width: '100%', height: 320 }}
        />
      </Stack>
    );
  };

  const renderSilhouette = () => {
    if (!silhouette || !Array.isArray(silhouette?.cluster_ids) || !Array.isArray(silhouette?.values)) return null;
    const perCluster = silhouette.cluster_ids
      .map((cid, i) => ({ cluster_id: cid, values: silhouette.values?.[i] }))
      .filter((c) => c && Array.isArray(c.values) && c.values.length)
      .sort((a, b) => Number(a.cluster_id) - Number(b.cluster_id));
    if (!perCluster.length) return null;

    const traces = perCluster.map((c) => ({
      type: 'violin',
      y: c.values,
      name: `c${c.cluster_id}`,
      box: { visible: true },
      meanline: { visible: true },
      points: false,
      hovertemplate: 'Silhouette=%{y:.3f}<extra></extra>',
    }));

    const overall = typeof silhouette?.avg === 'number' && Number.isFinite(silhouette.avg) ? silhouette.avg : null;

    return (
      <Stack gap={6}>
        <PlotHeader
          title="Silhouette plot (per cluster)"
          help="Distribution of per-sample silhouette values for each cluster (higher is better)."
        />
        <Plot
          data={traces}
          layout={{
            margin: { ...PLOT_MARGIN, b: 70, t: 30 },
            xaxis: {
              title: AXIS_TITLE('Cluster'),
              tickfont: { size: 10 },
              showgrid: false,
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: AXIS_TITLE('Silhouette value'),
              tickfont: AXIS_TICK,
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.4)',
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            ...(overall != null
              ? {
                  shapes: [
                    {
                      type: 'line',
                      x0: -0.5,
                      x1: perCluster.length - 0.5,
                      y0: overall,
                      y1: overall,
                      line: { width: 1, dash: 'dash' },
                    },
                  ],
                  annotations: [
                    {
                      x: 0.02,
                      xref: 'paper',
                      y: overall,
                      yref: 'y',
                      text: `mean=${overall.toFixed(3)}`,
                      showarrow: false,
                      font: { size: 12 },
                      align: 'left',
                      bgcolor: 'rgba(255,255,255,0.7)',
                    },
                  ],
                }
              : {}),
            ...PLOT_BG,
            showlegend: true,
            legend: LEGEND_TOP,
          }}
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          style={{ width: '100%', height: 320 }}
        />
      </Stack>
    );
  };

  const renderElbow = () => {
    if (!elbow || !Array.isArray(elbow?.x) || !Array.isArray(elbow?.y)) return null;
    if (!elbow.x.length || !elbow.y.length) return null;
    return (
      <Stack gap={6}>
        <PlotHeader title="Elbow curve" help="Objective vs number of clusters/components (when computed)." />
        <Plot
          data={[
            {
              type: 'scatter',
              mode: 'lines+markers',
              x: elbow.x,
              y: elbow.y,
              hovertemplate: 'k=%{x}<br>value=%{y:.3f}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={{
            margin: PLOT_MARGIN,
            xaxis: {
              title: AXIS_TITLE('k'),
              tickfont: AXIS_TICK,
              showgrid: false,
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: AXIS_TITLE('Objective'),
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
          style={{ width: '100%', height: 320 }}
        />
      </Stack>
    );
  };

  const renderCompactSep = () => {
    if (!compactSep || !Array.isArray(compactSep?.cluster_ids)) return null;
    const x = toFiniteNumbers(compactSep.compactness);
    const y = toFiniteNumbers(compactSep.separation);
    const ids = compactSep.cluster_ids;
    if (x.length === 0 || y.length === 0 || ids.length !== x.length || ids.length !== y.length) return null;

    return (
      <Stack gap={6}>
        <PlotHeader
          title="Cluster compactness vs separation"
          help="Each point is a cluster: compactness is within-cluster spread, separation is distance to the nearest other cluster centroid."
        />
        <Plot
          data={[
            {
              type: 'scatter',
              mode: 'markers+text',
              x,
              y,
              text: ids.map((c) => String(c)),
              textposition: 'top center',
              marker: { size: 10, opacity: 0.75 },
              hovertemplate: 'Cluster=%{text}<br>Compact=%{x:.3f}<br>Sep=%{y:.3f}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={{
            margin: PLOT_MARGIN,
            xaxis: {
              title: AXIS_TITLE('Compactness'),
              tickfont: AXIS_TICK,
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.4)',
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: AXIS_TITLE('Separation'),
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
          style={{ width: '100%', height: 320 }}
        />
      </Stack>
    );
  };

  const renderKdist = () => {
    if (!kdist || !Array.isArray(kdist?.y) || kdist.y.length === 0) return null;
    const y = toFiniteNumbers(kdist.y);
    if (!y.length) return null;
    const x = Array.from({ length: y.length }, (_, i) => i + 1);
    const kText = typeof kdist?.k === 'number' ? ` (k=${kdist.k})` : '';
    return (
      <Stack gap={6}>
        <PlotHeader
          title={`k-distance plot${kText}`}
          help="Sorted distance to the k-th nearest neighbor. Often used to guide DBSCAN eps selection."
        />
        <Plot
          data={[
            {
              type: 'scatter',
              mode: 'lines',
              x,
              y,
              hovertemplate: 'Rank=%{x}<br>Dist=%{y:.3f}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={{
            margin: PLOT_MARGIN,
            xaxis: {
              title: AXIS_TITLE('Rank'),
              tickfont: AXIS_TICK,
              showgrid: false,
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: AXIS_TITLE('k-distance'),
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
          style={{ width: '100%', height: 320 }}
        />
      </Stack>
    );
  };

  const renderCoreCounts = () => {
    if (!coreCounts) return null;
    const core = Number(coreCounts?.core);
    const border = Number(coreCounts?.border);
    const noise = Number(coreCounts?.noise);
    if (![core, border, noise].every((v) => Number.isFinite(v))) return null;
    return (
      <Stack gap={6}>
        <PlotHeader title="Core vs border vs noise counts" help="Available for density-based clustering (e.g., DBSCAN)." />
        <Plot
          data={[
            {
              type: 'bar',
              x: ['Core', 'Border', 'Noise'],
              y: [core, border, noise],
              hovertemplate: '%{x}: %{y}<extra></extra>',
              showlegend: false,
            },
          ]}
          layout={{
            margin: PLOT_MARGIN,
            xaxis: {
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
          style={{ width: '100%', height: 320 }}
        />
      </Stack>
    );
  };

  const renderSpectralGap = () => {
    if (!spectral || !Array.isArray(spectral?.values) || spectral.values.length === 0) return null;
    const y = toFiniteNumbers(spectral.values);
    if (!y.length) return null;
    const x = Array.from({ length: y.length }, (_, i) => i + 1);
    return (
      <Stack gap={6}>
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
            margin: PLOT_MARGIN,
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
          style={{ width: '100%', height: 320 }}
        />
      </Stack>
    );
  };

  if (!hasAnyGlobal && !hasAnyModelSpecific) {
    return (
      <Text size="sm" c="dimmed">
        No visualization payload was returned for this run.
      </Text>
    );
  }

  return (
    <Stack gap="md">
      <Group justify="space-between" align="center">
        <Text fw={600} size="md">
          Visualizations
        </Text>
        <Tooltip
          withArrow
          label="Plots derived from cluster summary, per-sample outputs, and model diagnostics."
          position="top-end"
        >
          <ActionIcon variant="subtle" size="sm" aria-label="Visualizations help">
            <IconInfoCircle size={16} />
          </ActionIcon>
        </Tooltip>
      </Group>

      {hasAnyGlobal ? (
        <>
          <Text fw={600} size="sm">
            Global plots
          </Text>
          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
            {renderEmbedding()}
            {renderClusterSizeBar()}
            {renderLorenz()}
            {renderDistanceHist()}
            {renderSilhouette()}
            {renderFeatureProfiles()}
            {renderSeparationMatrix()}
          </SimpleGrid>
        </>
      ) : null}

      {hasAnyModelSpecific ? (
        <>
          <Text fw={600} size="sm">
            Model-specific plots
          </Text>
          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md">
            {renderElbow()}
            {renderCompactSep()}
            {renderKdist()}
            {renderCoreCounts()}
            {renderSpectralGap()}
          </SimpleGrid>

          {gmmEllipses && gmmEllipses?.components?.length ? (
            <Text size="xs" c="dimmed">
              Note: covariance ellipses are overlaid on the embedding scatter when available.
            </Text>
          ) : null}
        </>
      ) : null}
    </Stack>
  );
}
