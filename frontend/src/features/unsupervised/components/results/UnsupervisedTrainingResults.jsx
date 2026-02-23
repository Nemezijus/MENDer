import { useMemo } from 'react';
import { Divider, SimpleGrid, Stack, Text, Tooltip } from '@mantine/core';
import Plot from 'react-plotly.js';

const PLOT_MARGIN = { l: 55, r: 20, t: 12, b: 55 };

const AXIS_TITLE = (text) => ({ text, font: { size: 13, weight: 'bold' } });
const AXIS_TICK = { size: 11 };

const PLOT_BG = {
  plot_bgcolor: '#ffffff',
  paper_bgcolor: 'rgba(0,0,0,0)',
};

// Match the confusion-matrix look (white → blue).
const MENDER_BLUE_SCALE = [
  [0, '#ffffff'],
  [1, 'hsl(210, 80%, 45%)'],
];

const PLOTLY_COLORS = [
  '#636EFA',
  '#EF553B',
  '#00CC96',
  '#AB63FA',
  '#FFA15A',
  '#19D3F3',
  '#FF6692',
  '#B6E880',
  '#FF97FF',
  '#FECB52',
];

const TILE_MIN_HEIGHT = 420;
const PLOT_HEIGHT = 320;

const TILE_STACK_STYLE = {
  height: '100%',
  minHeight: TILE_MIN_HEIGHT,
  justifyContent: 'flex-start',
};

const PLOT_BOX_STYLE = {
  flex: 1,
  display: 'flex',
  alignItems: 'flex-end',
};

const LIGHT_BORDER = '1px solid #e5e7eb';

function ClusterLegend({ entries }) {
  if (!entries?.length) return null;

  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: 10,
        justifyContent: 'center',
        marginTop: 2,
        marginBottom: 2,
      }}
    >
      {entries.map((e) => (
        <div
          key={e.label}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 6,
            padding: '2px 8px',
            borderRadius: 999,
            background: 'rgba(0,0,0,0.03)',
          }}
        >
          <span
            style={{
              width: 10,
              height: 10,
              borderRadius: 999,
              background: e.color,
              display: 'inline-block',
            }}
          />
          <Text size="sm" fw={500}>
            {e.label}
          </Text>
        </div>
      ))}
    </div>
  );
}


function toFiniteNumbers(arr) {
  if (!Array.isArray(arr)) return [];
  return arr
    .map((v) => (typeof v === 'number' ? v : Number(v)))
    .filter((v) => Number.isFinite(v));
}

function hexToRgba(hex, alpha = 0.25) {
  if (typeof hex !== 'string' || !hex.startsWith('#') || (hex.length !== 7 && hex.length !== 4)) {
    return `rgba(0,0,0,${alpha})`;
  }

  let r;
  let g;
  let b;

  if (hex.length === 4) {
    r = parseInt(hex[1] + hex[1], 16);
    g = parseInt(hex[2] + hex[2], 16);
    b = parseInt(hex[3] + hex[3], 16);
  } else {
    r = parseInt(hex.slice(1, 3), 16);
    g = parseInt(hex.slice(3, 5), 16);
    b = parseInt(hex.slice(5, 7), 16);
  }

  if (![r, g, b].every((v) => Number.isFinite(v))) return `rgba(0,0,0,${alpha})`;
  return `rgba(${r},${g},${b},${alpha})`;
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
  const titleNode = (
    <div
      style={{
        minHeight: 28,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      <Text fw={600} size="md" ta="center">{title}</Text>
    </div>
  );

  return help ? (
    <Tooltip label={help} withArrow position="top">
      {titleNode}
    </Tooltip>
  ) : (
    titleNode
  );
}


function SectionDivider({ title, help }) {
  const divider = <Divider label={title} labelPosition="center" />;
  return help ? (
    <Tooltip label={help} withArrow position="top">
      <div>{divider}</div>
    </Tooltip>
  ) : (
    divider
  );
}

const LEGEND_TOP = {
  orientation: 'h',
  yanchor: 'top',
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
  const dendrogram = plotData?.dendrogram || null;

  const clusterLabel = (cid) => {
    const c = Number(cid);
    if (!Number.isFinite(c)) return String(cid);
    if (c === -1) return 'Noise';
    return `C${c + 1}`;
  };

  const hasAnyGlobal = Boolean(
    (embedding && embX.length && embY.length) ||
      sizes.length ||
      (distanceHist && Array.isArray(distanceHist?.x) && Array.isArray(distanceHist?.y)) ||
      centroids ||
      sepMatrix ||
      silhouette,
  );

  const hasAnyModelSpecific = Boolean(elbow || compactSep || kdist || coreCounts || spectral || gmmEllipses || dendrogram);

  const renderEmbedding = () => {
    if (!embedding || !embX.length || !embY.length) return null;

    const traces = [];
    const labels = embLabel && embLabel.length === embX.length ? embLabel : null;

    const legendEntries = [];

    if (labels) {
      const unique = Array.from(new Set(labels.map((v) => Number(v)).filter((v) => Number.isFinite(v)))).sort(
        (a, b) => a - b,
      );

      unique.forEach((lab, idx) => {
        const xs = [];
        const ys = [];
        for (let i = 0; i < labels.length; i += 1) {
          if (Number(labels[i]) === lab) {
            xs.push(embX[i]);
            ys.push(embY[i]);
          }
        }
        if (!xs.length) return;

        const color = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
        legendEntries.push({ label: clusterLabel(lab), color });

        traces.push({
          type: 'scattergl',
          mode: 'markers',
          name: clusterLabel(lab),
          x: xs,
          y: ys,
          marker: { size: 6, opacity: 0.75, color },
          hovertemplate: `Cluster=${clusterLabel(lab)}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>`,
          showlegend: false,
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
        const color = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
        traces.push({
          type: 'scatter',
          mode: 'lines',
          name: `Component ${idx + 1}`,
          x: pts.x,
          y: pts.y,
          line: { width: 2, color },
          hoverinfo: 'skip',
          showlegend: false,
        });
      });
    }

    return (
      <Stack gap={4} style={TILE_STACK_STYLE}>
        <PlotHeader
          title="2D embedding scatter"
          help="A 2D projection of your (preprocessed) features. Points are colored by cluster label when available."
        />
        {legendEntries.length ? <ClusterLegend entries={legendEntries} /> : null}

        <div style={PLOT_BOX_STYLE}>
          <Plot
            data={traces}
            layout={{
              margin: { ...PLOT_MARGIN, t: 14, b: 48 },
              xaxis: {
                title: AXIS_TITLE('Embedding 1'),
                tickfont: AXIS_TICK,
                showgrid: true,
                gridcolor: 'rgba(200,200,200,0.35)',
                zeroline: false,
                showline: true,
                linecolor: '#000',
                linewidth: 1,
                automargin: true,
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
                scaleanchor: 'x',
                scaleratio: 1,
                automargin: true,
              },
              hovermode: 'closest',
              height: PLOT_HEIGHT,
              ...PLOT_BG,
            }}
            config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
            style={{ width: '100%', height: '100%' }}
          />
        </div>
      </Stack>
    );
  };

  const renderClusterSizeBar = () => {
    if (!sizes.length) return null;
    return (
      <Stack gap={4} style={TILE_STACK_STYLE}>
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
            margin: { ...PLOT_MARGIN, t: 24 },
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
          style={{ width: '100%', height: 300, marginTop: 'auto' }}
        />
      </Stack>
    );
  };

  const renderLorenz = () => {
    if (!lorenz) return null;
    const giniText = lorenz.gini == null ? '' : ` (Gini ≈ ${lorenz.gini.toFixed(3)})`;
    return (
      <Stack gap={4} style={TILE_STACK_STYLE}>
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
            margin: { ...PLOT_MARGIN, t: 24 },
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
          style={{ width: '100%', height: 300, marginTop: 'auto' }}
        />
      </Stack>
    );
  };

  const renderDistanceHist = () => {
    const xs = distanceHist?.x;
    const ys = distanceHist?.y;
    if (!Array.isArray(xs) || !Array.isArray(ys) || xs.length === 0 || ys.length === 0) return null;
    return (
      <Stack gap={4} style={TILE_STACK_STYLE}>
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
            margin: { ...PLOT_MARGIN, t: 24 },
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
          style={{ width: '100%', height: 300, marginTop: 'auto' }}
        />
      </Stack>
    );
  };

  const renderFeatureProfiles = () => {
    if (!centroids) return null;
    const clusterIds = centroids?.cluster_ids;
    const featIdx = centroids?.feature_idx;
    const z = centroids?.values;
    if (!Array.isArray(clusterIds) || !Array.isArray(featIdx) || !Array.isArray(z) || !z.length) return null;

    const featureLabels = featIdx.map((i) => `f${i}`);
    const showFeatureTicks = featureLabels.length <= 25;

    return (
      <Stack gap={4} style={TILE_STACK_STYLE}>
        <PlotHeader
          title="Per-cluster feature profiles"
          help="Mean feature values per cluster (shown for top-variance features)."
        />

        <div style={{ width: '100%', marginTop: 'auto' }}>
          <div style={{ width: '100%', aspectRatio: '1 / 1', minHeight: PLOT_HEIGHT }}>
            <Plot
              data={[
              {
                type: 'heatmap',
                z,
                x: featureLabels,
                y: clusterIds.map((c) => clusterLabel(c)),
                hovertemplate: 'Cluster=%{y}<br>Feature=%{x}<br>Value=%{z:.3f}<extra></extra>',
                colorscale: MENDER_BLUE_SCALE,
                showscale: false,
              },
            ]}
            layout={{
              margin: { ...PLOT_MARGIN, l: 60, t: 14, b: 50 },
              xaxis: {
                title: AXIS_TITLE('Features'),
                tickfont: { size: 10 },
                showgrid: false,
                zeroline: false,
                showline: true,
                mirror: true,
                linecolor: '#e5e7eb',
                linewidth: 1,
                ticks: 'outside',
                showticklabels: showFeatureTicks,
                tickangle: showFeatureTicks ? -45 : 0,
                automargin: true,
              },
              yaxis: {
                title: { text: '' },
                tickfont: { size: 11 },
                showgrid: false,
                zeroline: false,
                showline: true,
                mirror: true,
                linecolor: '#e5e7eb',
                linewidth: 1,
                ticks: 'outside',
                ticklen: 3,
                showticklabels: true,
                automargin: true,
              },
              ...PLOT_BG,
            }}
            config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
            style={{ width: '100%', height: '100%' }}
          />
          </div>
        </div>
      </Stack>
    );
  };

  const renderSeparationMatrix = () => {
    if (!sepMatrix) return null;
    const ids = sepMatrix?.cluster_ids;
    const z = sepMatrix?.values;
    if (!Array.isArray(ids) || !Array.isArray(z) || !z.length) return null;

    return (
      <Stack gap={4} style={TILE_STACK_STYLE}>
        <PlotHeader
          title="Pairwise cluster separation matrix"
          help="Pairwise distances between cluster centroids (in feature space)."
        />

        <div style={{ width: '100%', marginTop: 'auto' }}>
          <div style={{ width: '100%', aspectRatio: '1 / 1', minHeight: PLOT_HEIGHT }}>
            <Plot
              data={[
                {
                  type: 'heatmap',
                  z,
                  x: ids.map((c) => clusterLabel(c)),
                  y: ids.map((c) => clusterLabel(c)),
                  hovertemplate: '%{y} vs %{x}<br>Distance=%{z:.3f}<extra></extra>',
                  colorscale: MENDER_BLUE_SCALE,
                  showscale: false,
                },
              ]}
              layout={{
                margin: { ...PLOT_MARGIN, l: 60, t: 14, b: 50 },
                xaxis: {
                  title: AXIS_TITLE('Cluster'),
                  tickfont: { size: 10 },
                  showgrid: false,
                  zeroline: false,
                  showline: true,
                  mirror: true,
                  linecolor: '#e5e7eb',
                  linewidth: 1,
                  ticks: 'outside',
                  automargin: true,
                  constrain: 'domain',
                },
                yaxis: {
                  title: AXIS_TITLE('Cluster'),
                  tickfont: { size: 10 },
                  showgrid: false,
                  zeroline: false,
                  showline: true,
                  mirror: true,
                  linecolor: '#e5e7eb',
                  linewidth: 1,
                  ticks: 'outside',
                  automargin: true,
                  scaleanchor: 'x',
                  scaleratio: 1,
                  constrain: 'domain',
                },
                ...PLOT_BG,
              }}
              config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
              style={{ width: '100%', height: '100%' }}
            />
          </div>
        </div>
      </Stack>
    );
  };

  const renderSilhouette = () => {
    if (!silhouette) return null;
    const clusterIds = silhouette?.cluster_ids;
    const groups = silhouette?.values;
    if (!Array.isArray(clusterIds) || !Array.isArray(groups) || clusterIds.length !== groups.length) return null;

    const avg = Number(silhouette?.avg);

    let yLower = 10;
    const traces = [];
    const hoverPoints = [];
    let globalMin = 1;

    clusterIds.forEach((cid, idx) => {
      const vals = toFiniteNumbers(groups[idx]).sort((a, b) => a - b);
      if (!vals.length) return;

      globalMin = Math.min(globalMin, vals[0]);

      const yUpper = yLower + vals.length;
      const ySeq = Array.from({ length: vals.length }, (_, i) => yLower + i);

      const color = PLOTLY_COLORS[idx % PLOTLY_COLORS.length];
      const fillcolor = hexToRgba(color, 0.25);

      // Filled polygon (mirrors sklearn's fill_betweenx between x=0 and silhouette values)
      // Build a simple, non-self-intersecting polygon even when values are negative.
      const zeros = new Array(vals.length).fill(0);
      const xPoly = [...zeros, ...vals.slice().reverse(), 0];
      const yPoly = [...ySeq, ...ySeq.slice().reverse(), ySeq[0]];

      traces.push({
        type: 'scatter',
        mode: 'lines',
        x: xPoly,
        y: yPoly,
        fill: 'toself',
        fillcolor,
        line: { width: 0, color },
        hoverinfo: 'skip',
        showlegend: false,
      });

      // Invisible markers to enable per-sample hover
      hoverPoints.push({
        type: 'scatter',
        mode: 'markers',
        x: vals,
        y: ySeq,
        marker: { size: 8, opacity: 0 },
        hovertemplate: `Cluster=${clusterLabel(cid)}<br>Silhouette=%{x:.3f}<extra></extra>`,
        showlegend: false,
      });

      // Cluster label (centered)
      traces.push({
        type: 'scatter',
        mode: 'text',
        x: [Math.max(0.02, Math.min(0.2, (Math.max(...vals) + Math.min(...vals)) / 2))],
        y: [(yLower + yUpper) / 2],
        text: [clusterLabel(cid)],
        textfont: { size: 11, color },
        hoverinfo: 'skip',
        showlegend: false,
      });

      yLower = yUpper + 10;
    });

    if (!traces.length) return null;

    const xMin = Math.max(-1, Math.min(-0.1, globalMin - 0.05));

    const yTop = Math.max(0, yLower - 10);

    const shapes = [];
    const annotations = [];
    if (Number.isFinite(avg)) {
      shapes.push({
        type: 'line',
        x0: avg,
        x1: avg,
        y0: 0,
        y1: yTop,
        line: { dash: 'dash', width: 2, color: '#000' },
      });

      annotations.push({
        x: avg,
        y: yTop,
        xref: 'x',
        yref: 'y',
        text: `Average = ${avg.toFixed(3)}`,
        showarrow: false,
        xanchor: 'left',
        yanchor: 'top',
        font: { size: 12, color: '#000' },
        bgcolor: 'rgba(255,255,255,0.75)',
        bordercolor: 'rgba(0,0,0,0.15)',
        borderwidth: 1,
        borderpad: 2,
      });
    }

    return (
      <Stack gap={4} style={TILE_STACK_STYLE}>
        <PlotHeader
          title="Silhouette scores"
          help="Silhouette coefficients can be negative when some samples are closer to a neighboring cluster than to their assigned cluster."
        />
        <Plot
          data={[...traces, ...hoverPoints]}
          layout={{
            margin: { ...PLOT_MARGIN, t: 18, l: 55 },
            xaxis: {
              title: AXIS_TITLE('Silhouette coefficient'),
              tickfont: AXIS_TICK,
              range: [xMin, 1.0],
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.35)',
              zeroline: true,
              zerolinecolor: 'rgba(0,0,0,0.35)',
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            yaxis: {
              title: { text: '' },
              tickfont: { size: 10 },
              showgrid: false,
              showticklabels: false,
              zeroline: false,
              showline: false,
            },
            shapes,
            annotations,
            ...PLOT_BG,
          }}
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          style={{ width: '100%', height: PLOT_HEIGHT, marginTop: 'auto' }}
        />
      </Stack>
    );
  };

  const renderElbow = () => {
    if (!elbow || !Array.isArray(elbow?.x) || !Array.isArray(elbow?.y)) return null;
    if (!elbow.x.length || !elbow.y.length) return null;
    return (
      <Stack gap={4} style={TILE_STACK_STYLE}>
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
          style={{ width: '100%', height: 320, marginTop: 'auto' }}
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
      <Stack gap={4} style={TILE_STACK_STYLE}>
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
          style={{ width: '100%', height: 320, marginTop: 'auto' }}
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
      <Stack gap={4} style={TILE_STACK_STYLE}>
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
          style={{ width: '100%', height: 320, marginTop: 'auto' }}
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
      <Stack gap={4} style={TILE_STACK_STYLE}>
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
          style={{ width: '100%', height: 320, marginTop: 'auto' }}
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
      <Stack gap={4} style={TILE_STACK_STYLE}>
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
          style={{ width: '100%', height: 320, marginTop: 'auto' }}
        />
      </Stack>
    );
  };

  const renderDendrogram = () => {
    if (!dendrogram || !Array.isArray(dendrogram?.segments) || !dendrogram.segments.length) return null;

    const traces = dendrogram.segments.map((seg, i) => ({
      type: 'scatter',
      mode: 'lines',
      x: seg.x,
      y: seg.y,
      hoverinfo: 'skip',
      showlegend: false,
      line: { width: 1 },
      name: `seg-${i}`,
    }));

    // Leaf markers (hover + cluster color)
    const leafOrder = Array.isArray(dendrogram?.leaf_order) ? dendrogram.leaf_order : [];
    const leafLabels = Array.isArray(dendrogram?.leaf_labels) ? dendrogram.leaf_labels : leafOrder.map((v) => String(v));
    const leafX = Array.isArray(dendrogram?.leaf_x) ? dendrogram.leaf_x : [];
    const leafClusterIds = Array.isArray(dendrogram?.leaf_cluster_ids) ? dendrogram.leaf_cluster_ids : [];

    const leafCount = Math.min(leafLabels.length, leafX.length);
    if (leafCount > 0 && leafClusterIds.length >= leafCount) {
      const byCluster = new Map();
      for (let i = 0; i < leafCount; i += 1) {
        const cid = Number(leafClusterIds[i]);
        if (!byCluster.has(cid)) byCluster.set(cid, []);
        byCluster.get(cid).push(i);
      }

      Array.from(byCluster.entries())
        .sort((a, b) => a[0] - b[0])
        .forEach(([cid, idxs]) => {
          traces.push({
            type: 'scatter',
            mode: 'markers',
            x: idxs.map((j) => leafX[j]),
            y: idxs.map(() => 0),
            text: idxs.map((j) => leafLabels[j]),
            name: clusterLabel(cid),
            marker: { size: 7, opacity: 0.9 },
            hovertemplate: `Leaf=%{text}<br>Cluster=${clusterLabel(cid)}<extra></extra>`,
          });
        });
    }

    return (
      <Stack gap={4} style={TILE_STACK_STYLE}>
        <PlotHeader
          title="Dendrogram"
          help="Hierarchical clustering dendrogram (AgglomerativeClustering). Leaf markers are colored by the assigned cluster label."
        />
        <Plot
          data={traces}
          layout={{
            margin: { ...PLOT_MARGIN, t: 30, l: 50 },
            xaxis: {
              title: AXIS_TITLE('Leaves'),
              tickfont: { size: 10 },
              showgrid: false,
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
              showticklabels: false,
            },
            yaxis: {
              title: AXIS_TITLE('Distance'),
              tickfont: AXIS_TICK,
              showgrid: true,
              gridcolor: 'rgba(200,200,200,0.4)',
              zeroline: false,
              showline: true,
              linecolor: '#000',
              linewidth: 1,
            },
            ...PLOT_BG,
            hovermode: 'closest',
            legend: LEGEND_TOP,
          }}
          config={{ displayModeBar: false, responsive: true, useResizeHandler: true }}
          style={{ width: '100%', height: 360, marginTop: 'auto' }}
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
      {hasAnyGlobal ? (
        <>
          <Text fw={700} size="xl" ta="center">
            Unsupervised model results
          </Text>
          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md" style={{ alignItems: 'stretch' }}>
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
          <Divider my="sm" />
          <Text fw={700} size="xl" ta="center">
            Model-specific plots
          </Text>
          <SimpleGrid cols={{ base: 1, md: 2 }} spacing="md" style={{ alignItems: 'stretch' }}>
            {renderElbow()}
            {renderCompactSep()}
            {renderKdist()}
            {renderCoreCounts()}
            {renderSpectralGap()}
          </SimpleGrid>

          {renderDendrogram()}

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
