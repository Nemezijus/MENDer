import { HEATMAP_COLORSCALE, safeNum } from './resultsFormat.js';

export function fmtIntish(x, placeholder = '—') {
  const n = safeNum(x);
  if (n == null) return placeholder;
  return String(Math.round(n));
}

export function buildCorrHeatmapTrace({
  matrix,
  labels,
  showText = true,
  hoverValueSource = 'customdata', // 'customdata' | 'text'
}) {
  if (!matrix || !labels?.length) return null;

  const zColor = matrix.map((row) =>
    row.map((v) => {
      const n = safeNum(v);
      if (n == null) return null;
      return (n + 1) / 2;
    }),
  );

  const text = showText
    ? matrix.map((row) =>
        row.map((v) => {
          const n = safeNum(v);
          return n == null ? '' : n.toFixed(2);
        }),
      )
    : undefined;

  const useCustomdata = hoverValueSource === 'customdata';

  return {
    type: 'heatmap',
    x: labels,
    y: labels,
    z: zColor,
    zmin: 0,
    zmax: 1,
    colorscale: HEATMAP_COLORSCALE,
    showscale: true,
    ...(useCustomdata ? { customdata: matrix } : null),
    colorbar: {
      x: 1.02,
      xanchor: 'left',
      xpad: 0,
      thickness: 12,
      len: 0.92,
      outlinewidth: 0,
      tickvals: [0, 0.5, 1],
      ticktext: ['-1', '0', '1'],
    },
    text: showText ? text : undefined,
    texttemplate: showText ? '%{text}' : undefined,
    hovertemplate:
      hoverValueSource === 'customdata'
        ? '<b>%{y}</b> vs <b>%{x}</b><br>corr: %{customdata:.2f}<extra></extra>'
        : '<b>%{y}</b> vs <b>%{x}</b><br>corr: %{text}<extra></extra>',
  };
}

export function buildAbsDiffHeatmapTrace({ matrix, labels, showText = true }) {
  if (!matrix || !labels?.length) return null;

  const flat = matrix
    .flat()
    .map((v) => safeNum(v))
    .filter((v) => v != null);
  const zmax = flat.length ? Math.max(...flat) : 1;

  const text = showText
    ? matrix.map((row) =>
        row.map((v) => {
          const n = safeNum(v);
          return n == null ? '' : n.toFixed(2);
        }),
      )
    : undefined;

  return {
    type: 'heatmap',
    x: labels,
    y: labels,
    z: matrix,
    zmin: 0,
    zmax: zmax || 1,
    colorscale: HEATMAP_COLORSCALE,
    showscale: true,
    colorbar: {
      x: 1.02,
      xanchor: 'left',
      xpad: 0,
      thickness: 12,
      len: 0.92,
      outlinewidth: 0,
    },
    text: showText ? text : undefined,
    texttemplate: showText ? '%{text}' : undefined,
    hovertemplate: '<b>%{y}</b> vs <b>%{x}</b><br>|Δ|: %{z:.4f}<extra></extra>',
  };
}
