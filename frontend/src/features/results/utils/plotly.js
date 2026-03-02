// Shared Plotly helpers for results panels.

export const PLOT_CONFIG = {
  displayModeBar: false,
  responsive: true,
  useResizeHandler: true,
};

export const DEFAULT_MARGIN = { l: 70, r: 35, t: 20, b: 70 };

// CSS hooks (see features/results/styles/resultTables.css)
export const PLOT_CONTAINER_CLASS = 'resultsPlotOuter';
export const PLOT_INNER_CLASS = 'resultsPlotInner';

export const LEGEND_INSIDE = {
  x: 0.98,
  y: 0.02,
  xanchor: 'right',
  yanchor: 'bottom',
  bgcolor: 'rgba(255,255,255,0.7)',
  borderwidth: 0,
};

export function makeBaseLayout(overrides = {}) {
  return {
    margin: DEFAULT_MARGIN,
    hovermode: 'closest',
    plot_bgcolor: '#ffffff',
    paper_bgcolor: 'rgba(0,0,0,0)',
    ...overrides,
  };
}
