export const PLOT_BG = {
  plot_bgcolor: '#ffffff',
  paper_bgcolor: 'rgba(0,0,0,0)',
};

export const AXIS_TICK = { size: 11 };

export const AXIS_TITLE = (text) => ({
  text,
  font: { size: 13, weight: 'bold' },
});

export const PLOT_MARGIN_STD = { l: 55, r: 20, t: 12, b: 55 };
export const PLOT_MARGIN_TIGHT = { l: 55, r: 20, t: 16, b: 55 };

export const LEGEND_TOP = {
  orientation: 'h',
  y: 1.12,
  x: 0,
  xanchor: 'left',
  yanchor: 'bottom',
  font: { size: 11 },
};

// Match the confusion-matrix look (white → blue).
export const MENDER_BLUE_SCALE = [
  [0, '#ffffff'],
  [1, 'hsl(210, 80%, 45%)'],
];

export const PLOTLY_COLORS = [
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

export const LIGHT_BORDER = '1px solid #e5e7eb';
