export const TILE_MIN_HEIGHT = 420;
export const PLOT_HEIGHT = 320;

export const TILE_STACK_STYLE = {
  height: '100%',
  minHeight: TILE_MIN_HEIGHT,
  justifyContent: 'flex-start',
};

export const PLOT_BOX_STYLE = {
  flex: 1,
  display: 'flex',
  alignItems: 'flex-end',
};

// This legend positioning matches the original UnsupervisedTrainingResults.jsx behavior.
export const LEGEND_TOP_TIGHT = {
  orientation: 'h',
  yanchor: 'top',
  y: 1.12,
  xanchor: 'left',
  x: 0,
  font: { size: 11 },
};
