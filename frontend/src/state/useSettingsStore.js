import { create } from 'zustand';

export const useSettingsStore = create((set) => ({
  // Global / project-wide defaults
  scaleMethod: 'standard',
  metric: null,

  setScaleMethod: (scaleMethod) =>
    set({
      scaleMethod: scaleMethod || 'none',
    }),

  setMetric: (metric) =>
    set({
      metric: metric || null,
    }),
}));