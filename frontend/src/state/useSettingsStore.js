import { create } from 'zustand';

export const useSettingsStore = create((set) => ({
  /**
   * Global settings overrides only.
   *
   * IMPORTANT:
   *   Do not hardcode Engine defaults here.
   *   Leave fields `undefined` to fall back to `/api/v1/schema/defaults`.
   */
  scaleMethod: undefined,
  metric: undefined,

  setScaleMethod: (scaleMethod) => set({ scaleMethod }),

  setMetric: (metric) => set({ metric }),
}));