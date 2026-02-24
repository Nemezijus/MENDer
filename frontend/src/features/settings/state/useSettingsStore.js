import { create } from 'zustand';

import { makeReset } from '../../../shared/state/storeFactories.js';

const INITIAL_STATE = {
  /**
   * Global settings overrides only.
   *
   * IMPORTANT:
   *   Do not hardcode Engine defaults here.
   *   Leave fields `undefined` to fall back to `/api/v1/schema/defaults`.
   */
  scaleMethod: undefined,
  metric: undefined,
};

export const useSettingsStore = create((set) => ({
  ...INITIAL_STATE,

  resetSettings: makeReset(INITIAL_STATE, set),

  setScaleMethod: (scaleMethod) => set({ scaleMethod }),
  setMetric: (metric) => set({ metric }),
}));
