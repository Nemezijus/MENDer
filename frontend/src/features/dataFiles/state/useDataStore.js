import { create } from 'zustand';

const initialState = {
  xPath: 'data/classical/wine/wine_features.mat',
  yPath: 'data/classical/wine/wine_labels.mat',
  npzPath: null,
  // Override-only: empty string means "unset" (backend will default to X/y).
  xKey: '',
  yKey: '',
  inspectReport: null,
  taskSelected: null,
  xDisplay: '',
  yDisplay: '',
  npzDisplay: '',
};

export const useDataStore = create((set) => ({
  // raw state
  ...initialState,

  // setters
  setXPath: (xPath) => set({ xPath }),
  setYPath: (yPath) => set({ yPath }),
  setNpzPath: (npzPath) => set({ npzPath }),
  setXKey: (xKey) => set({ xKey }),
  setYKey: (yKey) => set({ yKey }),
  setInspectReport: (inspectReport) => set({ inspectReport }),
  setTaskSelected: (taskSelected) => set({ taskSelected }),

  setXDisplay: (xDisplay) => set({ xDisplay }),
  setYDisplay: (yDisplay) => set({ yDisplay }),
  setNpzDisplay: (npzDisplay) => set({ npzDisplay }),

  // utility
  reset: () => set({ ...initialState }),
}));
