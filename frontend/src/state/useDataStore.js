import { create } from 'zustand';

// Keep the same initial defaults as the old DataContext
const initialState = {
  xPath: 'data/classical/wine/wine_features.mat',
  yPath: 'data/classical/wine/wine_labels.mat',
  npzPath: null,
  xKey: 'X',
  yKey: 'y',
  inspectReport: null,
  taskSelected: null, // 'classification' | 'regression' | null
};

export const useDataStore = create((set) => ({
  // raw state
  ...initialState,

  // setters
  setXPath: (xPath) => set({ xPath }),
  setYPath: (yPath) => set({ yPath }),
  setNPZPath: (npzPath) => set({ npzPath }),
  setXKey: (xKey) => set({ xKey }),
  setYKey: (yKey) => set({ yKey }),
  setInspectReport: (inspectReport) => set({ inspectReport }),
  setTaskSelected: (taskSelected) => set({ taskSelected }),

  // utility
  reset: () => set({ ...initialState }),
}));
