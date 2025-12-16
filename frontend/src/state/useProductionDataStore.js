import { create } from 'zustand';

export const useProductionDataStore = create((set) => ({
  // paths / keys
  xPath: '',
  yPath: '',
  npzPath: '',
  xKey: 'X',
  yKey: 'y',

  inspectReport: null,
  // setter for inspectReport
  setInspectReport: (inspectReport) => set({ inspectReport }),
  // setters
  setXPath: (xPath) => set({ xPath }),
  setYPath: (yPath) => set({ yPath }),
  setNpzPath: (npzPath) => set({ npzPath }),
  setXKey: (xKey) => set({ xKey }),
  setYKey: (yKey) => set({ yKey }),

  // helper to clear everything
  reset: () =>
    set({
      xPath: '',
      yPath: '',
      npzPath: '',
      xKey: 'X',
      yKey: 'y',
    }),
}));
