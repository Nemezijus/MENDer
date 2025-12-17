import { create } from 'zustand';

const initialState = {
  xPath: '',
  yPath: '',
  npzPath: '',
  xKey: 'X',
  yKey: 'y',
  inspectReport: null,

  xDisplay: '',
  yDisplay: '',
  npzDisplay: '',
};

export const useProductionDataStore = create((set) => ({
  // raw state
  ...initialState,

  // setters
  setXPath: (xPath) => set({ xPath }),
  setYPath: (yPath) => set({ yPath }),

  // keep your existing naming convention
  setNpzPath: (npzPath) => set({ npzPath }),

  setXKey: (xKey) => set({ xKey }),
  setYKey: (yKey) => set({ yKey }),

  setInspectReport: (inspectReport) => set({ inspectReport }),

  // NEW setters for UX display
  setXDisplay: (xDisplay) => set({ xDisplay }),
  setYDisplay: (yDisplay) => set({ yDisplay }),
  setNpzDisplay: (npzDisplay) => set({ npzDisplay }),

  // utility
  reset: () => set({ ...initialState }),
}));
