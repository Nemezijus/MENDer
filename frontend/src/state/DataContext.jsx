// src/state/DataContext.jsx
import { createContext, useContext, useMemo, useState } from 'react';

const DataContext = createContext(null);

export function DataProvider({ children }) {
  // Shared config used by all tabs
  const [xPath, setXPath] = useState('data/classical/wine/wine_features.mat');
  const [yPath, setYPath] = useState('data/classical/wine/wine_labels.mat');
  const [npzPath, setNPZPath] = useState(null);
  const [xKey, setXKey] = useState('X');
  const [yKey, setYKey] = useState('y');

  // Inspect report and readiness
  const [inspectReport, setInspectReport] = useState(null);
  const dataReady = !!inspectReport && inspectReport?.n_samples > 0;

  const value = useMemo(() => ({
    // paths/keys
    xPath, setXPath,
    yPath, setYPath,
    npzPath, setNPZPath,
    xKey, setXKey,
    yKey, setYKey,

    // report
    inspectReport, setInspectReport,

    // derived
    dataReady,
  }), [xPath, yPath, npzPath, xKey, yKey, inspectReport, dataReady]);

  return <DataContext.Provider value={value}>{children}</DataContext.Provider>;
}

export function useDataCtx() {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error('useDataCtx must be used within DataProvider');
  return ctx;
}
