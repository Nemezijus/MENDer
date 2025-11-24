import { createContext, useContext, useState, useMemo } from 'react';

const ProductionDataContext = createContext(null);

export function ProductionDataProvider({ children }) {
  const [xPath, setXPath] = useState('');
  const [yPath, setYPath] = useState('');
  const [npzPath, setNpzPath] = useState('');
  const [xKey, setXKey] = useState('X');
  const [yKey, setYKey] = useState('y');

  const value = useMemo(
    () => ({
      xPath,
      setXPath,
      yPath,
      setYPath,
      npzPath,
      setNpzPath,
      xKey,
      setXKey,
      yKey,
      setYKey,
      dataReady: Boolean(xPath || npzPath),
    }),
    [xPath, yPath, npzPath, xKey, yKey]
  );

  return (
    <ProductionDataContext.Provider value={value}>
      {children}
    </ProductionDataContext.Provider>
  );
}

export function useProductionDataCtx() {
  const ctx = useContext(ProductionDataContext);
  if (!ctx) {
    throw new Error('useProductionDataCtx must be used within a ProductionDataProvider');
  }
  return ctx;
}
