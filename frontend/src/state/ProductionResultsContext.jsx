import { createContext, useContext, useState, useMemo } from 'react';

const ProductionResultsContext = createContext(null);

export function ProductionResultsProvider({ children }) {
  const [applyResult, setApplyResult] = useState(null); // ApplyModelResponse
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);

  const value = useMemo(
    () => ({
      applyResult,
      setApplyResult,
      isRunning,
      setIsRunning,
      error,
      setError,
    }),
    [applyResult, isRunning, error]
  );

  return (
    <ProductionResultsContext.Provider value={value}>
      {children}
    </ProductionResultsContext.Provider>
  );
}

export function useProductionResultsCtx() {
  const ctx = useContext(ProductionResultsContext);
  if (!ctx) {
    throw new Error('useProductionResultsCtx must be used within a ProductionResultsProvider');
  }
  return ctx;
}
