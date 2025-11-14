import { createContext, useContext, useState } from 'react';

const RunModelResultsContext = createContext(null);

export function RunModelResultsProvider({ children }) {
  const [result, setResult] = useState(null);

  return (
    <RunModelResultsContext.Provider value={{ result, setResult }}>
      {children}
    </RunModelResultsContext.Provider>
  );
}

export function useRunModelResultsCtx() {
  const ctx = useContext(RunModelResultsContext);
  if (!ctx) {
    throw new Error('useRunModelResultsCtx must be used within RunModelResultsProvider');
  }
  return ctx;
}
