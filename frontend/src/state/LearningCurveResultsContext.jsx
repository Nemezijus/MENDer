// state/LearningCurveResultsContext.jsx
import { createContext, useContext, useState } from 'react';

const LearningCurveResultsContext = createContext(null);

export function LearningCurveResultsProvider({ children }) {
  const [result, setResult] = useState(null);
  const [nSplits, setNSplits] = useState(5);
  const [withinPct, setWithinPct] = useState(0.99); // 99% of peak by default

  const value = {
    result,
    setResult,
    nSplits,
    setNSplits,
    withinPct,
    setWithinPct,
  };

  return (
    <LearningCurveResultsContext.Provider value={value}>
      {children}
    </LearningCurveResultsContext.Provider>
  );
}

export function useLearningCurveResultsCtx() {
  const ctx = useContext(LearningCurveResultsContext);
  if (!ctx) {
    throw new Error('useLearningCurveResultsCtx must be used within LearningCurveResultsProvider');
  }
  return ctx;
}
