import { createContext, useContext, useState, useMemo } from "react";

const ModelArtifactContext = createContext(null);

export function ModelArtifactProvider({ children }) {
  const [artifact, setArtifact] = useState(null);

  const value = useMemo(() => ({
    artifact,
    setArtifact,
    clearArtifact: () => setArtifact(null),
  }), [artifact]);

  return (
    <ModelArtifactContext.Provider value={value}>
      {children}
    </ModelArtifactContext.Provider>
  );
}

export function useModelArtifact() {
  const ctx = useContext(ModelArtifactContext);
  if (!ctx) throw new Error("useModelArtifact must be used within ModelArtifactProvider");
  return ctx;
}
