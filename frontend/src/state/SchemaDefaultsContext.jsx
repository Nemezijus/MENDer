import { createContext, useContext, useEffect, useMemo, useState } from 'react';
import { getAllDefaults } from '../api/schema';

const SchemaDefaultsContext = createContext(null);

export function SchemaDefaultsProvider({ children }) {
  const [payload, setPayload] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const data = await getAllDefaults();
        if (!alive) return;
        setPayload(data);
      } catch (e) {
        if (!alive) return;
        setError(e);
      } finally {
        if (!alive) return;
        setLoading(false);
      }
    })();
    return () => { alive = false; };
  }, []);

  const value = useMemo(() => {
    if (!payload) {
      return {
        loading, error,
        models: null, scale: null, features: null, split: null, eval: null, enums: null,
        getModelDefaults: () => null,
        getModelMeta: () => null,
        getScaleDefaults: () => null,
        getFeaturesDefaults: () => null,
        getEvalDefaults: () => null,
        getSplitDefaults: (mode) => null,
      };
    }
    const { models, scale, features, split, eval: evalSection, enums } = payload;

    return {
      loading, error,
      models, scale, features, split, eval: evalSection, enums,
      // handy getters
      getModelDefaults: (algo) => models?.defaults?.[algo] ?? null,
      getModelMeta: (algo) => models?.meta?.[algo] ?? null,
      getScaleDefaults: () => scale?.defaults ?? null,
      getFeaturesDefaults: () => features?.defaults ?? null,
      getEvalDefaults: () => evalSection?.defaults ?? null,
      getSplitDefaults: (mode) => (mode === 'kfold' ? split?.kfold?.defaults : split?.holdout?.defaults) ?? null,
    };
  }, [payload, loading, error]);

  return (
    <SchemaDefaultsContext.Provider value={value}>
      {children}
    </SchemaDefaultsContext.Provider>
  );
}

export function useSchemaDefaults() {
  const ctx = useContext(SchemaDefaultsContext);
  if (!ctx) throw new Error('useSchemaDefaults must be used within SchemaDefaultsProvider');
  return ctx;
}
