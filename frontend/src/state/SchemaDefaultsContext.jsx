// src/state/SchemaDefaultsContext.jsx
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
        const data = await getAllDefaults(); // { models:{schema,defaults,meta}, scale, features, split, eval, enums }
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
        // getters
        getModelDefaults: () => null,
        getModelMeta: () => null,
        getScaleDefaults: () => null,
        getFeaturesDefaults: () => null,
        getEvalDefaults: () => null,
        getSplitDefaults: () => null,
        // new helpers
        meta: null,
        algoList: [],
        getCompatibleAlgos: () => [],
      };
    }

    const { models, scale, features, split, eval: evalSection, enums } = payload;
    const meta = models?.meta ?? {};

    // Compute a stable ordering without using hooks inside this memo
    const algoList = (models?.defaults && Object.keys(models.defaults).length > 0)
      ? Object.keys(models.defaults)
      : ['logreg','svm','tree','forest','knn','linreg']; // fallback order

    const getCompatibleAlgos = (task) => {
      if (!task) return algoList; // no filter if task unknown
      return algoList.filter((algo) => {
        const t = meta?.[algo]?.task;
        // if backend didn’t annotate, don’t hide it
        return !t || t === task;
      });
    };

    return {
      loading, error,
      models, scale, features, split, eval: evalSection, enums,

      // handy getters
      getModelDefaults: (algo) => models?.defaults?.[algo] ?? null,
      getModelMeta: (algo) => models?.meta?.[algo] ?? null,
      getScaleDefaults: () => scale?.defaults ?? null,
      getFeaturesDefaults: () => features?.defaults ?? null,
      getEvalDefaults: () => evalSection?.defaults ?? null,
      getSplitDefaults: (mode) =>
        (mode === 'kfold' ? split?.kfold?.defaults : split?.holdout?.defaults) ?? null,

      // helpers for model filtering
      meta,
      algoList,
      getCompatibleAlgos,
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
