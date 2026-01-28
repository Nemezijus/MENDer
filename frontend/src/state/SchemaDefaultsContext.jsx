// src/state/SchemaDefaultsContext.jsx
import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getAllDefaults } from '../api/schema';

/**
 * Fetch + normalize the /schema/defaults payload using TanStack Query.
 *
 * Backend returns roughly:
 * {
 *   models:     { schema, defaults, meta },
 *   ensembles:  { schema, defaults },
 *   scale:      { schema, defaults },
 *   features:   { schema, defaults },
 *   split:      { ... },
 *   eval:       { schema, defaults },
 *   enums:      { ... }
 * }
 */

function useSchemaDefaultsQuery() {
  return useQuery({
    queryKey: ['schema-defaults'],
    queryFn: getAllDefaults,
    staleTime: 5 * 60 * 1000, // 5 minutes: treat as "fresh"
    cacheTime: 60 * 60 * 1000, // 1 hour cache
    refetchOnWindowFocus: false,
  });
}

export function useSchemaDefaults() {
  const { data, isLoading, isError, error } = useSchemaDefaultsQuery();

  const value = useMemo(() => {
    if (!data) {
      return {
        raw: null,

        models: null,
        ensembles: null,

        scale: null,
        features: null,
        split: null,
        eval: null,

        unsupervised: null,

        enums: {},
        loading: isLoading,
        error: isError ? (error ?? null) : null,

        getModelDefaults: () => null,
        getModelMeta: () => null,
        getCompatibleAlgos: () => [],

        getUnsupervisedEvalDefaults: () => null,

        getEnsembleDefaults: () => null,
        getCompatibleEnsembles: () => [],
      };
    }

    const models = data.models ?? null;
    const ensembles = data.ensembles ?? null;

    const scale = data.scale ?? null;
    const features = data.features ?? null;
    const split = data.split ?? null;
    const evalCfg = data.eval ?? null;

    // Optional unsupervised (clustering) defaults
    const unsupervised = data.unsupervised ?? null;
    const unsupervisedEval = unsupervised?.eval ?? null;
    // unsupervised.run exists mainly to expose schema/defaults for UI, but it
    // will be partially-populated because model selection is required.
    const unsupervisedRun = unsupervised?.run ?? null;

    const enums = data.enums ?? {};

    // ---- models helpers ----------------------------------------------------
    const modelDefaults = models?.defaults ?? {};
    const modelMeta = models?.meta ?? {};

    const getModelDefaults = (algo) => {
      if (!algo) return null;
      return modelDefaults[algo] ?? null;
    };

    const getModelMeta = (algo) => {
      if (!algo) return null;
      return modelMeta[algo] ?? null;
    };

    const getCompatibleAlgos = (task) => {
      const all = Object.keys(modelDefaults);
      if (!task) return all;

      return all.filter((name) => {
        const m = modelMeta[name];
        if (!m) return true; // if no meta, don't exclude
        const t = m.task;
        if (!t) return true;
        // Backwards compatibility: older backends might call this "clustering".
        if (t === 'clustering' && task === 'unsupervised') return true;
        if (Array.isArray(t)) return t.includes(task);
        return t === task;
      });
    };

    // ---- unsupervised helpers --------------------------------------------
    const getUnsupervisedEvalDefaults = () => {
      return unsupervisedEval?.defaults ?? null;
    };

    // ---- ensembles helpers -------------------------------------------------
    const ensembleDefaults = ensembles?.defaults ?? {};

    const getEnsembleDefaults = (kind) => {
      if (!kind) return null;
      return ensembleDefaults[kind] ?? null;
    };

    // For now all ensembles are “compatible”; later we can filter by task
    // if we add metadata like { kind: { task: ... } }.
    const getCompatibleEnsembles = (_task) => Object.keys(ensembleDefaults);

    return {
      raw: data,

      models,
      ensembles,

      scale,
      features,
      split,
      eval: evalCfg,

      unsupervised: unsupervised
        ? {
            ...unsupervised,
            eval: unsupervisedEval,
            run: unsupervisedRun,
          }
        : null,

      enums,
      loading: isLoading,
      error: isError ? (error ?? null) : null,

      getModelDefaults,
      getModelMeta,
      getCompatibleAlgos,

      getUnsupervisedEvalDefaults,

      getEnsembleDefaults,
      getCompatibleEnsembles,
    };
  }, [data, isLoading, isError, error]);

  return value;
}
