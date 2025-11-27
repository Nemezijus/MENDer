// src/state/SchemaDefaultsContext.jsx
import { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { getAllDefaults } from '../api/schema';

/**
 * Fetch + normalize the /schema/defaults payload using TanStack Query.
 *
 * Backend returns roughly:
 * {
 *   models: { schema, defaults, meta },
 *   scale: { schema, defaults },
 *   features: { schema, defaults },
 *   split: { schema, defaults },
 *   eval: { schema, defaults },
 *   enums: { ... }
 * }
 */

function useSchemaDefaultsQuery() {
  return useQuery({
    queryKey: ['schema-defaults'],
    queryFn: getAllDefaults,
    // tweak as you like; these are reasonable starting points:
    staleTime: 5 * 60 * 1000,     // 5 minutes: treat as "fresh"
    cacheTime: 60 * 60 * 1000,    // 1 hour cache
    refetchOnWindowFocus: false,  // no surprise refetch on tab focus
  });
}

export function useSchemaDefaults() {
  const { data, isLoading, isError, error } = useSchemaDefaultsQuery();

  const value = useMemo(() => {
    if (!data) {
      return {
        raw: null,
        models: null,
        scale: null,
        features: null,
        split: null,
        eval: null,
        enums: {},
        loading: isLoading,
        error: isError ? (error ?? null) : null,
        getModelDefaults: () => null,
        getModelMeta: () => null,
        getCompatibleAlgos: () => [],
      };
    }

    const models = data.models ?? null;
    const scale = data.scale ?? null;
    const features = data.features ?? null;
    const split = data.split ?? null;
    const evalCfg = data.eval ?? null;
    const enums = data.enums ?? {};

    const defaults = models?.defaults ?? {};
    const meta = models?.meta ?? {};

    const getModelDefaults = (algo) => {
      if (!algo) return null;
      return defaults[algo] ?? null;
    };

    const getModelMeta = (algo) => {
      if (!algo) return null;
      return meta[algo] ?? null;
    };

    const getCompatibleAlgos = (task) => {
      const all = Object.keys(defaults);
      if (!task) return all;

      return all.filter((name) => {
        const m = meta[name];
        if (!m) return true; // if no meta, don't exclude
        const t = m.task;
        if (!t) return true;
        if (Array.isArray(t)) return t.includes(task);
        return t === task;
      });
    };

    return {
      raw: data,
      models,
      scale,
      features,
      split,
      eval: evalCfg,
      enums,
      loading: isLoading,
      error: isError ? (error ?? null) : null,
      getModelDefaults,
      getModelMeta,
      getCompatibleAlgos,
    };
  }, [data, isLoading, isError, error]);

  return value;
}
