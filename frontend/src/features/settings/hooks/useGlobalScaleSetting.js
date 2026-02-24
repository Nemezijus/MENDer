import { useCallback, useMemo } from 'react';

import { useSchemaDefaults } from '../../../shared/schema/SchemaDefaultsContext.jsx';
import { useSettingsStore } from '../state/useSettingsStore.js';

/**
 * Global scaling is stored as an override only.
 *
 * - When unset (undefined), the engine/schema default is used.
 * - When user explicitly selects the schema default, we clear the override
 *   so payloads remain minimal and future schema updates are respected.
 */
export function useGlobalScaleSetting() {
  const { scale } = useSchemaDefaults();

  const scaleDefaultRaw = scale?.defaults?.method ?? null;
  const scaleDefault = scaleDefaultRaw != null ? String(scaleDefaultRaw) : null;

  const scaleMethodOverride = useSettingsStore((s) => s.scaleMethod);
  const setScaleMethod = useSettingsStore((s) => s.setScaleMethod);

  const effectiveScaleMethod = useMemo(() => {
    return scaleMethodOverride ?? scaleDefault ?? null;
  }, [scaleMethodOverride, scaleDefault]);

  const isOverridden = scaleMethodOverride != null;

  const handleChange = useCallback(
    (v) => {
      const next = v ? String(v) : undefined;

      // Keep store as an override: clear if user selects the schema default.
      if (scaleDefault != null && next === scaleDefault) {
        setScaleMethod(undefined);
        return;
      }

      setScaleMethod(next);
    },
    [scaleDefault, setScaleMethod],
  );

  return {
    scaleDefault,
    effectiveScaleMethod,
    isOverridden,
    setScaleMethod: handleChange,
  };
}
