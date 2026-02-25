import { useCallback, useState } from 'react';

import { toErrorText } from '../../../shared/utils/errors.js';

/**
 * useTuningRunner
 *
 * Small helper to standardize the async "run" lifecycle across tuning panels:
 * - loading + error state
 * - optional preflight validation
 * - onStart reset hooks
 * - consistent error normalization
 */
export function useTuningRunner() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const run = useCallback(async ({ preflight, onStart, request, onSuccess } = {}) => {
    const preflightMsg = preflight ? preflight() : null;
    if (preflightMsg) {
      setError(String(preflightMsg));
      return;
    }

    setError(null);
    setLoading(true);
    try {
      if (onStart) onStart();
      const data = await request();
      if (onSuccess) onSuccess(data);
    } catch (e) {
      setError(toErrorText(e));
    } finally {
      setLoading(false);
    }
  }, []);

  return {
    loading,
    error,
    setError,
    clearError: () => setError(null),
    run,
  };
}
