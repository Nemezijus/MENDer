import { useQuery } from '@tanstack/react-query';

import { getTuningDefaults } from '../api/schema';

/**
 * Fetch backend-owned tuning request-model defaults.
 *
 * Source of truth: backend (request boundary models).
 * Endpoint: GET /api/v1/schema/tuning-defaults
 */
export function useTuningDefaultsQuery() {
  return useQuery({
    queryKey: ['tuning-defaults'],
    queryFn: getTuningDefaults,
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 60 * 60 * 1000, // 1 hour
    refetchOnWindowFocus: false,
  });
}
