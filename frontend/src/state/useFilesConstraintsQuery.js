import { useQuery } from '@tanstack/react-query';

import { getFilesConstraints } from '../api/files';

/**
 * Fetch backend-owned upload/data IO constraints.
 *
 * Source of truth: backend (boundary concern).
 * Endpoint: GET /api/v1/files/constraints
 */
export function useFilesConstraintsQuery() {
  return useQuery({
    queryKey: ['files-constraints'],
    queryFn: getFilesConstraints,
    staleTime: 5 * 60 * 1000, // 5 minutes
    cacheTime: 60 * 60 * 1000, // 1 hour
    refetchOnWindowFocus: false,
  });
}
