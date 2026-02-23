import { useMutation } from '@tanstack/react-query';

import { inspectData, inspectProductionData } from '../api/dataApi.js';

export function useInspectDataMutation() {
  return useMutation({
    mutationKey: ['inspect-data'],
    mutationFn: (payload) => inspectData(payload),
  });
}

export function useInspectProductionDataMutation() {
  return useMutation({
    mutationKey: ['inspect-production-data'],
    mutationFn: (payload) => inspectProductionData(payload),
  });
}
