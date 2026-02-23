import { useMutation } from '@tanstack/react-query';
import { inspectData } from '../api/dataApi';
import { inspectProductionData } from '../api/dataApi';

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