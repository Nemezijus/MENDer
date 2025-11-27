import { useMutation } from '@tanstack/react-query';
import { inspectData } from '../api/data';

export function useInspectDataMutation() {
  return useMutation({
    mutationKey: ['inspect-data'],
    mutationFn: (payload) => inspectData(payload),
  });
}