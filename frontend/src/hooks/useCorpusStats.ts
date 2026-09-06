import { useQuery } from '@tanstack/react-query';
import { api, type CorpusStats } from '../lib';
import { queryKeys } from '../lib/queryClient';

export function useCorpusStats() {
  return useQuery({
    queryKey: queryKeys.corpusStats(),
    queryFn: () => api.get<CorpusStats>('/corpus/stats'),
    staleTime: 1000 * 60 * 10,
  });
}