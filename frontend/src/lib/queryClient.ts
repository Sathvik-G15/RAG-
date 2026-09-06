import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5,
      gcTime: 1000 * 60 * 30,
      retry: (failureCount, error) => {
        if (error instanceof Error && 'status' in error && error.status === 401) {
          return false;
        }
        return failureCount < 2;
      },
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 0,
    },
  },
});

export const queryKeys = {
  analyze: (query: string) => ['analyze', query] as const,
  corpusStats: () => ['corpus', 'stats'] as const,
  metrics: () => ['metrics'] as const,
  queryHistory: (id: string) => ['query', 'history', id] as const,
  evaluationRuns: () => ['evaluation', 'runs'] as const,
  evaluationResults: (id: string) => ['evaluation', 'results', id] as const,
};