import { useMutation } from '@tanstack/react-query';
import { api, type AnalyzeRequest, type AnalyzeResponse } from '../lib';

export function useAnalyze() {
  return useMutation({
    mutationFn: (payload: AnalyzeRequest) => api.post<AnalyzeResponse>('/analyze', payload),
    onError: (error) => {
      console.error('Analysis failed:', error);
    },
  });
}

export function useRetrieve() {
  return useMutation({
    mutationFn: (payload: AnalyzeRequest) => api.post('/retrieve', payload),
  });
}