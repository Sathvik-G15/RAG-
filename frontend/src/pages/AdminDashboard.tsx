import { useQuery } from '@tanstack/react-query';
import { api, type Metrics } from '../lib';
import { queryKeys } from '../lib/queryClient';
import { Card, CardHeader, CardTitle, CardContent, Badge, SkeletonCard, Button } from '../components/ui';
import { DisclaimerBanner } from '../components/medical';
import { formatPercent } from '../lib/utils';

const METRIC_LABELS = [
  { key: 'accuracy', label: 'Accuracy', format: formatPercent },
  { key: 'avg_confidence', label: 'Avg Confidence', format: formatPercent },
  { key: 'avg_hallucination', label: 'Avg Hallucination', format: formatPercent },
  { key: 'avg_retrieval_k', label: 'Avg Retrieval K (Budget)', format: (v: number) => v.toFixed(2) },
  { key: 'budget_exhausted_rate', label: 'Budget Exhausted Rate', format: formatPercent },
  { key: 'avg_latency_ms', label: 'Avg Latency (ms)', format: (v: number) => `${v.toFixed(1)}ms` },
] as const;

export function AdminDashboard() {
  const { data: metrics, isLoading, isError, refetch } = useQuery({
    queryKey: queryKeys.metrics(),
    queryFn: () => api.get<Metrics>('/metrics'),
    staleTime: 1000 * 60 * 5,
  });

  if (isLoading) {
    return (
      <div className="space-y-6">
        <DisclaimerBanner />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {METRIC_LABELS.map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <div className="space-y-6">
        <DisclaimerBanner />
        <Card variant="outlined" padding="lg" className="border-medical-danger-200 dark:border-medical-danger-800">
          <div className="flex items-center gap-3">
            <svg className="flex-shrink-0 h-6 w-6 text-medical-danger-600" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <div>
              <p className="font-semibold text-medical-danger-800 dark:text-medical-danger-200">Failed to load metrics</p>
              <p className="text-sm text-medical-danger-600 dark:text-medical-danger-400">Please try again or check the backend connection.</p>
            </div>
          </div>
          <Button variant="primary" onClick={() => refetch()} className="mt-4">
            Retry
          </Button>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      <Card variant="elevated" padding="lg">
        <CardHeader>
          <CardTitle>Admin Metrics Dashboard</CardTitle>
          <p className="text-medical-neutral-600 dark:text-medical-neutral-400">
            AEB pipeline evaluation over {metrics?.n ?? 0} seed clinical queries.
          </p>
        </CardHeader>

        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {METRIC_LABELS.map(({ key, label, format }) => (
              <Card key={key} variant="outlined" padding="md" className="text-center card-hover">
                <div className="text-3xl font-bold text-medical-primary-600 dark:text-medical-primary-400">
                  {metrics ? format(metrics[key as keyof Metrics] as number) : '—'}
                </div>
                <div className="mt-1 text-sm text-medical-neutral-500 dark:text-medical-neutral-400">{label}</div>
              </Card>
            ))}
          </div>

          <div className="mt-6 pt-6 border-t border-medical-neutral-200 dark:border-medical-neutral-700">
            <div className="flex items-center gap-3">
              <Badge variant="info">Method: {metrics?.method ?? 'AEB'}</Badge>
              <Badge variant="outline">Adaptive Evidence Budgeting</Badge>
            </div>
          </div>

          <div className="mt-4 flex justify-end">
            <Button variant="ghost" onClick={() => refetch()} size="sm">
              Refresh Metrics
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}