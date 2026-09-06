import { useQuery } from '@tanstack/react-query';
import { api, type CorpusStats as CorpusStatsType } from '../lib';
import { queryKeys } from '../lib/queryClient';
import { Card, CardHeader, CardTitle, CardContent, Table, SkeletonCard, SkeletonTable, Button } from '../components/ui';
import { DisclaimerBanner } from '../components/medical';
import { formatNumber, formatPercent, formatDateTime } from '../lib/utils';

export function CorpusStats() {
  const { data: stats, isLoading, isError, refetch } = useQuery({
    queryKey: queryKeys.corpusStats(),
    queryFn: () => api.get<CorpusStatsType>('/corpus/stats'),
    staleTime: 1000 * 60 * 10,
  });

  const specialtyEntries = stats
    ? Object.entries(stats.specialty_distribution || {}).sort((a, b) => b[1] - a[1])
    : [];
  const sourceEntries = stats
    ? Object.entries(stats.source_distribution || {}).sort((a, b) => b[1] - a[1])
    : [];

  if (isLoading) {
    return (
      <div className="space-y-6">
        <DisclaimerBanner />
        <Card variant="elevated" padding="lg">
          <CardHeader>
            <CardTitle>Corpus Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <SkeletonCard />
            <SkeletonTable rows={5} columns={3} className="mt-6" />
            <SkeletonTable rows={5} columns={3} className="mt-6" />
          </CardContent>
        </Card>
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
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 001.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
            </svg>
            <div>
              <p className="font-semibold text-medical-danger-800 dark:text-medical-danger-200">Failed to load corpus statistics</p>
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

  if (!stats) {
    return (
      <div className="space-y-6">
        <DisclaimerBanner />
        <Card variant="elevated" padding="lg" className="text-center">
          <p className="text-medical-neutral-500 dark:text-medical-neutral-400">No corpus data available</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      <Card variant="elevated" padding="lg">
        <CardHeader>
          <CardTitle>Corpus Statistics</CardTitle>
          <p className="text-medical-neutral-600 dark:text-medical-neutral-400">
            Overview of the clinical guidelines corpus coverage and distribution.
          </p>
        </CardHeader>

        <CardContent className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Card variant="outlined" padding="md" className="text-center card-hover">
              <div className="text-3xl font-bold text-medical-primary-600 dark:text-medical-primary-400">
                {formatNumber(stats.total_chunks || 0)}
              </div>
              <div className="mt-1 text-sm text-medical-neutral-500 dark:text-medical-neutral-400">Total Chunks</div>
            </Card>
            <Card variant="outlined" padding="md" className="text-center card-hover">
              <div className="text-3xl font-bold text-medical-primary-600 dark:text-medical-primary-400">
                {formatNumber(stats.total_documents || 0)}
              </div>
              <div className="mt-1 text-sm text-medical-neutral-500 dark:text-medical-neutral-400">Total Documents</div>
            </Card>
            <Card variant="outlined" padding="md" className="text-center card-hover">
              <div className="text-3xl font-bold text-medical-primary-600 dark:text-medical-primary-400">
                {specialtyEntries.length}
              </div>
              <div className="mt-1 text-sm text-medical-neutral-500 dark:text-medical-neutral-400">Specialties</div>
            </Card>
            <Card variant="outlined" padding="md" className="text-center card-hover">
              <div className="text-3xl font-bold text-medical-primary-600 dark:text-medical-primary-400">
                {sourceEntries.length}
              </div>
              <div className="mt-1 text-sm text-medical-neutral-500 dark:text-medical-neutral-400">Sources</div>
            </Card>
          </div>

          <div>
            <h3 className="font-semibold text-medical-neutral-900 dark:text-medical-neutral-100 mb-3">
              Specialty Distribution
            </h3>
            {specialtyEntries.length > 0 ? (
              <Table
                columns={[
                  { key: 'specialty', header: 'Specialty' },
                  { key: 'chunks', header: 'Chunks', render: (_row: unknown, _idx: number, cell: unknown) => formatNumber(cell as number) },
                  { key: 'percentage', header: 'Percentage', render: (_row: unknown, _idx: number, cell: unknown) => formatPercent(cell as number) },
                ]}
                data={specialtyEntries.map(([specialty, count]) => ({
                  specialty: specialty || 'Unknown',
                  chunks: count,
                  percentage: count / (stats.total_chunks || 1),
                }))}
                keyExtractor={(row) => row.specialty}
                striped
                hoverable
              />
            ) : (
              <p className="text-medical-neutral-500 dark:text-medical-neutral-400">No specialty data available</p>
            )}
          </div>

          <div>
            <h3 className="font-semibold text-medical-neutral-900 dark:text-medical-neutral-100 mb-3">
              Source Distribution
            </h3>
            {sourceEntries.length > 0 ? (
              <Table
                columns={[
                  { key: 'source', header: 'Source' },
                  { key: 'chunks', header: 'Chunks', render: (_row: unknown, _idx: number, cell: unknown) => formatNumber(cell as number) },
                  { key: 'percentage', header: 'Percentage', render: (_row: unknown, _idx: number, cell: unknown) => formatPercent(cell as number) },
                ]}
                data={sourceEntries.map(([source, count]) => ({
                  source,
                  chunks: count,
                  percentage: count / (stats.total_chunks || 1),
                }))}
                keyExtractor={(row) => row.source}
                striped
                hoverable
              />
            ) : (
              <p className="text-medical-neutral-500 dark:text-medical-neutral-400">No source data available</p>
            )}
          </div>

          {stats.last_updated && (
            <div className="pt-4 border-t border-medical-neutral-200 dark:border-medical-neutral-700 flex items-center justify-between">
              <p className="text-sm text-medical-neutral-500 dark:text-medical-neutral-400">
                Last updated: {formatDateTime(stats.last_updated)}
              </p>
              <Button variant="ghost" size="sm" onClick={() => refetch()}>
                Refresh
              </Button>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}