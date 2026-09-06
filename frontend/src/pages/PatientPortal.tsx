import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { analyzeRequestSchema, type AnalyzeRequest, type AnalyzeResponse } from '../lib/validations';
import { useAnalyze } from '../hooks/useAnalyze';
import { Button, Textarea, Input, Select, Card, Badge, Table } from '../components/ui';
import { ConfidenceBadge, DisclaimerBanner } from '../components/medical';
import { toast } from 'sonner';
import { formatPercent } from '../lib/utils';

export function PatientPortal() {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [showEvidence, setShowEvidence] = useState(false);

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<AnalyzeRequest>({
    resolver: zodResolver(analyzeRequestSchema),
    defaultValues: {
      query: '',
      age: undefined,
      gender: undefined,
      comorbidities: [],
      medications: [],
      allergies: [],
      vitals: undefined,
    },
  });

  const { mutate: analyze, isPending } = useAnalyze();

  const onSubmit = (data: AnalyzeRequest) => {
    const payload = {
      ...data,
      age: data.age ?? null,
      gender: data.gender ?? null,
      comorbidities: data.comorbidities || [],
      medications: data.medications || [],
      allergies: data.allergies || [],
      vitals: data.vitals ?? null,
    };

    analyze(payload, {
      onSuccess: (response) => {
        setResult(response);
        toast.success('Analysis complete');
      },
      onError: (error) => {
        toast.error(error.message || 'Analysis failed');
      },
    });
  };

  const handleReset = () => {
    reset();
    setResult(null);
    setShowEvidence(false);
  };

  const genderOptions = [
    { value: 'male', label: 'Male' },
    { value: 'female', label: 'Female' },
    { value: 'other', label: 'Other' },
  ];

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      <Card variant="elevated" padding="lg">
        <div className="space-y-6">
          <div>
            <h1 className="text-2xl font-bold text-medical-neutral-900 dark:text-medical-neutral-100">
              Patient Symptom Checker
            </h1>
            <p className="mt-1 text-medical-neutral-600 dark:text-medical-neutral-400">
              Describe symptoms. The system retrieves evidence, reasons, and estimates confidence.
            </p>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <Textarea
              label="Symptoms"
              placeholder="e.g. 55-year-old diabetic male with central chest pain and sweating"
              rows={3}
              {...register('query')}
              error={errors.query?.message}
              disabled={isPending}
            />

            <div className="grid gap-4 sm:grid-cols-3">
              <Input
                label="Age"
                type="number"
                placeholder="Age"
                {...register('age', { valueAsNumber: true })}
                error={errors.age?.message}
                disabled={isPending}
              />
              <Select
                label="Gender"
                options={genderOptions}
                placeholder="Select gender"
                {...register('gender')}
                error={errors.gender?.message}
                disabled={isPending}
              />
              <Input
                label="Comorbidities"
                placeholder="Comma separated (e.g. diabetes, hypertension)"
                {...register('comorbidities', {
                  setValueAs: (value: string) => value.split(',').map((s) => s.trim()).filter(Boolean),
                })}
                error={errors.comorbidities?.message}
                disabled={isPending}
              />
            </div>

            <div className="flex gap-3">
              <Button type="submit" loading={isPending} size="lg" className="flex-1">
                {isPending ? 'Analyzing…' : 'Analyze Symptoms'}
              </Button>
              <Button type="button" variant="ghost" onClick={handleReset} disabled={isPending}>
                Clear
              </Button>
            </div>
          </form>
        </div>
      </Card>

      {result && (
        <Card variant="elevated" padding="lg" className="animate-fade-in">
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <h2 className="text-xl font-bold text-medical-neutral-900 dark:text-medical-neutral-100">
                {result.primary_diagnosis || 'No confident diagnosis'}
              </h2>
              <ConfidenceBadge
                confidence={result.confidence}
                decision={result.decision}
                riskLevel={result.risk_level}
                hallucinationScore={result.hallucination_score}
              />
            </div>

            <div className="flex flex-wrap items-center gap-4 text-sm text-medical-neutral-600 dark:text-medical-neutral-400">
              <span>
                Decision: <strong className="text-medical-neutral-900 dark:text-medical-neutral-100">{result.decision}</strong>
              </span>
              <span>
                Retrieval budget: <strong className="text-medical-neutral-900 dark:text-medical-neutral-100">{result.retrieval_k_used}</strong>
              </span>
              <span>
                Steps: <strong className="text-medical-neutral-900 dark:text-medical-neutral-100">{result.retrieval_steps}</strong>
              </span>
            </div>

            {result.escalated_reason && (
              <div className="rounded-lg bg-medical-danger-50 dark:bg-medical-danger-900/30 border border-medical-danger-200 dark:border-medical-danger-800 p-4">
                <div className="flex gap-3">
                  <svg className="flex-shrink-0 h-5 w-5 text-medical-danger-600" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <p className="text-medical-danger-800 dark:text-medical-danger-200">{result.escalated_reason}</p>
                </div>
              </div>
            )}

            {result.hallucination_score > 0.3 && (
              <div className="rounded-lg bg-medical-warning-50 dark:bg-medical-warning-900/30 border border-medical-warning-200 dark:border-medical-warning-800 p-4">
                <div className="flex gap-3">
                  <svg className="flex-shrink-0 h-5 w-5 text-medical-warning-600" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                    <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <p className="text-medical-warning-800 dark:text-medical-warning-200">
                    {Math.round(result.hallucination_score * 100)}% of claims not supported by retrieved evidence.
                  </p>
                </div>
              </div>
            )}

            <div>
              <h3 className="font-semibold text-medical-neutral-900 dark:text-medical-neutral-100 mb-3">
                Differential Diagnosis
              </h3>
              <Table
                columns={[
                  { key: 'diagnosis', header: 'Diagnosis' },
                  { key: 'probability', header: 'Probability', render: (_row: unknown, _idx: number, cell: unknown) => formatPercent(cell as number) },
                  { key: 'sources', header: 'Sources', render: (_row: unknown, _idx: number, cell: unknown) => (cell as string[]).join(', ') || '—' },
                ]}
                data={result.differential}
                keyExtractor={(row) => row.diagnosis}
                striped
                hoverable
                emptyMessage="No differential diagnoses available"
              />
            </div>

            <div>
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-medical-neutral-900 dark:text-medical-neutral-100">
                  Evidence Used ({result.evidence.length})
                </h3>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowEvidence(!showEvidence)}
                >
                  {showEvidence ? 'Hide' : 'Show'} Evidence
                </Button>
              </div>

              {showEvidence && (
                <div className="space-y-3 max-h-96 overflow-y-auto">
                  {result.evidence.map((e, i) => (
                    <div key={i} className="rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800/50 p-4 border border-medical-neutral-200 dark:border-medical-neutral-700">
                      <div className="flex items-start justify-between gap-3 mb-2">
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" size="sm">#{e.rank}</Badge>
                          <span className="font-medium text-medical-neutral-900 dark:text-medical-neutral-100">{e.source}</span>
                          {e.title && <span className="text-medical-neutral-500 dark:text-medical-neutral-400">— {e.title}</span>}
                        </div>
                      </div>
                      <p className="text-sm text-medical-neutral-700 dark:text-medical-neutral-300 whitespace-pre-wrap">{e.text}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="pt-4 border-t border-medical-neutral-200 dark:border-medical-neutral-700">
              <p className="text-sm text-medical-neutral-500 dark:text-medical-neutral-400">{result.disclaimer}</p>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}