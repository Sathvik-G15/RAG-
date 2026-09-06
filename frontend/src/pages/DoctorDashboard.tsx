import { useState } from 'react';
import { useAnalyze } from '../hooks/useAnalyze';
import { Button, Select, Card, CardContent, Badge, Table, CardHeader, CardTitle } from '../components/ui';
import { ConfidenceBadge, DisclaimerBanner } from '../components/medical';
import { toast } from 'sonner';
import { formatPercent } from '../lib/utils';

const PRESETS = [
  { id: 0, label: '55yo diabetic male: central chest pain, sweating, SOB', query: '55-year-old diabetic male presents with central chest pain, sweating, and shortness of breath.' },
  { id: 1, label: '70yo male: productive cough, fever 39°C, crackles', query: '70-year-old male with productive cough, fever 39C, and crackles on auscultation.' },
  { id: 2, label: '45yo woman: dysuria, urinary frequency, no flank pain', query: '45-year-old woman with dysuria and urinary frequency without flank pain.' },
  { id: 3, label: '30yo woman: urticaria, lip swelling, hypotension post-penicillin', query: '30-year-old woman with urticaria, lip swelling, and hypotension after receiving penicillin.' },
  { id: 4, label: '65yo man: sudden right-sided weakness, slurred speech', query: '65-year-old man with sudden right-sided weakness and slurred speech.' },
] as const;

export function DoctorDashboard() {
  const [caseNo, setCaseNo] = useState(0);
  const [result, setResult] = useState<{
    primary_diagnosis: string | null;
    confidence: number;
    decision: 'diagnose' | 'escalate' | 'abstain';
    risk_level: 'low' | 'medium' | 'high' | 'critical';
    hallucination_score: number;
    retrieval_steps: number;
    retrieval_k_used: number;
    confidence_curve: number[];
    reasoning: string;
    differential: Array<{ diagnosis: string; probability: number; sources: string[] }>;
    evidence: Array<{ rank: number; source: string; title: string | null; text: string }>;
  } | null>(null);

  const { mutate: analyze, isPending } = useAnalyze();

  const handleRun = () => {
    const query = PRESETS[caseNo].query;
    analyze(
      { query },
      {
        onSuccess: (response) => {
          setResult(response);
          toast.success('AEB Pipeline complete');
        },
        onError: (error) => {
          toast.error(error.message || 'Pipeline failed');
        },
      }
    );
  };

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      <Card variant="elevated" padding="lg">
        <CardHeader>
          <CardTitle>Doctor Review Dashboard</CardTitle>
          <p className="text-medical-neutral-600 dark:text-medical-neutral-400">
            Select a clinical case preset and run the AEB pipeline to see adaptive evidence retrieval in action.
          </p>
        </CardHeader>

        <CardContent>
          <div className="space-y-4">
            <Select
              label="Clinical Case"
              options={PRESETS.map((p) => ({ value: String(p.id), label: p.label }))}
              placeholder="Select a case..."
              onChange={(value) => setCaseNo(Number(value))}
              defaultValue={String(caseNo)}
              disabled={isPending}
            />

            <Button onClick={handleRun} loading={isPending} size="lg" className="w-full sm:w-auto">
              {isPending ? 'Running AEB Pipeline…' : 'Run AEB Pipeline'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {result && (
        <Card variant="elevated" padding="lg" className="animate-fade-in space-y-6">
          <CardHeader>
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <CardTitle>{result.primary_diagnosis || 'Escalated'}</CardTitle>
                <p className="text-medical-neutral-600 dark:text-medical-neutral-400">
                  Confidence {Math.round(result.confidence * 100)}% · Steps: {result.retrieval_steps}
                </p>
              </div>
              <ConfidenceBadge
                confidence={result.confidence}
                decision={result.decision}
                riskLevel={result.risk_level as 'low' | 'medium' | 'high' | 'critical'}
                hallucinationScore={result.hallucination_score}
              />
            </div>
          </CardHeader>

          <CardContent className="space-y-6 pt-0">
            {result.reasoning && (
              <div>
                <h3 className="font-semibold text-medical-neutral-900 dark:text-medical-neutral-100 mb-3">
                  Clinical Reasoning
                </h3>
                <div className="rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800/50 p-4 border border-medical-neutral-200 dark:border-medical-neutral-700">
                  <p className="whitespace-pre-wrap text-medical-neutral-700 dark:text-medical-neutral-300">{result.reasoning}</p>
                </div>
              </div>
            )}

            <div>
              <h3 className="font-semibold text-medical-neutral-900 dark:text-medical-neutral-100 mb-3">
                Confidence Across Retrieval Rounds
              </h3>
              <div className="rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800/50 p-4 border border-medical-neutral-200 dark:border-medical-neutral-700">
                <div className="flex items-end gap-2 h-48" role="img" aria-label="Confidence curve across retrieval rounds">
                  {result.confidence_curve.map((c, i) => (
                    <div key={i} className="flex-1 flex flex-col items-center justify-end">
                      <div
                        className="w-full rounded-t bg-medical-primary-600 transition-all duration-300"
                        style={{ height: `${Math.round(c * 100)}%` }}
                        role="img"
                        aria-label={`Round ${i + 1}: ${Math.round(c * 100)}% confidence`}
                      />
                      <span className="mt-2 text-xs text-medical-neutral-500 dark:text-medical-neutral-400">
                        Round {i + 1}
                      </span>
                    </div>
                  ))}
                </div>
                <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
                  <div className="text-center">
                    <p className="text-2xl font-bold text-medical-primary-600">{Math.round(result.confidence_curve[0] * 100)}%</p>
                    <p className="text-xs text-medical-neutral-500">Initial</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-medical-primary-600">{Math.round(result.confidence_curve[result.confidence_curve.length - 1] * 100)}%</p>
                    <p className="text-xs text-medical-neutral-500">Final</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-medical-primary-600">{result.retrieval_steps}</p>
                    <p className="text-xs text-medical-neutral-500">Rounds</p>
                  </div>
                  <div className="text-center">
                    <p className="text-2xl font-bold text-medical-primary-600">{result.retrieval_k_used}</p>
                    <p className="text-xs text-medical-neutral-500">Total K</p>
                  </div>
                </div>
              </div>
            </div>

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
              <h3 className="font-semibold text-medical-neutral-900 dark:text-medical-neutral-100 mb-3">
                Evidence Audit Trail
              </h3>
              <div className="space-y-3 max-h-96 overflow-y-auto">
                {result.evidence.map((e, i) => (
                  <div key={i} className="rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800/50 p-4 border border-medical-neutral-200 dark:border-medical-neutral-700">
                    <div className="flex items-start justify-between gap-3 mb-2">
                      <div className="flex items-center gap-2">
                        <Badge variant="outline" size="sm">Rank {e.rank}</Badge>
                        <span className="font-medium text-medical-neutral-900 dark:text-medical-neutral-100">{e.source}</span>
                        {e.title && <span className="text-medical-neutral-500 dark:text-medical-neutral-400">— {e.title}</span>}
                      </div>
                    </div>
                    <p className="text-sm text-medical-neutral-700 dark:text-medical-neutral-300 whitespace-pre-wrap">{e.text}</p>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}