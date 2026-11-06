import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { analyzeRequestSchema, type AnalyzeRequest, type AnalyzeResponse } from '../lib/validations';
import { useAnalyze } from '../hooks/useAnalyze';
import { Button, Textarea, Input, Select, Card, Badge } from '../components/ui';
import { ConfidenceBadge, DisclaimerBanner } from '../components/medical';
import { toast } from 'sonner';
import { 
  Sparkles, Stethoscope, Copy, Check, ChevronDown, ChevronUp, 
  AlertCircle, FileText, HeartPulse, Activity
} from 'lucide-react';

interface QuickPreset {
  label: string;
  category: 'answer' | 'followup' | 'escalate';
  query: string;
  age: number;
  gender: string;
  comorbidities?: string[];
  vitals?: {
    systolic_bp?: number;
    diastolic_bp?: number;
    heart_rate?: number;
    oxygen_saturation?: number;
    temperature?: number;
  };
}

const QUICK_PRESETS: QuickPreset[] = [
  {
    label: 'Hypothyroidism (Answer)',
    category: 'answer',
    query: 'fatigue, cold intolerance, weight gain, constipation, elevated tsh',
    age: 40,
    gender: 'female',
    comorbidities: ['hypothyroidism'],
  },
  {
    label: 'GERD Reflux (Answer)',
    category: 'answer',
    query: 'heartburn, acid regurgitation, epigastric burning, worse after meals, lying down, gerd',
    age: 45,
    gender: 'male',
  },
  {
    label: 'Tension Headache (Answer)',
    category: 'answer',
    query: 'bilateral tension headache, dull pressure, band around head, no nausea',
    age: 32,
    gender: 'female',
  },
  {
    label: 'Hypertension (Ask Followup)',
    category: 'followup',
    query: 'high blood pressure, hypertension, bp 148/92, no headache, no chest pain',
    age: 56,
    gender: 'male',
    comorbidities: ['hypertension'],
    vitals: { systolic_bp: 148, diastolic_bp: 92, heart_rate: 76 },
  },
  {
    label: 'Migraine Spectrum (Ask Followup)',
    category: 'followup',
    query: 'throbbing headache, mild nausea, recurrent headache, migraine',
    age: 34,
    gender: 'female',
  },
  {
    label: 'Chest Pain / ACS (Emergency)',
    category: 'escalate',
    query: 'crushing chest pain, left arm pain, sweating, diaphoresis, shortness of breath',
    age: 58,
    gender: 'male',
    vitals: { systolic_bp: 140, heart_rate: 110, oxygen_saturation: 94 },
  },
  {
    label: 'Stroke Symptoms (Emergency)',
    category: 'escalate',
    query: 'sudden facial droop, slurred speech, right arm weakness, acute stroke',
    age: 67,
    gender: 'female',
  },
];

export function PatientPortal() {
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [showEvidence, setShowEvidence] = useState(false);
  const [showVitals, setShowVitals] = useState(false);
  const [copied, setCopied] = useState(false);

  const {
    register,
    handleSubmit,
    setValue,
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

  const handleApplyPreset = (preset: QuickPreset) => {
    setValue('query', preset.query);
    setValue('age', preset.age);
    setValue('gender', preset.gender as 'male' | 'female' | 'other');
    setValue('comorbidities', preset.comorbidities || []);
    if (preset.vitals) {
      setShowVitals(true);
      setValue('vitals', preset.vitals);
    }
    toast.info(`Loaded preset: ${preset.label}`);
  };

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
        toast.success('Clinical decision analysis complete');
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
    setShowVitals(false);
  };

  const handleCopySummary = () => {
    if (!result) return;
    const summary = [
      '==================================================',
      'CAAR-CDSS CLINICAL DECISION SUMMARY',
      '==================================================',
      `Assessment: ${result.primary_diagnosis || 'Abstained / Escalated'}`,
      `Calibrated Confidence: ${Math.round(result.confidence * 100)}%`,
      `Triage Decision: ${result.decision.toUpperCase()}`,
      `Risk Classification: ${result.risk_level.toUpperCase()}`,
      `Guideline Chunks Inspected: ${result.retrieval_k_used} (Steps: ${result.retrieval_steps})`,
      ...(result.escalated_reason ? [`Escalation Reason: ${result.escalated_reason}`] : []),
      '',
      'DIFFERENTIAL DIAGNOSES:',
      ...result.differential.map(d => ` • ${d.diagnosis} (${Math.round(d.probability * 100)}%)`),
      '',
      ...(result.followup_questions && result.followup_questions.length > 0 ? [
        'RECOMMENDED CLINICAL FOLLOW-UP QUESTIONS:',
        ...result.followup_questions.map((q, i) => ` ${i+1}. ${q}`),
        ''
      ] : []),
      'DISCLAIMER:',
      result.disclaimer,
      '==================================================',
    ].join('\n');

    navigator.clipboard.writeText(summary);
    setCopied(true);
    toast.success('Copied clinical summary to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  const genderOptions = [
    { value: 'male', label: 'Male' },
    { value: 'female', label: 'Female' },
    { value: 'other', label: 'Other' },
  ];

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <DisclaimerBanner />

      {/* Main Symptom Checker Card */}
      <Card variant="elevated" padding="lg" className="border-medical-neutral-200 dark:border-medical-neutral-800 shadow-sm">
        <div className="space-y-6">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-medical-primary-100 dark:bg-medical-primary-950 text-medical-primary-600 dark:text-medical-primary-400">
                <HeartPulse className="h-4 w-4" />
              </span>
              <h1 className="text-xl sm:text-2xl font-bold tracking-tight text-medical-neutral-900 dark:text-medical-neutral-100">
                Patient Symptom Checker & Triage
              </h1>
            </div>
            <p className="text-sm text-medical-neutral-600 dark:text-medical-neutral-400">
              Enter clinical symptoms or choose a verified benchmark preset. The AEB pipeline retrieves guidelines, generates diagnostic hypotheses, and provides calibrated triage.
            </p>
          </div>

          {/* Quick Demo Presets Chips */}
          <div className="rounded-xl bg-medical-neutral-100/70 dark:bg-medical-neutral-800/60 p-3.5 border border-medical-neutral-200 dark:border-medical-neutral-700/60">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-medical-neutral-700 dark:text-medical-neutral-300 mb-2">
              <Sparkles className="h-3.5 w-3.5 text-amber-500" />
              <span>Quick Benchmark Presets (Live Tested):</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {QUICK_PRESETS.map((preset) => {
                const badgeColor = {
                  answer: 'hover:border-emerald-500 hover:text-emerald-700 dark:hover:text-emerald-300',
                  followup: 'hover:border-amber-500 hover:text-amber-700 dark:hover:text-amber-300',
                  escalate: 'hover:border-rose-500 hover:text-rose-700 dark:hover:text-rose-300',
                }[preset.category];

                const dotColor = {
                  answer: 'bg-emerald-500',
                  followup: 'bg-amber-500',
                  escalate: 'bg-rose-500',
                }[preset.category];

                return (
                  <button
                    key={preset.label}
                    type="button"
                    onClick={() => handleApplyPreset(preset)}
                    disabled={isPending}
                    className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-white dark:bg-medical-neutral-800 text-medical-neutral-700 dark:text-medical-neutral-300 border border-medical-neutral-200 dark:border-medical-neutral-700 shadow-2xs transition-all ${badgeColor}`}
                  >
                    <span className={`h-1.5 w-1.5 rounded-full ${dotColor}`} />
                    <span>{preset.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <Textarea
              label="Chief Complaint & Symptom Narrative"
              placeholder="e.g. fatigue, cold intolerance, weight gain, constipation, elevated tsh"
              rows={3}
              {...register('query')}
              error={errors.query?.message}
              disabled={isPending}
            />

            <div className="grid gap-3 sm:grid-cols-3">
              <Input
                label="Age (Years)"
                type="number"
                placeholder="e.g. 40"
                {...register('age', { valueAsNumber: true })}
                error={errors.age?.message}
                disabled={isPending}
              />
              <Select
                label="Biological Sex / Gender"
                options={genderOptions}
                placeholder="Select gender"
                {...register('gender')}
                error={errors.gender?.message}
                disabled={isPending}
              />
              <Input
                label="Known Comorbidities"
                placeholder="e.g. diabetes, hypertension"
                {...register('comorbidities', {
                  setValueAs: (value: unknown) => {
                    if (Array.isArray(value)) return value;
                    if (typeof value === 'string') {
                      return value.split(',').map((s) => s.trim()).filter(Boolean);
                    }
                    return [];
                  },
                })}
                error={errors.comorbidities?.message}
                disabled={isPending}
              />
            </div>

            {/* Optional Vitals Toggle */}
            <div className="rounded-lg border border-medical-neutral-200 dark:border-medical-neutral-700/80 overflow-hidden">
              <button
                type="button"
                onClick={() => setShowVitals(!showVitals)}
                className="w-full flex items-center justify-between px-4 py-2.5 bg-medical-neutral-50 dark:bg-medical-neutral-800/50 text-xs font-semibold text-medical-neutral-700 dark:text-medical-neutral-300 hover:bg-medical-neutral-100 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Activity className="h-4 w-4 text-medical-primary-600 dark:text-medical-primary-400" />
                  <span>Clinical Vitals (Optional Objective Measurements)</span>
                </div>
                {showVitals ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
              </button>

              {showVitals && (
                <div className="p-4 grid gap-3 grid-cols-2 sm:grid-cols-4 bg-white dark:bg-medical-neutral-900 border-t border-medical-neutral-200 dark:border-medical-neutral-700">
                  <Input
                    label="Systolic BP (mmHg)"
                    type="number"
                    placeholder="120"
                    {...register('vitals.systolic_bp', { valueAsNumber: true })}
                    disabled={isPending}
                  />
                  <Input
                    label="Diastolic BP (mmHg)"
                    type="number"
                    placeholder="80"
                    {...register('vitals.diastolic_bp', { valueAsNumber: true })}
                    disabled={isPending}
                  />
                  <Input
                    label="Heart Rate (BPM)"
                    type="number"
                    placeholder="72"
                    {...register('vitals.heart_rate', { valueAsNumber: true })}
                    disabled={isPending}
                  />
                  <Input
                    label="SpO2 (%)"
                    type="number"
                    placeholder="98"
                    {...register('vitals.oxygen_saturation', { valueAsNumber: true })}
                    disabled={isPending}
                  />
                </div>
              )}
            </div>

            {/* Submit & Reset actions */}
            <div className="flex flex-col sm:flex-row items-center gap-3 pt-2">
              <Button type="submit" loading={isPending} size="lg" className="w-full sm:flex-1 shadow-md shadow-medical-primary-500/20">
                {isPending ? 'Running Adaptive Retrieval & Calibration…' : 'Analyze Symptoms with CAAR-CDSS'}
              </Button>
              <Button type="button" variant="ghost" onClick={handleReset} disabled={isPending} className="w-full sm:w-auto">
                Clear
              </Button>
            </div>
          </form>
        </div>
      </Card>

      {/* Results Section */}
      {result && (
        <Card variant="elevated" padding="lg" className="animate-fade-in border-medical-neutral-200 dark:border-medical-neutral-800 shadow-md space-y-6">
          {/* Top Decision Card */}
          <div>
            <ConfidenceBadge
              confidence={result.confidence}
              decision={result.decision}
              riskLevel={result.risk_level}
              hallucinationScore={result.hallucination_score}
            />
          </div>

          {/* Quick Metrics Bar */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800/60 p-3 border border-medical-neutral-200 dark:border-medical-neutral-700">
              <span className="text-[11px] font-medium text-medical-neutral-500 dark:text-medical-neutral-400 block">Primary Diagnosis</span>
              <span className="text-sm font-bold text-medical-neutral-900 dark:text-medical-neutral-100 truncate block">
                {result.primary_diagnosis || 'Abstained'}
              </span>
            </div>
            <div className="rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800/60 p-3 border border-medical-neutral-200 dark:border-medical-neutral-700">
              <span className="text-[11px] font-medium text-medical-neutral-500 dark:text-medical-neutral-400 block">Triage Route</span>
              <span className="text-sm font-bold text-medical-neutral-900 dark:text-medical-neutral-100 uppercase block">
                {result.decision}
              </span>
            </div>
            <div className="rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800/60 p-3 border border-medical-neutral-200 dark:border-medical-neutral-700">
              <span className="text-[11px] font-medium text-medical-neutral-500 dark:text-medical-neutral-400 block">Guideline Chunks</span>
              <span className="text-sm font-bold text-medical-neutral-900 dark:text-medical-neutral-100 block">
                {result.retrieval_k_used} chunks ({result.retrieval_steps} steps)
              </span>
            </div>
            <div className="rounded-lg bg-medical-neutral-50 dark:bg-medical-neutral-800/60 p-3 border border-medical-neutral-200 dark:border-medical-neutral-700">
              <span className="text-[11px] font-medium text-medical-neutral-500 dark:text-medical-neutral-400 block">Acuity Level</span>
              <span className="text-sm font-bold text-medical-neutral-900 dark:text-medical-neutral-100 capitalize block">
                {result.risk_level.replace('_', ' ')}
              </span>
            </div>
          </div>

          {/* Escalation Reason Alert (if present) */}
          {result.escalated_reason && (
            <div className="rounded-xl bg-rose-50 dark:bg-rose-950/40 border border-rose-300 dark:border-rose-800/80 p-4">
              <div className="flex items-start gap-3">
                <AlertCircle className="h-5 w-5 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-sm text-rose-950 dark:text-rose-100">
                    Clinical Escalation Required
                  </h4>
                  <p className="text-xs text-rose-800 dark:text-rose-300 mt-1 leading-relaxed">
                    {result.escalated_reason}
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Follow-up Questions Box (if present) */}
          {result.followup_questions && result.followup_questions.length > 0 && (
            <div className="rounded-xl bg-amber-50/90 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800/80 p-5 space-y-3">
              <div className="flex items-center gap-2">
                <span className="flex h-6 w-6 items-center justify-center rounded-full bg-amber-200 dark:bg-amber-900/60 text-amber-900 dark:text-amber-200 text-xs font-bold">
                  ?
                </span>
                <h3 className="font-bold text-amber-950 dark:text-amber-100 text-base">
                  Recommended Follow-Up Clarifying Questions
                </h3>
              </div>
              <p className="text-xs text-amber-800 dark:text-amber-300">
                The decision engine identified diagnostic overlap. Answering these targeted clinical questions will disambiguate the differential:
              </p>
              <div className="space-y-2">
                {result.followup_questions.map((q, idx) => (
                  <div key={idx} className="flex items-start gap-2.5 p-3 rounded-lg bg-white/80 dark:bg-medical-neutral-800/80 border border-amber-200 dark:border-amber-800/60 text-xs text-medical-neutral-800 dark:text-medical-neutral-200">
                    <span className="font-bold text-amber-700 dark:text-amber-400 mt-0.5">#{idx + 1}</span>
                    <span className="leading-relaxed">{q}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Differential Diagnosis Table with Visual Probability Bars */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-bold text-sm text-medical-neutral-900 dark:text-medical-neutral-100 flex items-center gap-2">
                <Stethoscope className="h-4 w-4 text-medical-primary-600" />
                <span>Ranked Differential Diagnosis</span>
              </h3>
              <span className="text-xs text-medical-neutral-500">
                Calculated from guidelines & patient presentation
              </span>
            </div>

            <div className="rounded-xl border border-medical-neutral-200 dark:border-medical-neutral-700 overflow-hidden bg-white dark:bg-medical-neutral-800">
              <table className="w-full text-xs text-left">
                <thead className="bg-medical-neutral-50 dark:bg-medical-neutral-900/60 border-b border-medical-neutral-200 dark:border-medical-neutral-700 text-medical-neutral-600 dark:text-medical-neutral-400">
                  <tr>
                    <th className="py-2.5 px-4 font-semibold">Diagnosis</th>
                    <th className="py-2.5 px-4 font-semibold w-52">Estimated Probability</th>
                    <th className="py-2.5 px-4 font-semibold">Supporting Guidelines</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-medical-neutral-100 dark:divide-medical-neutral-700/60">
                  {result.differential.map((d, i) => {
                    const pct = Math.round(d.probability * 100);
                    return (
                      <tr key={i} className="hover:bg-medical-neutral-50/60 dark:hover:bg-medical-neutral-700/30 transition-colors">
                        <td className="py-3 px-4 font-medium text-medical-neutral-900 dark:text-medical-neutral-100">
                          <div className="flex items-center gap-2">
                            <span className="text-medical-neutral-400 font-mono text-[10px]">#{i + 1}</span>
                            <span>{d.diagnosis}</span>
                          </div>
                        </td>
                        <td className="py-3 px-4">
                          <div className="flex items-center gap-2.5">
                            <div className="flex-1 h-2 bg-medical-neutral-100 dark:bg-medical-neutral-700 rounded-full overflow-hidden">
                              <div
                                className={`h-full rounded-full ${i === 0 ? 'bg-medical-primary-600' : 'bg-medical-neutral-400'}`}
                                style={{ width: `${Math.max(4, pct)}%` }}
                              />
                            </div>
                            <span className="font-bold text-medical-neutral-800 dark:text-medical-neutral-200 w-10 text-right">
                              {pct}%
                            </span>
                          </div>
                        </td>
                        <td className="py-3 px-4 text-medical-neutral-500">
                          {d.sources && d.sources.length > 0 ? (
                            <div className="flex flex-wrap gap-1">
                              {d.sources.map((src, si) => (
                                <Badge key={si} variant="outline" size="sm" className="text-[10px]">
                                  {src}
                                </Badge>
                              ))}
                            </div>
                          ) : (
                            <span className="text-medical-neutral-400">ChromaDB Guidelines</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Evidence Inspector */}
          <div className="space-y-3 pt-2 border-t border-medical-neutral-200 dark:border-medical-neutral-700">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText className="h-4 w-4 text-medical-primary-600" />
                <h3 className="font-bold text-sm text-medical-neutral-900 dark:text-medical-neutral-100">
                  Retrieved Clinical Evidence ({result.evidence.length} Chunks)
                </h3>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowEvidence(!showEvidence)}
                className="text-xs"
              >
                {showEvidence ? 'Hide Evidence' : 'View Retrieved Chunks'}
              </Button>
            </div>

            {showEvidence && (
              <div className="space-y-2.5 max-h-96 overflow-y-auto pr-1">
                {result.evidence.map((e, i) => (
                  <div key={i} className="rounded-xl bg-medical-neutral-50 dark:bg-medical-neutral-850 p-4 border border-medical-neutral-200 dark:border-medical-neutral-700 text-xs space-y-1.5">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <Badge variant="info" size="sm" className="font-mono text-[10px]">
                          Rank #{e.rank}
                        </Badge>
                        <span className="font-bold text-medical-neutral-900 dark:text-medical-neutral-100">
                          {e.title || e.source}
                        </span>
                      </div>
                      <Badge variant="outline" size="sm" className="text-[10px]">
                        {e.source}
                      </Badge>
                    </div>
                    <p className="text-medical-neutral-700 dark:text-medical-neutral-300 leading-relaxed whitespace-pre-wrap font-sans">
                      {e.text}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Action Footer: Copy Summary & Disclaimer */}
          <div className="pt-4 border-t border-medical-neutral-200 dark:border-medical-neutral-700 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <p className="text-[11px] text-medical-neutral-500 dark:text-medical-neutral-400 max-w-xl">
              {result.disclaimer}
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={handleCopySummary}
              className="gap-1.5 shrink-0"
            >
              {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
              <span>{copied ? 'Copied' : 'Copy Clinical Report'}</span>
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
}
