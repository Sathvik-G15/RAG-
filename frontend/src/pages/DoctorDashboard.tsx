import { useState } from 'react';
import { useAnalyze } from '../hooks/useAnalyze';
import { Button, Textarea, Select, Card, CardContent, Badge, CardHeader, CardTitle } from '../components/ui';
import { ConfidenceBadge, DisclaimerBanner } from '../components/medical';
import { toast } from 'sonner';
import { type AnalyzeResponse } from '../lib/validations';
import { 
  Stethoscope, FileText, 
  Copy, Check, Activity 
} from 'lucide-react';

const CLINICAL_PRESETS = [
  {
    id: 0,
    label: '🟢 Primary Hypothyroidism — Fatigue, cold intolerance, elevated TSH (Answer)',
    query: 'fatigue, cold intolerance, weight gain, constipation, elevated tsh',
    expectedTier: 'Answer / Home Care',
  },
  {
    id: 1,
    label: '🟢 GERD — Heartburn, acid regurgitation, postprandial (Answer)',
    query: 'heartburn, acid regurgitation, epigastric burning, worse after meals, lying down, gerd',
    expectedTier: 'Answer / Home Care',
  },
  {
    id: 2,
    label: '🟢 Tension Headache — Dull pressure band, no nausea (Answer)',
    query: 'bilateral tension headache, dull pressure, band around head, no nausea',
    expectedTier: 'Answer / Home Care',
  },
  {
    id: 3,
    label: '🟢 Influenza — Acute fever, myalgia, dry cough (Answer)',
    query: 'sudden fever, myalgia, headache, dry cough, fatigue, chills, influenza season',
    expectedTier: 'Answer / Home Care',
  },
  {
    id: 4,
    label: '🟡 Hypertension Outpatient — BP 148/92, asymptomatic (Ask Followup)',
    query: 'high blood pressure, hypertension, bp 148/92, no headache, no chest pain',
    expectedTier: 'Ask Followup',
  },
  {
    id: 5,
    label: '🟡 Migraine Spectrum — Throbbing, mild nausea (Ask Followup)',
    query: 'throbbing headache, mild nausea, recurrent headache, migraine',
    expectedTier: 'Ask Followup',
  },
  {
    id: 6,
    label: '🟡 Cluster Headache — Unilateral orbital pain, lacrimation (Ask Followup)',
    query: 'severe unilateral orbital headache, lacrimation, nasal congestion, cluster headache',
    expectedTier: 'Ask Followup',
  },
  {
    id: 7,
    label: '🔴 Acute Coronary Syndrome — Crushing chest pain, diaphoresis (Escalate Emergency)',
    query: 'crushing chest pain, left arm pain, sweating, diaphoresis, shortness of breath',
    expectedTier: 'Emergency Escalate',
  },
  {
    id: 8,
    label: '🔴 Acute Ischemic Stroke — Facial droop, slurred speech (Escalate Emergency)',
    query: 'sudden facial droop, slurred speech, right arm weakness, acute stroke',
    expectedTier: 'Emergency Escalate',
  },
  {
    id: 9,
    label: '🔴 Anaphylaxis — Urticaria, lip swelling, peanut allergy (Escalate Emergency)',
    query: 'acute urticaria, lip swelling, angioedema, throat tightness, peanut allergy',
    expectedTier: 'Emergency Escalate',
  },
  {
    id: 10,
    label: '🔴 Community-Acquired Pneumonia — Productive cough, pleuritic pain (Escalate Urgent)',
    query: 'fever, productive cough, purulent sputum, pleuritic chest pain, dyspnea, lobar infiltrate',
    expectedTier: 'Urgent Escalate',
  },
  {
    id: 11,
    label: '🔴 Acute Asthma Flare — Wheezing, bronchospasm, inhaler (Escalate Urgent)',
    query: 'acute wheeze, shortness of breath, chest tightness, bronchospasm, inhaler use',
    expectedTier: 'Urgent Escalate',
  },
];

export function DoctorDashboard() {
  const [caseNo, setCaseNo] = useState(0);
  const [customQuery, setCustomQuery] = useState(CLINICAL_PRESETS[0].query);
  const [isCustomMode, setIsCustomMode] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [showEvidence, setShowEvidence] = useState(false);
  const [copied, setCopied] = useState(false);

  const { mutate: analyze, isPending } = useAnalyze();

  const handleRun = () => {
    const query = isCustomMode ? customQuery : (CLINICAL_PRESETS[caseNo]?.query || CLINICAL_PRESETS[0].query);
    analyze(
      { query },
      {
        onSuccess: (response) => {
          setResult(response);
          toast.success('Adaptive Evidence Budgeting complete');
        },
        onError: (error) => {
          toast.error(error.message || 'Pipeline evaluation failed');
        },
      }
    );
  };

  const handleSelectPreset = (id: number) => {
    setCaseNo(id);
    const p = CLINICAL_PRESETS.find(c => c.id === id);
    if (p) {
      setCustomQuery(p.query);
    }
  };

  const handleCopyReport = () => {
    if (!result) return;
    const report = [
      '======================================================',
      'CAAR-CDSS CLINICAL CONSULTATION REPORT',
      '======================================================',
      `Date/Time: ${new Date().toISOString()}`,
      `Primary Diagnostic Assessment: ${result.primary_diagnosis || 'Abstained / Escalate'}`,
      `Calibrated System Confidence: ${Math.round(result.confidence * 100)}%`,
      `Triage Routing: ${result.decision.toUpperCase()} | Acuity Level: ${result.risk_level.toUpperCase()}`,
      `Adaptive Retrieval Budget Used: ${result.retrieval_k_used} chunks across ${result.retrieval_steps} AEB rounds`,
      ...(result.escalated_reason ? [`Escalation Advisory: ${result.escalated_reason}`] : []),
      '',
      'DIFFERENTIAL DIAGNOSES (RANKED):',
      ...result.differential.map((d, i) => `  ${i+1}. ${d.diagnosis} - Probability: ${Math.round(d.probability * 100)}%`),
      '',
      ...(result.followup_questions && result.followup_questions.length > 0 ? [
        'DISCRIMINATING CLINICAL FOLLOW-UP QUESTIONS:',
        ...result.followup_questions.map((q, i) => `  [Q${i+1}] ${q}`),
        ''
      ] : []),
      'RETRIEVED GUIDELINE SOURCES:',
      ...result.evidence.slice(0, 5).map(e => `  • [${e.source}] ${e.title}`),
      '======================================================',
    ].join('\n');

    navigator.clipboard.writeText(report);
    setCopied(true);
    toast.success('Consultation note copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6 max-w-5xl mx-auto">
      <DisclaimerBanner />

      {/* Control Console Card */}
      <Card variant="elevated" padding="lg" className="border-medical-neutral-200 dark:border-medical-neutral-800 shadow-sm">
        <CardHeader>
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-medical-primary-100 dark:bg-medical-primary-950 text-medical-primary-600 dark:text-medical-primary-400">
              <Stethoscope className="h-5 w-5" />
            </span>
            <div>
              <CardTitle className="text-xl sm:text-2xl">Clinician Decision Support Console</CardTitle>
              <p className="text-xs sm:text-sm text-medical-neutral-600 dark:text-medical-neutral-400 mt-0.5">
                Execute Adaptive Evidence Budgeting (AEB) with calibrated confidence and DeBERTa claim verification.
              </p>
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-4 pt-2">
          {/* Preset Selector */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-xs font-semibold text-medical-neutral-700 dark:text-medical-neutral-300">
                Validated Benchmark Clinical Cases (18,480 Chunks)
              </label>
              <button
                type="button"
                onClick={() => setIsCustomMode(!isCustomMode)}
                className="text-xs font-medium text-medical-primary-600 dark:text-medical-primary-400 hover:underline"
              >
                {isCustomMode ? '← Use Case Preset' : '✎ Edit Query Freehand'}
              </button>
            </div>

            {!isCustomMode ? (
              <Select
                options={CLINICAL_PRESETS.map((p) => ({ value: String(p.id), label: p.label }))}
                value={String(caseNo)}
                onChange={(e) => handleSelectPreset(Number(e.target.value))}
                disabled={isPending}
              />
            ) : (
              <Textarea
                rows={3}
                value={customQuery}
                onChange={(e) => setCustomQuery(e.target.value)}
                placeholder="Enter freehand clinical case query..."
                disabled={isPending}
              />
            )}
          </div>

          <div className="flex flex-col sm:flex-row items-center gap-3 pt-2">
            <Button
              onClick={handleRun}
              loading={isPending}
              size="lg"
              className="w-full sm:w-auto shadow-md shadow-medical-primary-500/20"
            >
              {isPending ? 'Executing AEB Retrieval & Verification…' : 'Run Clinical AEB Pipeline'}
            </Button>
            <div className="text-xs text-medical-neutral-500 dark:text-medical-neutral-400">
              Corpus: <span className="font-semibold text-medical-neutral-700 dark:text-medical-neutral-300">ChromaDB + Unified Guidelines</span> · Model: <span className="font-semibold text-medical-neutral-700 dark:text-medical-neutral-300">Phi-3-mini</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Analysis Result Display */}
      {result && (
        <Card variant="elevated" padding="lg" className="animate-fade-in border-medical-neutral-200 dark:border-medical-neutral-800 shadow-md space-y-6">
          <CardHeader className="p-0">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
              <div>
                <span className="text-[11px] font-bold tracking-wider uppercase text-medical-primary-600 dark:text-medical-primary-400">
                  Primary Assessment
                </span>
                <CardTitle className="text-2xl font-extrabold text-medical-neutral-900 dark:text-medical-neutral-100">
                  {result.primary_diagnosis || 'Diagnostic Escalation Required'}
                </CardTitle>
                <p className="text-xs text-medical-neutral-500 dark:text-medical-neutral-400 mt-1">
                  Confidence: <strong className="text-medical-neutral-800 dark:text-medical-neutral-200">{Math.round(result.confidence * 100)}%</strong> · AEB Steps: {result.retrieval_steps} · Evidence Budget: {result.retrieval_k_used} chunks
                </p>
              </div>

              <div className="shrink-0 w-full sm:w-auto">
                <ConfidenceBadge
                  confidence={result.confidence}
                  decision={result.decision}
                  riskLevel={result.risk_level}
                  hallucinationScore={result.hallucination_score}
                />
              </div>
            </div>
          </CardHeader>

          <CardContent className="p-0 space-y-6">
            {/* Follow-up Questions Card */}
            {result.followup_questions && result.followup_questions.length > 0 && (
              <div className="rounded-xl bg-amber-50/90 dark:bg-amber-950/40 border border-amber-300 dark:border-amber-800/80 p-5 space-y-3">
                <div className="flex items-center gap-2">
                  <span className="flex h-6 w-6 items-center justify-center rounded-full bg-amber-200 dark:bg-amber-900/60 text-amber-900 dark:text-amber-200 text-xs font-bold">
                    ?
                  </span>
                  <h3 className="font-bold text-amber-950 dark:text-amber-100 text-base">
                    Clinical Follow-up Clarifying Questions
                  </h3>
                </div>
                <p className="text-xs text-amber-800 dark:text-amber-300">
                  Moderate diagnostic confidence triggered automated triage clarification. Inquire about the following discriminating features:
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

            {/* Differential Diagnoses with Visual Probability Bars */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="font-bold text-sm text-medical-neutral-900 dark:text-medical-neutral-100 flex items-center gap-2">
                  <Activity className="h-4 w-4 text-medical-primary-600" />
                  <span>Ranked Differential Probability</span>
                </h3>
                <span className="text-xs text-medical-neutral-500 font-mono">
                  Normalized across top candidates
                </span>
              </div>

              <div className="rounded-xl border border-medical-neutral-200 dark:border-medical-neutral-700 overflow-hidden bg-white dark:bg-medical-neutral-800">
                <table className="w-full text-xs text-left">
                  <thead className="bg-medical-neutral-50 dark:bg-medical-neutral-900/60 border-b border-medical-neutral-200 dark:border-medical-neutral-700 text-medical-neutral-600 dark:text-medical-neutral-400">
                    <tr>
                      <th className="py-2.5 px-4 font-semibold">Diagnosis</th>
                      <th className="py-2.5 px-4 font-semibold w-64">Probability Distribution</th>
                      <th className="py-2.5 px-4 font-semibold">Guideline Grounding</th>
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
                              <span className="font-semibold">{d.diagnosis}</span>
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            <div className="flex items-center gap-2.5">
                              <div className="flex-1 h-2.5 bg-medical-neutral-100 dark:bg-medical-neutral-700 rounded-full overflow-hidden">
                                <div
                                  className={`h-full rounded-full transition-all duration-500 ${
                                    i === 0 ? 'bg-gradient-to-r from-medical-primary-500 to-medical-primary-600' : 'bg-medical-neutral-400'
                                  }`}
                                  style={{ width: `${Math.max(4, pct)}%` }}
                                />
                              </div>
                              <span className="font-bold text-medical-neutral-800 dark:text-medical-neutral-200 w-12 text-right">
                                {pct}%
                              </span>
                            </div>
                          </td>
                          <td className="py-3 px-4">
                            {d.sources && d.sources.length > 0 ? (
                              <div className="flex flex-wrap gap-1">
                                {d.sources.map((s, si) => (
                                  <Badge key={si} variant="outline" size="sm" className="text-[10px]">
                                    {s}
                                  </Badge>
                                ))}
                              </div>
                            ) : (
                              <span className="text-[11px] text-emerald-700 dark:text-emerald-400 font-medium">
                                ✓ Guideline Entailed
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Evidence Inspector Accordion */}
            <div className="space-y-3 pt-2 border-t border-medical-neutral-200 dark:border-medical-neutral-700">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FileText className="h-4 w-4 text-medical-primary-600" />
                  <h3 className="font-bold text-sm text-medical-neutral-900 dark:text-medical-neutral-100">
                    Inspected Clinical Evidence ({result.evidence.length} Chunks)
                  </h3>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowEvidence(!showEvidence)}
                  className="text-xs"
                >
                  {showEvidence ? 'Collapse Evidence' : 'Inspect Chunks & Source IDs'}
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

            {/* Action Bar */}
            <div className="pt-4 border-t border-medical-neutral-200 dark:border-medical-neutral-700 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <p className="text-[11px] text-medical-neutral-500 dark:text-medical-neutral-400 max-w-xl">
                {result.disclaimer}
              </p>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleCopyReport}
                className="gap-1.5 shrink-0"
              >
                {copied ? <Check className="h-3.5 w-3.5 text-emerald-600" /> : <Copy className="h-3.5 w-3.5" />}
                <span>{copied ? 'Copied' : 'Copy Consultation Report'}</span>
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
