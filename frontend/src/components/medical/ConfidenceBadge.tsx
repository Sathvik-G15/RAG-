import { Badge } from '../ui';
import { clsx } from 'clsx';
import { ShieldCheck, AlertTriangle, Activity } from 'lucide-react';

export type RiskLevel =
  | 'low'
  | 'medium'
  | 'high'
  | 'critical'
  | 'emergency'
  | 'urgent'
  | 'routine'
  | 'home_care';

export type TriageDecision =
  | 'diagnose'
  | 'escalate'
  | 'abstain'
  | 'answer'
  | 'ask_followup';

interface ConfidenceBadgeProps {
  confidence: number;
  decision: TriageDecision;
  riskLevel: RiskLevel;
  hallucinationScore?: number;
  showDetails?: boolean;
}

export function ConfidenceBadge({
  confidence,
  decision,
  riskLevel,
  hallucinationScore,
  showDetails = true,
}: ConfidenceBadgeProps) {
  let variant: 'success' | 'warning' | 'danger' | 'info' = 'info';
  let label = '';
  let icon = '';
  let subtext = '';

  if (decision === 'escalate') {
    variant = 'danger';
    label = 'Escalated - Clinician Review Required';
    icon = '🔴';
    subtext = 'Red-flag presentation or low diagnostic certainty mandates clinician review.';
  } else if (decision === 'ask_followup') {
    variant = 'warning';
    label = 'Follow-up Required';
    icon = '🟡';
    subtext = 'Targeted clinical questions generated to resolve differential ambiguity.';
  } else if (confidence >= 0.85) {
    variant = 'success';
    label = 'High Confidence';
    icon = '🟢';
    subtext = 'Strong guideline evidence and high diagnostic certainty support this assessment.';
  } else if (confidence >= 0.7) {
    variant = 'warning';
    label = 'Moderate Confidence';
    icon = '🟡';
    subtext = 'Moderate guideline agreement; clinical corroboration recommended.';
  } else {
    variant = 'danger';
    label = 'Low Confidence';
    icon = '🔴';
    subtext = 'Diagnostic uncertainty is high; manual clinician examination advised.';
  }

  const riskColors: Record<RiskLevel, 'success' | 'warning' | 'danger' | 'info'> = {
    low: 'success',
    home_care: 'success',
    medium: 'warning',
    routine: 'warning',
    high: 'danger',
    urgent: 'danger',
    critical: 'danger',
    emergency: 'danger',
  };

  const riskLabels: Record<RiskLevel, string> = {
    low: 'Home Care',
    home_care: 'Home Care (Level 1)',
    medium: 'Routine Outpatient',
    routine: 'Routine Outpatient (Level 2)',
    high: 'Urgent Care',
    urgent: 'Urgent Care (Level 3)',
    critical: 'Emergency Dept',
    emergency: 'Emergency Dept (Immediate Level 4)',
  };

  const confPercent = Math.round(confidence * 100);

  // Style cards based on decision
  const containerStyles = {
    success: 'bg-emerald-50/80 dark:bg-emerald-950/40 border-emerald-300 dark:border-emerald-800/80 text-emerald-950 dark:text-emerald-100',
    warning: 'bg-amber-50/80 dark:bg-amber-950/40 border-amber-300 dark:border-amber-800/80 text-amber-950 dark:text-amber-100',
    danger: 'bg-rose-50/80 dark:bg-rose-950/40 border-rose-300 dark:border-rose-800/80 text-rose-950 dark:text-rose-100',
    info: 'bg-blue-50/80 dark:bg-blue-950/40 border-blue-300 dark:border-blue-800/80 text-blue-950 dark:text-blue-100',
  }[variant];

  const barColor = {
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
    danger: 'bg-rose-500',
    info: 'bg-blue-500',
  }[variant];

  return (
    <div className={clsx('rounded-xl border p-4 sm:p-5 shadow-sm transition-all', containerStyles)}>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        {/* Main Status Header */}
        <div className="flex items-start gap-3">
          <span className="text-xl shrink-0 mt-0.5" aria-hidden="true">{icon}</span>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <h3 className="font-bold text-base sm:text-lg tracking-tight">
                {label}
              </h3>
              <Badge variant={variant} size="sm" className="font-semibold uppercase tracking-wider text-[10px]">
                {decision}
              </Badge>
            </div>
            <p className="text-xs text-medical-neutral-600 dark:text-medical-neutral-400 mt-0.5">
              {subtext}
            </p>
          </div>
        </div>

        {/* Confidence Indicator Gauge */}
        <div className="sm:text-right shrink-0">
          <div className="text-xs font-semibold text-medical-neutral-500 dark:text-medical-neutral-400">
            Calibrated Confidence
          </div>
          <div className="text-2xl font-extrabold tracking-tight">
            <span>Confidence: {confPercent}%</span>
          </div>
          <div className="w-full sm:w-36 h-2 bg-black/10 dark:bg-white/10 rounded-full overflow-hidden mt-1">
            <div
              className={clsx('h-full rounded-full transition-all duration-500', barColor)}
              style={{ width: `${Math.min(100, Math.max(5, confPercent))}%` }}
            />
          </div>
        </div>
      </div>

      {showDetails && (
        <div className="mt-3.5 pt-3 border-t border-black/10 dark:border-white/10 flex flex-wrap items-center gap-2 text-xs">
          <div className="flex items-center gap-1.5 font-medium">
            <Activity className="h-3.5 w-3.5 opacity-70" />
            <span>Decision: {decision}</span>
          </div>

          <span className="opacity-40">•</span>

          <div className="flex items-center gap-1.5 font-medium">
            <span>Risk Tier:</span>
            <Badge variant={riskColors[riskLevel] || 'info'} size="sm" className="font-semibold">
              Risk: {riskLevel} ({riskLabels[riskLevel] || riskLevel})
            </Badge>
          </div>

          {hallucinationScore !== undefined && hallucinationScore > 0.3 ? (
            <>
              <span className="opacity-40">•</span>
              <Badge variant="danger" size="sm" className="font-semibold gap-1">
                <AlertTriangle className="h-3 w-3" />
                <span>⚠ {Math.round(hallucinationScore * 100)}% claims unsupported</span>
              </Badge>
            </>
          ) : (
            <>
              <span className="opacity-40">•</span>
              <span className="inline-flex items-center gap-1 text-emerald-700 dark:text-emerald-400 font-medium text-[11px]">
                <ShieldCheck className="h-3.5 w-3.5" />
                Guideline Grounded
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
}
