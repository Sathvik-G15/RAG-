import { Badge } from '../ui';

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

  if (decision === 'escalate') {
    variant = 'danger';
    label = 'Escalated - Clinician Review Required';
    icon = '🔴';
  } else if (decision === 'ask_followup') {
    variant = 'warning';
    label = 'Follow-up Required';
    icon = '🟡';
  } else if (confidence >= 0.85) {
    variant = 'success';
    label = 'High Confidence';
    icon = '🟢';
  } else if (confidence >= 0.7) {
    variant = 'warning';
    label = 'Moderate Confidence';
    icon = '🟡';
  } else {
    variant = 'danger';
    label = 'Low Confidence';
    icon = '🔴';
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

  return (
    <Badge variant={variant} dot size="md" className="w-full justify-center gap-2">
      <span className="flex items-center gap-2">
        <span aria-hidden="true">{icon}</span>
        <span className="font-medium">{label}</span>
      </span>
      {showDetails && (
        <span className="text-xs opacity-80 flex flex-wrap items-center gap-2">
          <span>Confidence: {Math.round(confidence * 100)}%</span>
          <span>Decision: {decision}</span>
          <Badge variant={riskColors[riskLevel] || 'info'} size="sm" className="px-1.5 py-0">
            Risk: {riskLevel}
          </Badge>
          {hallucinationScore !== undefined && hallucinationScore > 0.3 && (
            <Badge variant="danger" size="sm" className="px-1.5 py-0">
              ⚠ {Math.round(hallucinationScore * 100)}% claims unsupported
            </Badge>
          )}
        </span>
      )}
    </Badge>
  );
}