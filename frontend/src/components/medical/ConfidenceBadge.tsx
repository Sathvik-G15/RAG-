import { Badge } from '../ui';

interface ConfidenceBadgeProps {
  confidence: number;
  decision: 'diagnose' | 'escalate' | 'abstain';
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
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

  const riskColors = {
    low: 'success',
    medium: 'warning',
    high: 'danger',
    critical: 'danger',
  } as const;

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
          <Badge variant={riskColors[riskLevel]} size="sm" className="px-1.5 py-0">
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