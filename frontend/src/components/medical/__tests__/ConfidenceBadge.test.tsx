import { render, screen } from '@/test/utils';
import { ConfidenceBadge } from '../ConfidenceBadge';

describe('ConfidenceBadge', () => {
  it('renders high confidence badge', () => {
    render(<ConfidenceBadge confidence={0.9} decision="diagnose" riskLevel="low" />);
    expect(screen.getByText('High Confidence')).toBeInTheDocument();
    expect(screen.getByText('🟢')).toBeInTheDocument();
    expect(screen.getByText('Confidence: 90%')).toBeInTheDocument();
  });

  it('renders moderate confidence badge', () => {
    render(<ConfidenceBadge confidence={0.75} decision="diagnose" riskLevel="medium" />);
    expect(screen.getByText('Moderate Confidence')).toBeInTheDocument();
    expect(screen.getByText('🟡')).toBeInTheDocument();
  });

  it('renders low confidence badge', () => {
    render(<ConfidenceBadge confidence={0.5} decision="abstain" riskLevel="high" />);
    expect(screen.getByText('Low Confidence')).toBeInTheDocument();
    expect(screen.getByText('🔴')).toBeInTheDocument();
  });

  it('renders escalated badge', () => {
    render(<ConfidenceBadge confidence={0.9} decision="escalate" riskLevel="critical" />);
    expect(screen.getByText('Escalated - Clinician Review Required')).toBeInTheDocument();
    expect(screen.getByText('🔴')).toBeInTheDocument();
  });

  it('shows hallucination warning when score > 0.3', () => {
    render(<ConfidenceBadge confidence={0.8} decision="diagnose" riskLevel="low" hallucinationScore={0.5} />);
    expect(screen.getByText('⚠ 50% claims unsupported')).toBeInTheDocument();
  });

  it('hides hallucination warning when score <= 0.3', () => {
    render(<ConfidenceBadge confidence={0.8} decision="diagnose" riskLevel="low" hallucinationScore={0.2} />);
    expect(screen.queryByText(/claims unsupported/)).not.toBeInTheDocument();
  });
});