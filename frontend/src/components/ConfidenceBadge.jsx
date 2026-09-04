import React from "react";

export default function ConfidenceBadge({ 
  confidence, 
  decision, 
  riskLevel, 
  hallucinationScore,
  showDetails = true 
}) {
  // Determine confidence level and color
  let level = "low";
  let label = "Low Confidence";
  let color = "red";
  let icon = "🔴";

  if (confidence >= 0.85) {
    level = "high";
    label = "High Confidence";
    color = "green";
    icon = "🟢";
  } else if (confidence >= 0.7) {
    level = "moderate";
    label = "Moderate Confidence";
    color = "orange";
    icon = "🟡";
  }

  // Override for escalation
  if (decision === "escalate") {
    label = "Escalated - Clinician Review Required";
    color = "red";
    icon = "🔴";
  }

  const badgeStyles = {
    high: { background: "#dcfce7", color: "#15803d", border: "1px solid #86efac" },
    moderate: { background: "#ffedd5", color: "#c2410c", border: "1px solid #fcd6a5" },
    low: { background: "#fee2e2", color: "#b91c1c", border: "1px solid #fecaca" },
  };

  return (
    <div className="confidence-badge" style={{ 
      ...badgeStyles[level], 
      padding: "10px 14px", 
      borderRadius: "8px",
      display: "flex",
      alignItems: "center",
      gap: "10px",
      fontWeight: 600,
      fontSize: "0.95rem"
    }}>
      <span style={{ fontSize: "1.2rem" }}>{icon}</span>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: "0.95rem" }}>{label}</div>
        {showDetails && (
          <div style={{ fontSize: "0.8rem", opacity: 0.8, marginTop: "2px" }}>
            Confidence: {Math.round(confidence * 100)}% · Decision: {decision} · Risk: {risk_level || riskLevel}
            {hallucination_score && hallucinationScore > 0.3 && ` · ⚠ ${Math.round(hallucinationScore * 100)}% claims unsupported`}
          </div>
        )}
      </div>
    </div>
  );
}