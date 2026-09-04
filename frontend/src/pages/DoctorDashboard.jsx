import React, { useState } from "react";
import { analyze } from "../api";
import ConfidenceBadge from "../components/ConfidenceBadge";

const PRESETS = [
  "55-year-old diabetic male presents with central chest pain, sweating, and shortness of breath.",
  "70-year-old male with productive cough, fever 39C, and crackles on auscultation.",
  "45-year-old woman with dysuria and urinary frequency without flank pain.",
  "30-year-old woman with urticaria, lip swelling, and hypotension after receiving penicillin.",
  "65-year-old man with sudden right-sided weakness and slurred speech.",
];

export default function DoctorDashboard() {
  const [caseNo, setCaseNo] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  async function run() {
    setLoading(true);
    try {
      setResult(await analyze({ query: PRESETS[caseNo] }));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h1>Doctor Review Dashboard</h1>
      <div className="form">
        <select value={caseNo} onChange={(e) => setCaseNo(Number(e.target.value))}>
          {PRESETS.map((p, i) => (
            <option key={i} value={i}>{p.slice(0, 70)}…</option>
          ))}
        </select>
        <button onClick={run} disabled={loading}>
          {loading ? "Running…" : "Run AEB Pipeline"}
        </button>
      </div>

      {result && (
        <div className="card">
          <div className="row between">
            <h2>{result.primary_diagnosis || "Escalated"}</h2>
            <ConfidenceBadge 
              confidence={result.confidence} 
              decision={result.decision} 
              riskLevel={result.risk_level}
              hallucinationScore={result.hallucination_score}
            />
          </div>
          <p>Confidence {Math.round(result.confidence * 100)}% · Steps: {result.retrieval_steps}</p>
          <p className="reasoning">{result.reasoning}</p>

          <h3>Confidence across retrieval rounds</h3>
          <div className="bars">
            {result.confidence_curve.map((c, i) => (
              <div className="bar" key={i}>
                <div className="bar-fill" style={{ height: `${Math.round(c * 100)}%` }} />
                <span>{i + 1}</span>
              </div>
            ))}
          </div>

          <h3>Differential</h3>
          <table>
            <thead>
              <tr><th>Diagnosis</th><th>Probability</th><th>Sources</th></tr>
            </thead>
            <tbody>
              {result.differential.map((d) => (
                <tr key={d.diagnosis}>
                  <td>{d.diagnosis}</td>
                  <td>{Math.round(d.probability * 100)}%</td>
                  <td className="muted">{d.sources.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3>Evidence audit trail</h3>
          {result.evidence.map((e, i) => (
            <div className="evidence" key={i}>
              <div className="muted">rank {e.rank} · {e.source} {e.title && `— ${e.title}`}</div>
              {e.text}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
