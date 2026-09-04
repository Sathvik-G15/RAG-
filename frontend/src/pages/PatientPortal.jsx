import React, { useState } from "react";
import { analyze } from "../api";
import ConfidenceBadge from "../components/ConfidenceBadge";

export default function PatientPortal() {
  const [query, setQuery] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [comorbidities, setComorbidities] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function run() {
    if (!query.trim()) return;
    setLoading(true);
    setError("");
    try {
      const payload = {
        query,
        age: age ? Number(age) : null,
        gender: gender || null,
        comorbidities: comorbidities
          ? comorbidities.split(",").map((s) => s.trim()).filter(Boolean)
          : [],
      };
      setResult(await analyze(payload));
    } catch (e) {
      setError("Analysis failed. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="panel">
      <h1>Patient Symptom Checker</h1>
      <p className="muted">
        Describe symptoms. The system retrieves evidence, reasons, and estimates confidence.
      </p>
      <div className="form">
        <textarea
          rows={3}
          placeholder="e.g. 55-year-old diabetic male with central chest pain and sweating"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="row">
          <input placeholder="Age" value={age} onChange={(e) => setAge(e.target.value)} />
          <select value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="">Gender</option>
            <option>male</option>
            <option>female</option>
          </select>
          <input
            placeholder="Comorbidities (comma separated)"
            value={comorbidities}
            onChange={(e) => setComorbidities(e.target.value)}
          />
        </div>
        <button onClick={run} disabled={loading}>
          {loading ? "Analyzing…" : "Analyze"}
        </button>
      </div>
      {error && <div className="error">{error}</div>}

      {result && (
        <div className="card result">
          <div className="row between">
            <h2>{result.primary_diagnosis || "No confident diagnosis"}</h2>
          </div>
          <ConfidenceBadge 
            confidence={result.confidence} 
            decision={result.decision} 
            riskLevel={result.risk_level}
            hallucinationScore={result.hallucination_score}
          />
          <p>
            Decision <strong>{result.decision}</strong> · Retrieval budget used{" "}
            <strong>{result.retrieval_k_used}</strong>
          </p>
          {result.escalated_reason && (
            <div className="alert">⚠ {result.escalated_reason}</div>
          )}
          {result.hallucination_score > 0.3 && (
            <div className="alert">
              ⚠ {Math.round(result.hallucination_score * 100)}% of claims not supported by
              retrieved evidence.
            </div>
          )}
          <div className="differential">
            {result.differential.map((d) => (
              <div className="dx" key={d.diagnosis}>
                <span>{d.diagnosis}</span>
                <span>{Math.round(d.probability * 100)}%</span>
              </div>
            ))}
          </div>
          <details>
            <summary>Evidence used ({result.evidence.length})</summary>
            {result.evidence.map((e, i) => (
              <div className="evidence" key={i}>
                <div className="muted">{e.source} {e.title && `— ${e.title}`}</div>
                {e.text}
              </div>
            ))}
          </details>
          <p className="disclaimer">{result.disclaimer}</p>
        </div>
      )}
    </div>
  );
}
