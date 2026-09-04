import React, { useEffect, useState } from "react";
import { getMetrics } from "../api";

export default function AdminDashboard() {
  const [m, setM] = useState(null);

  useEffect(() => {
    getMetrics().then(setM);
  }, []);

  if (!m) return <div className="panel">Loading metrics…</div>;

  const rows = [
    ["Accuracy", m.accuracy],
    ["Avg confidence", m.avg_confidence],
    ["Avg hallucination", m.avg_hallucination],
    ["Avg retrieval k (budget)", m.avg_retrieval_k],
    ["Budget exhausted rate", m.budget_exhausted_rate],
    ["Avg latency (ms)", m.avg_latency_ms],
  ];

  return (
    <div className="panel">
      <h1>Admin Metrics Dashboard</h1>
      <p className="muted">AEB pipeline evaluation over {m.n} seed clinical queries.</p>
      <div className="grid">
        {rows.map(([label, value]) => (
          <div className="card stat" key={label}>
            <div className="stat-value">{typeof value === "number" ? value.toFixed(3) : value}</div>
            <div className="stat-label">{label}</div>
          </div>
        ))}
      </div>
      <div className="card">
        <h3>Method</h3>
        <p>{m.method} (Adaptive Evidence Budgeting)</p>
      </div>
    </div>
  );
}
