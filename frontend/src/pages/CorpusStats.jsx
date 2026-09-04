import React, { useEffect, useState } from "react";
import { fetchCorpusStats } from "../api";

export default function CorpusStats() {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchCorpusStats();
        setStats(data);
      } catch (e) {
        setError("Failed to load corpus statistics");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  if (loading) return <div className="card">Loading corpus statistics...</div>;
  if (error) return <div className="card error">{error}</div>;
  if (!stats) return <div className="card">No data available</div>;

  const specialtyEntries = Object.entries(stats.specialty_distribution || {}).sort(
    (a, b) => b[1] - a[1]
  );
  const sourceEntries = Object.entries(stats.source_distribution || {}).sort(
    (a, b) => b[1] - a[1]
  );

  return (
    <div className="panel">
      <h1>Corpus Statistics</h1>
      <div className="grid" style={{ marginTop: "16px" }}>
        <div className="stat">
          <div className="stat-value">{stats.total_chunks || 0}</div>
          <div className="stat-label">Total Chunks</div>
        </div>
        <div className="stat">
          <div className="stat-value">{stats.total_documents || 0}</div>
          <div className="stat-label">Total Documents</div>
        </div>
        <div className="stat">
          <div className="stat-value">{specialtyEntries.length}</div>
          <div className="stat-label">Specialties</div>
        </div>
        <div className="stat">
          <div className="stat-value">{sourceEntries.length}</div>
          <div className="stat-label">Sources</div>
        </div>
      </div>

      <h2 style={{ marginTop: "24px" }}>Specialty Distribution</h2>
      <div className="card">
        {specialtyEntries.length > 0 ? (
          <table>
            <thead>
              <tr><th>Specialty</th><th>Chunks</th><th>Percentage</th></tr>
            </thead>
            <tbody>
              {specialtyEntries.map(([specialty, count]) => (
                <tr key={specialty}>
                  <td>{specialty || "Unknown"}</td>
                  <td>{count}</td>
                  <td>{((count / stats.total_chunks) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">No specialty data available</p>
        )}
      </div>

      <h2 style={{ marginTop: "24px" }}>Source Distribution</h2>
      <div className="card">
        {sourceEntries.length > 0 ? (
          <table>
            <thead>
              <tr><th>Source</th><th>Chunks</th><th>Percentage</th></tr>
            </thead>
            <tbody>
              {sourceEntries.map(([source, count]) => (
                <tr key={source}>
                  <td>{source}</td>
                  <td>{count}</td>
                  <td>{((count / stats.total_chunks) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">No source data available</p>
        )}
      </div>

      {stats.last_updated && (
        <p className="muted" style={{ marginTop: "16px" }}>
          Last updated: {new Date(stats.last_updated).toLocaleString()}
        </p>
      )}
    </div>
  );
}