import React from "react";

export default function MonitoringPanel({ drift }) {
  if (!drift) return <p style={muted}>Refresh monitoring to inspect dataset and drift checks.</p>;
  const checks = drift.drift?.checks || {};
  return (
    <div>
      <p>Status: <strong>{drift.drift?.overall_status}</strong></p>
      {Object.entries(checks).map(([key, value]) => (
        <div key={key} style={{ display: "flex", justifyContent: "space-between", fontSize: 13, borderTop: "1px solid #e2e8f0", padding: "7px 0" }}>
          <span>{key}</span>
          <span>{Number(value).toFixed(3)}</span>
        </div>
      ))}
    </div>
  );
}

const muted = { color: "#66758a", fontSize: 13 };

