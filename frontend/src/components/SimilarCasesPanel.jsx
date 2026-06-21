import React from "react";

export default function SimilarCasesPanel({ similarCases }) {
  if (!similarCases?.length) return <p style={muted}>No retrieved cases yet.</p>;
  return (
    <div style={{ display: "grid", gap: 8 }}>
      {similarCases.map((item) => (
        <div key={`${item.case_id}-${item.similarity_score}`} style={{ border: "1px solid #e2e8f0", borderRadius: 6, padding: 10 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
            <strong>{item.case_id}</strong>
            <span>{Number(item.similarity_score).toFixed(3)}</span>
          </div>
          <p style={muted}>Label: {item.label || "unknown"}</p>
        </div>
      ))}
    </div>
  );
}

const muted = { color: "#66758a", fontSize: 13 };

