import React from "react";

export default function PredictionPanel({ prediction }) {
  if (!prediction) return <p style={muted}>Run analysis to view prediction probabilities, confidence, and uncertainty.</p>;
  const probabilities = prediction.probabilities || {};
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
        <strong>{prediction.prediction_label}</strong>
        <span>{Number(prediction.confidence || 0).toFixed(3)}</span>
      </div>
      <div style={{ marginTop: 12 }}>
        {Object.entries(probabilities).map(([label, value]) => (
          <div key={label} style={{ marginBottom: 8 }}>
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13 }}>
              <span>{label}</span><span>{Number(value).toFixed(3)}</span>
            </div>
            <div style={{ height: 8, background: "#e6edf5", borderRadius: 4 }}>
              <div style={{ height: 8, width: `${Math.max(2, Number(value) * 100)}%`, background: "#1f7a8c", borderRadius: 4 }} />
            </div>
          </div>
        ))}
      </div>
      {prediction.uncertainty && (
        <p style={muted}>Review required: {String(prediction.uncertainty.requires_review)}</p>
      )}
    </div>
  );
}

const muted = { color: "#66758a", fontSize: 13 };

