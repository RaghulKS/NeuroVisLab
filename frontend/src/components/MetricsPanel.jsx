import React from "react";

export default function MetricsPanel({ metrics }) {
  if (!metrics?.length) return <p style={muted}>No trained model metrics found.</p>;
  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
      <thead>
        <tr><th align="left">Model</th><th align="right">F1</th><th align="right">ECE</th></tr>
      </thead>
      <tbody>
        {metrics.map((model) => (
          <tr key={model.model_id}>
            <td style={cell}>{model.model_name}</td>
            <td style={cellRight}>{format(model.macro_f1)}</td>
            <td style={cellRight}>{format(model.expected_calibration_error)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function format(value) {
  return value === null || value === undefined ? "n/a" : Number(value).toFixed(3);
}

const muted = { color: "#66758a", fontSize: 13 };
const cell = { borderTop: "1px solid #e2e8f0", padding: "8px 0" };
const cellRight = { ...cell, textAlign: "right" };

