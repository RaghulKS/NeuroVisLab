import React from "react";

export default function HeatmapPanel({ explanation }) {
  if (!explanation) return <p style={muted}>Heatmap artifacts appear after analysis.</p>;
  return (
    <div>
      <p style={muted}>Method: {explanation.method}</p>
      <p style={mono}>{explanation.side_by_side_path}</p>
      <ul>
        {(explanation.reason_codes || []).map((reason) => <li key={reason}>{reason}</li>)}
      </ul>
    </div>
  );
}

const muted = { color: "#66758a", fontSize: 13 };
const mono = { fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace", fontSize: 12, color: "#314055", wordBreak: "break-all" };

