import React from "react";

export default function ReportPanel({ report }) {
  if (!report) return <p style={muted}>Generate a report after prediction and retrieval.</p>;
  return (
    <div>
      <p style={mono}>{report.report_path}</p>
      <pre style={{ maxHeight: 240, overflow: "auto", background: "#f8fafc", border: "1px solid #e2e8f0", borderRadius: 6, padding: 12, whiteSpace: "pre-wrap" }}>
        {report.markdown}
      </pre>
    </div>
  );
}

const muted = { color: "#66758a", fontSize: 13 };
const mono = { fontFamily: "ui-monospace, SFMono-Regular, Consolas, monospace", fontSize: 12, color: "#314055", wordBreak: "break-all" };

