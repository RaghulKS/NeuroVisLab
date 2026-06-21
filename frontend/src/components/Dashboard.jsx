import React from "react";
import { Activity, Brain, Database, FileText, Flame, Gauge, Search, UploadCloud } from "lucide-react";
import HeatmapPanel from "./HeatmapPanel.jsx";
import MetricsPanel from "./MetricsPanel.jsx";
import MonitoringPanel from "./MonitoringPanel.jsx";
import PredictionPanel from "./PredictionPanel.jsx";
import ReportPanel from "./ReportPanel.jsx";
import SimilarCasesPanel from "./SimilarCasesPanel.jsx";
import UploadPanel from "./UploadPanel.jsx";

const panelStyle = {
  border: "1px solid #d8dee9",
  borderRadius: 8,
  padding: 16,
  background: "#ffffff"
};

export default function Dashboard(props) {
  return (
    <main style={{ minHeight: "100vh", background: "#f4f7fb", color: "#172033", fontFamily: "Inter, system-ui, sans-serif" }}>
      <header style={{ borderBottom: "1px solid #d8dee9", background: "#102033", color: "white" }}>
        <div style={{ maxWidth: 1180, margin: "0 auto", padding: "18px 24px", display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 24, letterSpacing: 0 }}>NeuroVisionLab</h1>
            <p style={{ margin: "4px 0 0", color: "#d6e4f5", fontSize: 13 }}>Multimodal medical AI research platform. Not for clinical use.</p>
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <button title="Prepare demo data and train" onClick={props.onRunSetup} style={buttonStyle}><Database size={16} /> Prepare</button>
            <button title="Run image analysis" onClick={props.onRunPrediction} disabled={!props.selectedFile} style={buttonStyle}><Brain size={16} /> Analyze</button>
            <button title="Refresh monitoring" onClick={props.onRefreshMonitoring} style={buttonStyle}><Activity size={16} /> Monitor</button>
            <button title="Export report" onClick={props.onExportReport} disabled={!props.prediction} style={buttonStyle}><FileText size={16} /> Report</button>
          </div>
        </div>
      </header>
      <section style={{ maxWidth: 1180, margin: "0 auto", padding: 24 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 16, flexWrap: "wrap" }}>
          <span style={{ ...badgeStyle, background: "#e7f2ff" }}><Gauge size={15} /> {props.status}</span>
          {props.health && <span style={badgeStyle}>API {props.health.status}</span>}
          {props.error && <span style={{ ...badgeStyle, background: "#fee2e2", color: "#991b1b" }}>{props.error}</span>}
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "minmax(280px, 0.9fr) minmax(320px, 1.1fr)", gap: 16 }}>
          <section style={panelStyle}>
            <SectionTitle icon={<UploadCloud size={18} />} title="Upload" />
            <UploadPanel onFileChange={props.onFileChange} previewUrl={props.previewUrl} selectedFile={props.selectedFile} />
          </section>
          <section style={panelStyle}>
            <SectionTitle icon={<Brain size={18} />} title="Prediction" />
            <PredictionPanel prediction={props.prediction} />
          </section>
          <section style={panelStyle}>
            <SectionTitle icon={<Flame size={18} />} title="Heatmap" />
            <HeatmapPanel explanation={props.explanation} />
          </section>
          <section style={panelStyle}>
            <SectionTitle icon={<Search size={18} />} title="Similar Cases" />
            <SimilarCasesPanel similarCases={props.similarCases} />
          </section>
          <section style={panelStyle}>
            <SectionTitle icon={<Gauge size={18} />} title="Metrics" />
            <MetricsPanel metrics={props.metrics} />
          </section>
          <section style={panelStyle}>
            <SectionTitle icon={<Activity size={18} />} title="Monitoring" />
            <MonitoringPanel drift={props.drift} />
          </section>
          <section style={{ ...panelStyle, gridColumn: "1 / -1" }}>
            <SectionTitle icon={<FileText size={18} />} title="Report" />
            <ReportPanel report={props.report} />
          </section>
        </div>
      </section>
    </main>
  );
}

function SectionTitle({ icon, title }) {
  return <h2 style={{ display: "flex", alignItems: "center", gap: 8, margin: "0 0 12px", fontSize: 16, letterSpacing: 0 }}>{icon}{title}</h2>;
}

const buttonStyle = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  border: "1px solid #9bb4d4",
  background: "#f7fbff",
  color: "#102033",
  borderRadius: 6,
  padding: "8px 10px",
  cursor: "pointer",
  fontSize: 13
};

const badgeStyle = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  borderRadius: 6,
  padding: "7px 10px",
  background: "#edf2f7",
  color: "#172033",
  fontSize: 13
};

