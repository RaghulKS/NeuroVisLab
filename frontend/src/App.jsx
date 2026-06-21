import React, { useEffect, useState } from "react";
import Dashboard from "./components/Dashboard.jsx";
import {
  createReport,
  getDrift,
  getHealth,
  getModelMetrics,
  prepareData,
  retrieveSimilar,
  trainImageModel,
  uploadExplanation,
  uploadPrediction
} from "./api.js";

export default function App() {
  const [health, setHealth] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [similarCases, setSimilarCases] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [drift, setDrift] = useState(null);
  const [report, setReport] = useState(null);
  const [status, setStatus] = useState("Idle");
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealth().then(setHealth).catch((err) => setError(err.message));
    getModelMetrics().then((data) => setMetrics(data.models || [])).catch(() => {});
  }, []);

  function onFileChange(file) {
    setSelectedFile(file);
    setPrediction(null);
    setExplanation(null);
    setSimilarCases([]);
    setReport(null);
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setPreviewUrl(file ? URL.createObjectURL(file) : null);
  }

  async function runSetup() {
    setError(null);
    setStatus("Preparing data and training fallback models");
    await prepareData();
    await trainImageModel();
    const metricData = await getModelMetrics();
    setMetrics(metricData.models || []);
    setStatus("Model ready");
  }

  async function runPrediction() {
    if (!selectedFile) return;
    setError(null);
    setStatus("Running prediction, explanation, and retrieval");
    const metadata = { view_position: "PA" };
    const pred = await uploadPrediction(selectedFile, metadata);
    const exp = await uploadExplanation(selectedFile, metadata);
    const retrieved = await retrieveSimilar(selectedFile, metadata, 5);
    setPrediction(pred);
    setExplanation(exp);
    setSimilarCases(retrieved.similar_cases || []);
    setStatus("Analysis complete");
  }

  async function refreshMonitoring() {
    setError(null);
    setStatus("Loading monitoring profile");
    const data = await getDrift();
    setDrift(data);
    setStatus("Monitoring loaded");
  }

  async function exportReport() {
    if (!prediction) return;
    const payload = {
      case_id: `dashboard_${Date.now()}`,
      metadata: { source: "dashboard_upload" },
      prediction,
      heatmap_path: explanation?.heatmap_path,
      similar_cases: similarCases
    };
    const generated = await createReport(payload);
    setReport(generated);
  }

  return (
    <Dashboard
      health={health}
      status={status}
      error={error}
      selectedFile={selectedFile}
      previewUrl={previewUrl}
      prediction={prediction}
      explanation={explanation}
      similarCases={similarCases}
      metrics={metrics}
      drift={drift}
      report={report}
      onFileChange={onFileChange}
      onRunSetup={runSetup}
      onRunPrediction={runPrediction}
      onRefreshMonitoring={refreshMonitoring}
      onExportReport={exportReport}
    />
  );
}

