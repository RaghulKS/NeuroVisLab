const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

async function parseResponse(response) {
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed with ${response.status}`);
  }
  return response.json();
}

export async function getHealth() {
  return parseResponse(await fetch(`${API_BASE}/health`));
}

export async function prepareData() {
  return parseResponse(
    await fetch(`${API_BASE}/data/prepare`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ create_demo_if_missing: true })
    })
  );
}

export async function trainImageModel() {
  return parseResponse(
    await fetch(`${API_BASE}/train/image`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ epochs: 2, image_size: 128, model_name: "dashboard_image_model" })
    })
  );
}

export async function uploadPrediction(file, metadata = {}) {
  const form = new FormData();
  form.append("file", file);
  form.append("metadata_json", JSON.stringify(metadata));
  return parseResponse(await fetch(`${API_BASE}/predict/image`, { method: "POST", body: form }));
}

export async function uploadExplanation(file, metadata = {}) {
  const form = new FormData();
  form.append("file", file);
  form.append("metadata_json", JSON.stringify(metadata));
  return parseResponse(await fetch(`${API_BASE}/explain/image`, { method: "POST", body: form }));
}

export async function retrieveSimilar(file, metadata = {}, topK = 5) {
  const form = new FormData();
  form.append("file", file);
  form.append("metadata_json", JSON.stringify(metadata));
  form.append("top_k", String(topK));
  return parseResponse(await fetch(`${API_BASE}/retrieve/similar`, { method: "POST", body: form }));
}

export async function getModelMetrics() {
  return parseResponse(await fetch(`${API_BASE}/metrics/models`));
}

export async function getDrift() {
  return parseResponse(await fetch(`${API_BASE}/monitoring/drift`));
}

export async function createReport(payload) {
  return parseResponse(
    await fetch(`${API_BASE}/reports/case`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
  );
}

