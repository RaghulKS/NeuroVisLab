import React from "react";

export default function UploadPanel({ onFileChange, previewUrl, selectedFile }) {
  return (
    <div>
      <input type="file" accept="image/*" onChange={(event) => onFileChange(event.target.files?.[0] || null)} />
      {previewUrl ? (
        <img alt="Selected medical case" src={previewUrl} style={{ width: "100%", maxHeight: 280, objectFit: "contain", marginTop: 12, borderRadius: 6, background: "#111827" }} />
      ) : (
        <div style={{ height: 220, display: "grid", placeItems: "center", border: "1px dashed #9aa9bd", borderRadius: 6, color: "#607086" }}>Select an image</div>
      )}
      {selectedFile && <p style={{ fontSize: 13, color: "#44556b" }}>{selectedFile.name}</p>}
    </div>
  );
}

