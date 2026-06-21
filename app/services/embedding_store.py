from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.services.dataset_loader import CaseRecord


@dataclass
class RetrievedCase:
    case_id: str
    label: str | None
    similarity_score: float
    metadata: dict[str, Any]
    explanation: str


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-8)


class EmbeddingStore:
    def __init__(self, embeddings: np.ndarray | None = None, cases: list[dict[str, Any]] | None = None):
        self.embeddings = _normalize(embeddings) if embeddings is not None and len(embeddings) else np.empty((0, 0), dtype=np.float32)
        self.cases = cases or []
        self._backend = "numpy"
        self._faiss_index = None
        self._sklearn_index = None
        if len(self.embeddings):
            self._build_optional_index()

    def _build_optional_index(self) -> None:
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(self.embeddings.shape[1])
            index.add(self.embeddings.astype(np.float32))
            self._faiss_index = index
            self._backend = "faiss"
            return
        except Exception:
            pass
        try:
            from sklearn.neighbors import NearestNeighbors  # type: ignore

            index = NearestNeighbors(metric="cosine")
            index.fit(self.embeddings)
            self._sklearn_index = index
            self._backend = "sklearn"
        except Exception:
            self._backend = "numpy"

    @property
    def backend(self) -> str:
        return self._backend

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[RetrievedCase]:
        if len(self.embeddings) == 0:
            return []
        query = _normalize(query_embedding)[0]
        top_k = min(max(int(top_k), 1), len(self.embeddings))
        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(query.reshape(1, -1).astype(np.float32), top_k)
            ranked = list(zip(indices[0].tolist(), scores[0].tolist()))
        elif self._sklearn_index is not None:
            distances, indices = self._sklearn_index.kneighbors(query.reshape(1, -1), n_neighbors=top_k)
            ranked = [(int(i), float(1.0 - d)) for i, d in zip(indices[0], distances[0])]
        else:
            scores = self.embeddings @ query
            indices = np.argsort(-scores)[:top_k]
            ranked = [(int(i), float(scores[i])) for i in indices]
        output: list[RetrievedCase] = []
        for index, score in ranked:
            case = self.cases[index]
            label = case.get("label")
            metadata = {k: v for k, v in case.items() if k not in {"image_path", "label", "clinical_note"}}
            output.append(
                RetrievedCase(
                    case_id=str(case.get("patient_id") or Path(str(case.get("image_path", "case"))).stem),
                    label=str(label) if label is not None else None,
                    similarity_score=float(score),
                    metadata=metadata,
                    explanation=(
                        "Similarity is based on normalized image texture, intensity, text, and metadata embeddings; "
                        "it is not a clinical match statement."
                    ),
                )
            )
        return output

    def save(self, directory: str | Path) -> Path:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / "embeddings.npy", self.embeddings.astype(np.float32))
        (root / "cases.json").write_text(json.dumps(self.cases, indent=2, default=str), encoding="utf-8")
        (root / "index_metadata.json").write_text(json.dumps({"backend": self.backend, "n_cases": len(self.cases)}, indent=2), encoding="utf-8")
        return root

    @classmethod
    def load(cls, directory: str | Path) -> "EmbeddingStore":
        root = Path(directory)
        embeddings = np.load(root / "embeddings.npy")
        cases = json.loads((root / "cases.json").read_text(encoding="utf-8"))
        return cls(embeddings=embeddings, cases=cases)


def cases_to_dicts(records: list[CaseRecord]) -> list[dict[str, Any]]:
    return [asdict(record) for record in records]

