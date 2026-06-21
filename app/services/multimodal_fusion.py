from __future__ import annotations

import re
from collections import Counter

import numpy as np

from app.config import settings
from app.services.dataset_loader import CaseRecord
from app.services.preprocessing import feature_matrix, metadata_matrix


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


class SimpleTextEncoder:
    def __init__(self, max_features: int = 64, vocabulary: list[str] | None = None):
        self.max_features = max_features
        self.vocabulary = vocabulary or []

    def fit(self, texts: list[str | None]) -> "SimpleTextEncoder":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(self._tokens(text))
        self.vocabulary = [token for token, _ in counter.most_common(self.max_features)]
        return self

    def transform(self, texts: list[str | None]) -> np.ndarray:
        if not self.vocabulary:
            return np.empty((len(texts), 0), dtype=np.float32)
        vocab_index = {token: i for i, token in enumerate(self.vocabulary)}
        matrix = np.zeros((len(texts), len(self.vocabulary)), dtype=np.float32)
        for row, text in enumerate(texts):
            counts = Counter(self._tokens(text))
            total = max(sum(counts.values()), 1)
            for token, count in counts.items():
                col = vocab_index.get(token)
                if col is not None:
                    matrix[row, col] = count / total
        return matrix

    def fit_transform(self, texts: list[str | None]) -> np.ndarray:
        return self.fit(texts).transform(texts)

    @staticmethod
    def _tokens(text: str | None) -> list[str]:
        return [token.lower() for token in TOKEN_RE.findall(text or "")]


def build_multimodal_features(
    records: list[CaseRecord],
    image_size: int = 128,
    text_encoder: SimpleTextEncoder | None = None,
    fit_text: bool = True,
) -> tuple[np.ndarray, SimpleTextEncoder]:
    text_encoder = text_encoder or SimpleTextEncoder(max_features=settings.default_text_dim)
    image_features = feature_matrix(records, image_size=image_size)
    texts = [record.clinical_note for record in records]
    if fit_text:
        text_features = text_encoder.fit_transform(texts)
    else:
        text_features = text_encoder.transform(texts)
    structured_features = metadata_matrix(records)
    return np.concatenate([image_features, text_features, structured_features], axis=1).astype(np.float32), text_encoder


def evidence_summary_fields(record: CaseRecord, probabilities: dict[str, float]) -> dict[str, object]:
    top_label = max(probabilities, key=probabilities.get) if probabilities else "unknown"
    return {
        "top_model_signal": top_label,
        "clinical_text_available": bool(record.clinical_note),
        "metadata_available": {
            "age": record.age is not None,
            "sex": record.sex is not None,
            "view_position": record.view_position is not None,
        },
        "note": "Evidence fields summarize model inputs and outputs; they are not independent clinical findings.",
    }

