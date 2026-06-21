from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.config import settings
from app.services.dataset_loader import CaseRecord, create_synthetic_demo_dataset, load_any_dataset
from app.services.embedding_store import EmbeddingStore, RetrievedCase, cases_to_dicts
from app.services.multimodal_fusion import SimpleTextEncoder
from app.services.preprocessing import compute_image_features, metadata_feature_vector


def retrieval_embedding(record: CaseRecord, text_encoder: SimpleTextEncoder | None = None, image_size: int = 128) -> np.ndarray:
    image_features = compute_image_features(record.image_path, image_size=image_size)
    text_encoder = text_encoder or SimpleTextEncoder(max_features=settings.default_text_dim)
    text_features = text_encoder.transform([record.clinical_note])[0] if text_encoder.vocabulary else np.empty((0,), dtype=np.float32)
    metadata_features = metadata_feature_vector(record)
    return np.concatenate([image_features, text_features, metadata_features]).astype(np.float32)


def build_retrieval_index(
    dataset_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    image_size: int = 128,
) -> dict[str, Any]:
    splits = load_any_dataset(dataset_path or settings.data_dir)
    records = splits.all_records()
    if not records:
        demo_dir = create_synthetic_demo_dataset()
        splits = load_any_dataset(demo_dir)
        records = splits.all_records()
    text_encoder = SimpleTextEncoder(max_features=settings.default_text_dim).fit([record.clinical_note for record in records])
    embeddings = np.vstack([retrieval_embedding(record, text_encoder=text_encoder, image_size=image_size) for record in records])
    index_dir = Path(output_dir or settings.artifacts_dir / "retrieval_index")
    store = EmbeddingStore(embeddings=embeddings, cases=cases_to_dicts(records))
    store.save(index_dir)
    (index_dir / "text_vocabulary.txt").write_text("\n".join(text_encoder.vocabulary), encoding="utf-8")
    return {
        "index_dir": str(index_dir),
        "backend": store.backend,
        "n_cases": len(records),
        "classes": splits.classes,
        "disclaimer": settings.disclaimer,
    }


def _load_text_encoder(index_dir: Path) -> SimpleTextEncoder:
    vocab_path = index_dir / "text_vocabulary.txt"
    vocabulary = vocab_path.read_text(encoding="utf-8").splitlines() if vocab_path.exists() else []
    return SimpleTextEncoder(max_features=settings.default_text_dim, vocabulary=vocabulary)


def retrieve_similar_cases(
    image_path: str | Path,
    clinical_note: str | None = None,
    metadata: dict[str, Any] | None = None,
    index_dir: str | Path | None = None,
    top_k: int = 5,
    image_size: int = 128,
) -> list[dict[str, Any]]:
    root = Path(index_dir or settings.artifacts_dir / "retrieval_index")
    if not (root / "embeddings.npy").exists():
        build_retrieval_index(output_dir=root, image_size=image_size)
    store = EmbeddingStore.load(root)
    text_encoder = _load_text_encoder(root)
    query_record = CaseRecord(
        image_path=str(image_path),
        label="query",
        clinical_note=clinical_note,
        age=(metadata or {}).get("age"),
        sex=(metadata or {}).get("sex"),
        view_position=(metadata or {}).get("view_position"),
    )
    query = retrieval_embedding(query_record, text_encoder=text_encoder, image_size=image_size)
    results = store.search(query, top_k=top_k)
    return [case.__dict__ for case in results]
