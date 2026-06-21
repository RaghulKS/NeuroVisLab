from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    project_name: str = "NeuroVisionLab"
    environment: str = os.getenv("NVL_ENV", "local")
    data_dir: Path = Path(os.getenv("NVL_DATA_DIR", ROOT_DIR / "data"))
    artifacts_dir: Path = Path(os.getenv("NVL_ARTIFACTS_DIR", ROOT_DIR / "artifacts"))
    upload_dir: Path = Path(os.getenv("NVL_UPLOAD_DIR", ROOT_DIR / "uploads"))
    database_url: str = os.getenv("NVL_DATABASE_URL", f"sqlite:///{ROOT_DIR / 'artifacts' / 'neurovisionlab.db'}")
    default_image_size: int = int(os.getenv("NVL_IMAGE_SIZE", "128"))
    default_text_dim: int = int(os.getenv("NVL_TEXT_DIM", "64"))
    default_top_k: int = int(os.getenv("NVL_TOP_K", "5"))
    random_seed: int = int(os.getenv("NVL_RANDOM_SEED", "42"))
    disclaimer: str = (
        "Educational and research demo only. Outputs are not medical advice, "
        "not a diagnosis, and not for clinical use."
    )

    @property
    def sqlite_path(self) -> Path:
        if self.database_url.startswith("sqlite:///"):
            return Path(self.database_url.replace("sqlite:///", "", 1))
        return self.artifacts_dir / "neurovisionlab.db"

    def ensure_directories(self) -> None:
        for path in (self.data_dir, self.artifacts_dir, self.upload_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()

