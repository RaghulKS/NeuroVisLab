from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from app.api._compat import HTTPException, UploadFile
from app.config import settings


async def save_upload(file: UploadFile, subdir: str = "incoming") -> Path:
    settings.ensure_directories()
    suffix = Path(getattr(file, "filename", "upload.png") or "upload.png").suffix or ".png"
    target_dir = settings.upload_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{uuid4().hex}{suffix}"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    target_path.write_bytes(content)
    return target_path

