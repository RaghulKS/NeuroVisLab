from __future__ import annotations

from typing import Any, Callable


try:
    from fastapi import APIRouter, File, Form, HTTPException, UploadFile
except ImportError:  # pragma: no cover - used only in minimal environments.
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class UploadFile:
        filename: str = "upload.bin"

        async def read(self) -> bytes:
            return b""

    def File(default: Any = None, **_: Any) -> Any:
        return default

    def Form(default: Any = None, **_: Any) -> Any:
        return default

    class APIRouter:
        def __init__(self, *_: Any, **__: Any) -> None:
            self.routes: list[tuple[str, str, Callable[..., Any]]] = []

        def get(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self.routes.append(("GET", path, func))
                return func

            return decorator

        def post(self, path: str, **_: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                self.routes.append(("POST", path, func))
                return func

            return decorator

