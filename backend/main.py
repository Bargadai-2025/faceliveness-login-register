"""ASGI entry for uvicorn: `uvicorn main:app --reload` (re-exports the app from api)."""
from api import app

__all__ = ["app"]
