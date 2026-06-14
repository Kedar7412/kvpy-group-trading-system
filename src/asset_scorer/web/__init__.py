"""Web dashboard: FastAPI backend + single-page frontend over the ScoreStore."""

from .app import create_app

__all__ = ["create_app"]
