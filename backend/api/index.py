"""Vercel entry point.

Vercel's Python runtime serves the ASGI application exported as ``app`` from a
file under ``api/``. All routes are sent here by the rewrite in ``vercel.json``.
"""
import sys
from pathlib import Path

# Make the ``app`` package importable regardless of the function's working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402

__all__ = ["app"]
