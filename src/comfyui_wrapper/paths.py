"""Resolve project-relative and combinator-extension paths."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Package lives in src/comfyui_wrapper; project root is two parents up from here,
    or the CWD when installed as a site package."""
    here = Path(__file__).resolve()
    candidate = here.parents[2]
    if (candidate / "config.yaml").exists() or (candidate / "workflows").exists():
        return candidate
    return Path.cwd()


def resolve_path(value: str | Path, *, root: Path | None = None) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (root or project_root()) / path
