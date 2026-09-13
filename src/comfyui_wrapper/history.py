"""JSONL generation history for skip-exists and least-used weighting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}


def load_history(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def append_history(path: str | Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def compute_generation_hash(
    prompt: str,
    negative: str,
    *,
    steps: int,
    sampler: str,
    cfg: float,
    width: int,
    height: int,
    checkpoint: str = "",
    loras: Iterable[str] | None = None,
    extra: dict | None = None,
) -> str:
    """Hash generation parameters for duplicate detection (excludes seed)."""
    from comfyui_wrapper.workflow import normalize_sampler

    lora_names = sorted({Path(str(n)).stem for n in (loras or []) if n})
    key = {
        "prompt": prompt,
        "negative": negative,
        "steps": int(steps),
        "sampler": normalize_sampler(str(sampler)),
        "cfg": float(cfg),
        "width": int(width),
        "height": int(height),
        "checkpoint": Path(str(checkpoint)).name if checkpoint else "",
        "loras": lora_names,
    }
    if extra:
        key.update(extra)
    blob = json.dumps(key, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def existing_hashes(history: list[dict]) -> set[str]:
    return {r["gen_hash"] for r in history if r.get("gen_hash")}


def prefix_has_output(
    prefix: str,
    output_dir: str | Path | None = None,
    history: list[dict] | None = None,
) -> bool:
    """True if this job's SaveImage prefix already has an image on disk or in history.

    ComfyUI names files ``{prefix}_00001_.png``. Skip-exists uses this so a re-run
    of the same loop item (same filename prefix) is dropped even when a new
    random LoRA would produce a different prompt hash.
    """
    prefix = str(prefix or "").strip()
    if not prefix:
        return False
    for record in history or []:
        rec_prefix = str(record.get("filename_prefix") or "").strip()
        if rec_prefix and rec_prefix == prefix:
            return True
        for name in record.get("files") or []:
            if _name_matches_prefix(Path(str(name)).name, prefix):
                return True
    if not output_dir:
        return False
    folder = Path(output_dir)
    if not folder.is_dir():
        return False
    try:
        entries = folder.iterdir()
    except OSError:
        return False
    for path in entries:
        if not path.is_file():
            continue
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        if _name_matches_prefix(path.name, prefix):
            return True
    return False


def _name_matches_prefix(filename: str, prefix: str) -> bool:
    # ComfyUI: "{prefix}_00001_.png". Require the underscore so "b1" does not
    # match "b10".
    return filename.startswith(prefix + "_")


def frequency_map(history: list[dict]) -> dict[str, int]:
    freq: dict[str, int] = {}
    for record in history:
        for name in record.get("loras") or []:
            freq[name] = freq.get(name, 0) + 1
    return freq
