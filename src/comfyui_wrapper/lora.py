"""LoRA filename resolution and activation-text lookup."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

LORA_TAG_RE = re.compile(r"<lora:([^:>]+)(?::([^>]+))?>", re.IGNORECASE)

from comfyui_wrapper.config import WrapperConfig


@dataclass
class LoRARef:
    name: str
    file: str
    weight: float = 1.0
    activation: str = ""
    clip_weight: float | None = None

    def to_dict(self) -> dict:
        data = {
            "name": self.name,
            "file": self.file,
            "weight": self.weight,
            "activation": self.activation,
        }
        if self.clip_weight is not None:
            data["clip_weight"] = self.clip_weight
        return data


def stem(filename: str) -> str:
    return Path(filename).stem


def normalize_name(name: str) -> str:
    return Path(name).stem.strip().lower()


class LoRAResolver:
    """Map combinator/A1111 LoRA stems onto ComfyUI `loras/` filenames."""

    def __init__(
        self,
        files: Iterable[str] | None = None,
        cfg: WrapperConfig | None = None,
        client=None,
    ):
        self.cfg = cfg
        self.client = client
        self.files = [str(f).replace("\\", "/") for f in (files or [])]
        self._by_stem: dict[str, list[str]] = {}
        for f in self.files:
            self._by_stem.setdefault(normalize_name(Path(f).name), []).append(f)

    @classmethod
    def from_client(cls, client, cfg: WrapperConfig | None = None) -> "LoRAResolver":
        from comfyui_wrapper.models import list_loras

        try:
            files = list_loras(cfg, client)
        except Exception:
            files = list(client.list_loras() or [])
        return cls(files, cfg=cfg, client=client)

    def resolve_file(self, name: str) -> str:
        raw = str(name).replace("\\", "/").strip()
        if not raw:
            raise ValueError("Empty LoRA name")
        if raw in self.files:
            return self._comfy_name(raw)
        key = normalize_name(Path(raw).name)
        matches = self._by_stem.get(key) or []
        if len(matches) == 1:
            return self._comfy_name(matches[0])
        if len(matches) > 1:
            # Prefer a file whose relative path name matches exactly.
            exact = [m for m in matches if Path(m).name.lower() == Path(raw).name.lower()]
            if len(exact) == 1:
                return self._comfy_name(exact[0])
            raise ValueError(
                f"Ambiguous LoRA {name!r}; matches: {', '.join(matches)}"
            )
        if self.files:
            # Substring fallback: unique stem containing the query or vice versa.
            contains = [f for f in self.files if key in normalize_name(Path(f).name)]
            if len(contains) == 1:
                return self._comfy_name(contains[0])
            raise ValueError(f"Unknown LoRA {name!r}. ComfyUI has {len(self.files)} LoRA files.")
        if self.cfg is not None:
            from comfyui_wrapper.models import find_local, ensure_available

            local = find_local("loras", raw, self.cfg)
            if local is not None:
                return ensure_available("loras", local.name, self.cfg, self.client)
        # Offline: assume the stem plus .safetensors.
        if raw.lower().endswith((".safetensors", ".ckpt", ".pt", ".sft")):
            return raw
        return f"{raw}.safetensors"

    def _comfy_name(self, filename: str) -> str:
        if self.cfg is None:
            return filename
        from comfyui_wrapper.models import ensure_available

        return ensure_available("loras", filename, self.cfg, self.client)

    def activation_text(self, name: str) -> str:
        stem_name = Path(name).stem
        dirs: list[Path] = []
        if self.cfg:
            dirs.extend(self.cfg.paths.lora_text_dirs())
            if self.cfg.paths.combinator_root:
                webui = self.cfg.paths.combinator_root.parent.parent
                dirs.append(webui / "models" / "Lora")
        for folder in dirs:
            if not folder.exists():
                continue
            json_path = folder / f"{stem_name}.json"
            if json_path.exists():
                try:
                    with json_path.open("r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict):
                        for key in (
                            "activation text",
                            "activation_text",
                            "activation",
                            "trigger",
                            "ss_trigger_words",
                        ):
                            val = data.get(key)
                            if val:
                                if isinstance(val, list):
                                    return ", ".join(str(x) for x in val)
                                return str(val).strip()
                except Exception:
                    pass
            txt_path = folder / f"{stem_name}.txt"
            if txt_path.exists():
                try:
                    return txt_path.read_text(encoding="utf-8").strip()
                except Exception:
                    pass
        return ""

    def resolve(self, name: str, weight: float = 1.0, activation: str = "", clip_weight: float | None = None) -> LoRARef:
        filename = self.resolve_file(name)
        act = (activation or "").strip() or self.activation_text(name) or self.activation_text(filename)
        return LoRARef(
            name=Path(name).stem,
            file=filename,
            weight=float(weight),
            activation=act,
            clip_weight=clip_weight,
        )


def extract_lora_tags(prompt: str) -> tuple[str, list[tuple[str, float]]]:
    """Pull A1111 ``<lora:name:weight>`` tags out of a prompt string."""
    found: list[tuple[str, float]] = []

    def repl(match: re.Match) -> str:
        name = match.group(1).strip()
        raw_weight = match.group(2)
        try:
            weight = float(raw_weight) if raw_weight not in (None, "") else 1.0
        except ValueError:
            weight = 1.0
        found.append((name, weight))
        return ""

    cleaned = LORA_TAG_RE.sub(repl, prompt or "")
    cleaned = re.sub(r"\s+,", ",", cleaned)
    cleaned = re.sub(r",\s*,+", ",", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,")
    return cleaned, found


def parse_lora_flag(value: str) -> tuple[str, float]:
    """Parse 'name', 'name:0.8', or 'name:0.8:0.6' (clip weight ignored here)."""
    parts = value.rsplit(":", 2)
    if len(parts) == 1:
        return value, 1.0
    name, weight_s = parts[0], parts[1]
    try:
        return name, float(weight_s)
    except ValueError:
        return value, 1.0


def refs_from_config_list(items: Iterable, resolver: LoRAResolver) -> list[LoRARef]:
    refs: list[LoRARef] = []
    for item in items or []:
        if isinstance(item, str):
            name, weight = parse_lora_flag(item)
            refs.append(resolver.resolve(name, weight))
        elif isinstance(item, dict):
            refs.append(
                resolver.resolve(
                    item.get("name") or item.get("file") or "",
                    float(item.get("weight", 1.0)),
                    item.get("activation") or "",
                    item.get("clip_weight"),
                )
            )
    return refs
