"""Share A1111 / sd.webui models with ComfyUI without copying files.

The wrapper scans ``webui/models`` on disk (inferred from ``combinator_root``
or ``paths.webui_root``) and merges those names with whatever ComfyUI already
lists. If ComfyUI cannot see a file yet, we:

1. Keep ``extra_model_paths.yaml`` pointed at the A1111 folders (needs a ComfyUI
   restart the first time).
2. Create directory junctions / file hardlinks into ComfyUI's model dirs so the
   next filename scan picks them up — no bytes are copied.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

import yaml

from comfyui_wrapper.config import WrapperConfig

MODEL_EXTS = {".safetensors", ".ckpt", ".pt", ".pt2", ".bin", ".pth", ".sft", ".gguf"}

# ComfyUI folder_paths keys -> A1111-relative dirs under webui_root.
WEBUI_LAYOUT: dict[str, tuple[str, ...]] = {
    "checkpoints": ("models/Stable-diffusion",),
    "diffusion_models": ("models/Stable-diffusion", "models/diffusion_models", "models/unet"),
    "loras": ("models/Lora", "models/LyCORIS"),
    "vae": ("models/VAE",),
    "text_encoders": ("models/text_encoders", "models/text_encoder", "models/clip"),
    "embeddings": ("embeddings",),
    "hypernetworks": ("models/hypernetworks",),
    "controlnet": ("models/ControlNet",),
    "upscale_models": ("models/ESRGAN", "models/RealESRGAN", "models/SwinIR"),
}

# ComfyUI folder_paths keys -> dirs under comfyui_root.
COMFY_LAYOUT: dict[str, tuple[str, ...]] = {
    "checkpoints": ("models/checkpoints",),
    "diffusion_models": ("models/diffusion_models", "models/unet"),
    "loras": ("models/loras",),
    "vae": ("models/vae",),
    "text_encoders": ("models/text_encoders", "models/clip"),
    "embeddings": ("models/embeddings",),
    "hypernetworks": ("models/hypernetworks",),
    "controlnet": ("models/controlnet",),
    "upscale_models": ("models/upscale_models",),
}

COMFY_FOLDER = {
    "checkpoints": "checkpoints",
    "diffusion_models": "diffusion_models",
    "loras": "loras",
    "vae": "vae",
    "text_encoders": "text_encoders",
    "embeddings": "embeddings",
    "hypernetworks": "hypernetworks",
    "controlnet": "controlnet",
    "upscale_models": "upscale_models",
}

# Anima split files that live next to A1111 checkpoints / ComfyUI native folders.
ANIMA_TEXT_ENCODER = "qwen_3_06b_base.safetensors"
ANIMA_VAE = "qwen_image_vae.safetensors"

_SHARE_NOTES: list[str] = []
_SHARED = False
_REMOTE_CACHE: dict[str, list[str]] = {}


def reset_share_state() -> None:
    global _SHARED, _SHARE_NOTES
    _SHARED = False
    _SHARE_NOTES = []
    _REMOTE_CACHE.clear()


def infer_webui_root(cfg: WrapperConfig | None) -> Path | None:
    if cfg is not None:
        explicit = getattr(cfg.paths, "webui_root", None)
        if explicit:
            path = Path(explicit)
            if path.is_dir():
                return path
        combo = cfg.paths.combinator_root
        if combo:
            parent = Path(combo)
            # .../webui/extensions/sd-combinator-ext
            if parent.parent.name.lower() == "extensions":
                cand = parent.parent.parent
                if (cand / "models").is_dir():
                    return cand
            if (parent / "models").is_dir():
                return parent
    return None


def infer_comfyui_root(cfg: WrapperConfig | None) -> Path | None:
    candidates: list[Path] = []
    if cfg is not None:
        explicit = getattr(cfg.paths, "comfyui_root", None)
        if explicit:
            candidates.append(Path(explicit))
    env = os.environ.get("COMFYUI_ROOT") or os.environ.get("COMFYUI_PATH")
    if env:
        candidates.append(Path(env))
    home = Path.home()
    candidates.extend(
        [
            home / "ComfyUI_windows_portable" / "ComfyUI",
            home / "ComfyUI",
            home / "Documents" / "ComfyUI",
        ]
    )
    for cand in candidates:
        if not cand:
            continue
        if (cand / "main.py").is_file() or (cand / "models").is_dir():
            return cand
        nested = cand / "ComfyUI"
        if (nested / "main.py").is_file() or (nested / "models").is_dir():
            return nested
    return None


def webui_dirs(cfg: WrapperConfig | None, kind: str) -> list[Path]:
    root = infer_webui_root(cfg)
    if root is None:
        return []
    return [root / rel for rel in WEBUI_LAYOUT.get(kind, ()) if (root / rel).is_dir()]


def comfy_dirs(cfg: WrapperConfig | None, kind: str) -> list[Path]:
    root = infer_comfyui_root(cfg)
    if root is None:
        return []
    return [root / rel for rel in COMFY_LAYOUT.get(kind, ()) if (root / rel).is_dir()]


def scan_files(folders: Iterable[Path]) -> list[tuple[str, Path]]:
    """Return ``(posix-relative-name, absolute-path)`` for model files."""
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for folder in folders:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in MODEL_EXTS:
                continue
            rel = path.relative_to(folder).as_posix()
            key = rel.lower()
            if key in seen:
                continue
            seen.add(key)
            found.append((rel, path))
    found.sort(key=lambda item: item[0].lower())
    return found


def _match_name(folders: list[Path], raw: str) -> Path | None:
    want = Path(raw)
    want_name = want.name.lower()
    want_stem = want.stem.lower()
    want_rel = raw.lower()
    stem_hits: list[Path] = []
    for rel, path in scan_files(folders):
        if rel.lower() == want_rel:
            return path
        if path.name.lower() == want_name:
            return path
        if path.as_posix().replace("\\", "/").lower() == want_rel:
            return path
        if Path(raw).is_absolute():
            try:
                if path.resolve() == Path(raw).resolve():
                    return path
            except OSError:
                pass
        if path.stem.lower() == want_stem:
            stem_hits.append(path)
    if len(stem_hits) == 1:
        return stem_hits[0]
    return None


def find_local(kind: str, name: str, cfg: WrapperConfig | None) -> Path | None:
    raw = str(name or "").replace("\\", "/").strip()
    if not raw:
        return None
    found = _match_name(webui_dirs(cfg, kind), raw)
    if found is not None:
        return found
    found = _match_name(comfy_dirs(cfg, kind), raw)
    if found is not None:
        return found
    abs_path = Path(raw)
    if abs_path.is_file():
        return abs_path
    return None


def merge_names(*groups: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for name in group or []:
            key = str(name).replace("\\", "/").strip()
            if not key:
                continue
            low = key.lower()
            if low in seen:
                continue
            seen.add(low)
            out.append(key)
    return out


def _remote_list(kind: str, client) -> list[str]:
    if client is None:
        return []
    folder = COMFY_FOLDER.get(kind, kind)
    cache_key = f"{id(client)}:{folder}"
    if cache_key in _REMOTE_CACHE:
        return _REMOTE_CACHE[cache_key]
    try:
        names = [str(x).replace("\\", "/") for x in (client.list_models(folder) or [])]
    except Exception:
        names = []
    _REMOTE_CACHE[cache_key] = names
    return names


def list_kind(kind: str, cfg: WrapperConfig | None, client=None) -> list[str]:
    """ComfyUI-visible names first, then any extra files sitting in the A1111 folder."""
    remote = _remote_list(kind, client)
    remote_base = {Path(n).name.lower() for n in remote}
    extra: list[str] = []
    for rel, path in scan_files(webui_dirs(cfg, kind)):
        if rel.lower() in {n.lower() for n in remote}:
            continue
        if path.name.lower() in remote_base:
            continue
        extra.append(rel)
    return merge_names(remote, extra)


def list_checkpoints(cfg: WrapperConfig | None, client=None) -> list[str]:
    return list_kind("checkpoints", cfg, client)


def _disk_models(cfg: WrapperConfig | None, *relative: str) -> list[str]:
    root = infer_comfyui_root(cfg)
    if root is None:
        return []
    names: list[str] = []
    folder = root.joinpath(*relative)
    if not folder.is_dir():
        return names
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in MODEL_EXTS:
            names.append(path.relative_to(folder).as_posix())
    return names


def list_unets(cfg: WrapperConfig | None, client=None) -> list[str]:
    """Checkpoints + diffusion UNETs + GGUF unets (Flux / Krea / H3)."""
    names: list[str] = []
    names.extend(list_kind("checkpoints", cfg, client))
    if client is not None:
        try:
            names.extend(client.list_models("diffusion_models") or [])
        except Exception:
            pass
    names.extend(_disk_models(cfg, "models", "diffusion_models"))
    names.extend(_disk_models(cfg, "models", "unet"))
    return merge_names(names)


def list_text_encoders(cfg: WrapperConfig | None, client=None) -> list[str]:
    names: list[str] = list_kind("text_encoders", cfg, client)
    if client is not None:
        for folder in ("text_encoders", "clip"):
            try:
                names.extend(client.list_models(folder) or [])
            except Exception:
                pass
    names.extend(_disk_models(cfg, "models", "text_encoders"))
    names.extend(_disk_models(cfg, "models", "clip"))
    return merge_names(names)


def list_vaes(cfg: WrapperConfig | None, client=None) -> list[str]:
    names = list_kind("vae", cfg, client)
    names.extend(_disk_models(cfg, "models", "vae"))
    return merge_names(names)


def list_pdd_acc(cfg: WrapperConfig | None, client=None) -> list[str]:
    """PDD Acc files from ComfyUI/models/pdd_acc (not ordinary LoRAs)."""
    names: list[str] = []
    if client is not None:
        try:
            names.extend(client.list_models("pdd_acc") or [])
        except Exception:
            pass
    names.extend(_disk_models(cfg, "models", "pdd_acc"))
    return merge_names(names)


def list_model_patches(cfg: WrapperConfig | None, client=None) -> list[str]:
    """Anima LLLite and other model patches visible to ComfyUI."""
    names: list[str] = []
    if client is not None:
        try:
            names.extend(client.list_models("model_patches") or [])
        except Exception:
            pass
    names.extend(_disk_models(cfg, "models", "model_patches"))
    return merge_names(names)


def list_loras(cfg: WrapperConfig | None, client=None) -> list[str]:
    return list_kind("loras", cfg, client)


def list_embeddings(cfg: WrapperConfig | None, client=None) -> list[str]:
    return list_kind("embeddings", cfg, client)


def list_hypernetworks(cfg: WrapperConfig | None, client=None) -> list[str]:
    return list_kind("hypernetworks", cfg, client)


def huggingface_adetailer_dir() -> Path | None:
    root = Path.home() / ".cache" / "huggingface" / "hub" / "models--Bingsu--adetailer" / "snapshots"
    if not root.is_dir():
        return None
    snaps = [p for p in root.iterdir() if p.is_dir()]
    if not snaps:
        return None
    return max(snaps, key=lambda p: p.stat().st_mtime)


def adetailer_source_dirs(cfg: WrapperConfig | None) -> list[Path]:
    dirs: list[Path] = []
    webui = infer_webui_root(cfg)
    if webui is not None:
        for name in ("adetailer", "ADetailer"):
            folder = webui / "models" / name
            if folder.is_dir() and folder not in dirs:
                dirs.append(folder)
    hub = huggingface_adetailer_dir()
    if hub is not None:
        dirs.append(hub)
    return dirs


def link_adetailer_models(cfg: WrapperConfig | None) -> int:
    """Hardlink A1111 / HF-cache YOLO detectors into ComfyUI ultralytics folders."""
    comfy = infer_comfyui_root(cfg)
    if comfy is None:
        return 0
    bbox_dir = comfy / "models" / "ultralytics" / "bbox"
    segm_dir = comfy / "models" / "ultralytics" / "segm"
    bbox_dir.mkdir(parents=True, exist_ok=True)
    segm_dir.mkdir(parents=True, exist_ok=True)
    linked = 0
    seen: set[str] = set()
    for folder in adetailer_source_dirs(cfg):
        for src in folder.rglob("*"):
            if not src.is_file() or src.suffix.lower() != ".pt":
                continue
            key = src.name.lower()
            if key in seen:
                continue
            dest_dir = segm_dir if "-seg" in key or "_seg" in key else bbox_dir
            dest = dest_dir / src.name
            seen.add(key)
            if dest.exists() or dest.is_symlink():
                continue
            try:
                _hardlink_or_symlink(src, dest)
                linked += 1
            except OSError:
                continue
    return linked


def list_adetailer_models(cfg: WrapperConfig | None, client=None) -> list[str]:
    names: list[str] = []
    if client is not None:
        try:
            info = client.get("/object_info/UltralyticsDetectorProvider")
            required = ((info.get("UltralyticsDetectorProvider") or {}).get("input") or {}).get("required") or {}
            combo = required.get("model_name") or [[]]
            names.extend(str(x) for x in (combo[0] or []) if x)
        except Exception:
            pass
    comfy = infer_comfyui_root(cfg)
    if comfy is not None:
        for prefix, folder in (("bbox", comfy / "models" / "ultralytics" / "bbox"), ("segm", comfy / "models" / "ultralytics" / "segm")):
            if not folder.is_dir():
                continue
            for path in folder.glob("*.pt"):
                names.append(f"{prefix}/{path.name}")
    for folder in adetailer_source_dirs(cfg):
        for path in folder.rglob("*.pt"):
            prefix = "segm" if "-seg" in path.name.lower() else "bbox"
            names.append(f"{prefix}/{path.name}")
    return merge_names(names)


def list_upscalers(cfg: WrapperConfig | None, client=None) -> list[str]:
    names = ["latent"]
    names.extend(list_kind("upscale_models", cfg, client))
    return merge_names(names)


def _ensure_comfy_dir(cfg: WrapperConfig | None, kind: str) -> Path | None:
    dirs = comfy_dirs(cfg, kind)
    if dirs:
        return dirs[0]
    root = infer_comfyui_root(cfg)
    rels = COMFY_LAYOUT.get(kind)
    if root is None or not rels:
        return None
    dest = root / rels[0]
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _is_anima_unet(name: str) -> bool:
    return "anima" in Path(name).name.lower()


def link_anima_assets(cfg: WrapperConfig | None, client=None) -> list[str]:
    """Hardlink Anima UNETs / Qwen TE / Qwen VAE into the folders ComfyUI loaders scan."""
    notes: list[str] = []
    unet_dest = _ensure_comfy_dir(cfg, "diffusion_models")
    linked_unets = 0
    if unet_dest is not None:
        already = {p.name.lower() for p in unet_dest.iterdir() if p.is_file() or p.is_symlink()} if unet_dest.is_dir() else set()
        sources = scan_files(webui_dirs(cfg, "checkpoints") + webui_dirs(cfg, "diffusion_models") + comfy_dirs(cfg, "checkpoints"))
        for _rel, src in sources:
            if not _is_anima_unet(src.name):
                continue
            if src.name.lower() in already:
                continue
            try:
                _hardlink_or_symlink(src, unet_dest / src.name)
                already.add(src.name.lower())
                linked_unets += 1
            except OSError:
                continue
        if linked_unets:
            notes.append(f"Linked {linked_unets} Anima UNET(s) into diffusion_models (no copy)")
            _REMOTE_CACHE.clear()

    webui = infer_webui_root(cfg)
    vae_src = find_local("vae", ANIMA_VAE, cfg)
    if vae_src is not None:
        dests = list(comfy_dirs(cfg, "vae"))
        if webui is not None:
            dests.append(webui / "models" / "VAE")
        for dest_dir in dests:
            try:
                _hardlink_or_symlink(vae_src, dest_dir / ANIMA_VAE)
            except OSError:
                pass

    te_src = find_local("text_encoders", ANIMA_TEXT_ENCODER, cfg)
    if te_src is not None:
        te_dest = _ensure_comfy_dir(cfg, "text_encoders")
        if te_dest is not None:
            try:
                _hardlink_or_symlink(te_src, te_dest / ANIMA_TEXT_ENCODER)
            except OSError:
                pass
        if webui is not None:
            try:
                _hardlink_or_symlink(te_src, webui / "models" / "text_encoders" / ANIMA_TEXT_ENCODER)
            except OSError:
                pass
    return notes


def link_unseen_files(kind: str, cfg: WrapperConfig | None, client=None) -> int:
    """Hardlink A1111 files ComfyUI does not already see into its native folder."""
    dest = _ensure_comfy_dir(cfg, kind)
    if dest is None:
        return 0
    remote_base = {Path(n).name.lower() for n in _remote_list(kind, client)}
    already = {p.name.lower() for p in dest.iterdir() if p.is_file()} if dest.is_dir() else set()
    linked = 0
    for _rel, src in scan_files(webui_dirs(cfg, kind)):
        key = src.name.lower()
        if key in remote_base or key in already:
            continue
        try:
            _hardlink_or_symlink(src, dest / src.name)
            already.add(key)
            linked += 1
        except OSError:
            continue
    if linked:
        _REMOTE_CACHE.clear()
    return linked


def _hardlink_or_symlink(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        return
    try:
        os.link(src, dest)
        return
    except OSError:
        pass
    try:
        dest.symlink_to(src)
    except OSError as exc:
        raise OSError(f"Cannot link {src} -> {dest} without copying: {exc}") from exc


def create_junction(link: Path, target: Path) -> bool:
    """Directory junction (Windows) or symlink (POSIX). Returns True if created."""
    target = target.resolve()
    if not target.is_dir():
        return False
    if link.exists() or link.is_symlink():
        return False
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise OSError(result.stderr.strip() or result.stdout.strip() or "mklink /J failed")
        return True
    os.symlink(target, link, target_is_directory=True)
    return True


class _LiteralDumper(yaml.SafeDumper):
    """Keep newline-separated folder lists as YAML block scalars."""


def _literal_str_representer(dumper, data):
    style = "|" if "\n" in str(data) else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_LiteralDumper.add_representer(str, _literal_str_representer)


def extra_model_paths_payload(webui_root: Path) -> dict:
    root = str(webui_root).replace("\\", "/")
    return {
        "base_path": root,
        "checkpoints": "models/Stable-diffusion",
        "configs": "models/Stable-diffusion",
        "vae": "models/VAE",
        "loras": "models/Lora\nmodels/LyCORIS",
        "upscale_models": "models/ESRGAN\nmodels/RealESRGAN\nmodels/SwinIR",
        "embeddings": "embeddings",
        "hypernetworks": "models/hypernetworks",
        "controlnet": "models/ControlNet",
        "text_encoders": "models/text_encoders\nmodels/text_encoder\nmodels/clip",
    }


def _covers_webui(existing: dict, webui_root: Path) -> bool:
    base = str(existing.get("base_path") or "").strip()
    if not base:
        return False
    try:
        resolved = Path(base).resolve()
        webui = webui_root.resolve()
    except OSError:
        return False
    return resolved == webui or resolved == (webui / "models")


def _merge_a1111_block(existing: dict, webui_root: Path) -> dict | None:
    """Return an updated a1111 block if missing keys should be added, else None."""
    wanted = extra_model_paths_payload(webui_root)
    merged = dict(existing or {})
    changed = False
    for key, value in wanted.items():
        if key == "base_path":
            if not str(merged.get(key) or "").strip():
                merged[key] = value
                changed = True
            continue
        if key not in merged or merged.get(key) in (None, ""):
            merged[key] = value
            changed = True
    return merged if changed else None


def ensure_extra_model_paths(comfy_root: Path, webui_root: Path) -> bool:
    """Create or fill missing keys in the ``a1111`` mapping. Never drop existing keys."""
    path = comfy_root / "extra_model_paths.yaml"
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        data = loaded if isinstance(loaded, dict) else {}
    else:
        data = {}
    block_key = "a1111" if "a1111" in data or "a111" not in data else "a111"
    existing = data.get(block_key) if isinstance(data.get(block_key), dict) else {}
    if existing:
        merged = _merge_a1111_block(existing, webui_root)
        if merged is None:
            return False
        data[block_key] = merged
    else:
        data[block_key] = extra_model_paths_payload(webui_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    dumped = yaml.dump(data, Dumper=_LiteralDumper, sort_keys=False, allow_unicode=True)
    text = "# Shared A1111 models (managed by comfyui_wrapper — no copies)\n" + dumped
    path.write_text(text.replace("\\\\", "/"), encoding="utf-8")
    return True


def comfy_sees_webui(kind: str, cfg: WrapperConfig | None, client) -> bool:
    local = [p.name.lower() for _, p in scan_files(webui_dirs(cfg, kind))]
    if not local:
        return True
    remote = {Path(n).name.lower() for n in _remote_list(kind, client)}
    if not remote:
        return False
    return bool(set(local) & remote)


def share_webui_models(cfg: WrapperConfig, client=None, *, force: bool = False) -> list[str]:
    """Point ComfyUI at the A1111 models folder (yaml + junctions). Idempotent."""
    global _SHARED, _SHARE_NOTES
    if _SHARED and not force:
        return list(_SHARE_NOTES)
    notes: list[str] = []
    webui = infer_webui_root(cfg)
    comfy = infer_comfyui_root(cfg)
    if webui is None:
        notes.append("A1111 models folder not found. Set paths.webui_root or paths.combinator_root.")
        _SHARE_NOTES = notes
        _SHARED = True
        return notes
    notes.append(f"A1111 models: {webui / 'models'}")

    if comfy is not None:
        try:
            if ensure_extra_model_paths(comfy, webui):
                notes.append(
                    f"Wrote {comfy / 'extra_model_paths.yaml'} — restart ComfyUI once so it picks up the folders."
                )
        except OSError as exc:
            notes.append(f"Could not update extra_model_paths.yaml: {exc}")

        # Junctions only when a live ComfyUI still cannot see A1111 checkpoints.
        # extra_model_paths already covering the same folders would duplicate names.
        if client is not None and not comfy_sees_webui("checkpoints", cfg, client):
            for kind, rels in WEBUI_LAYOUT.items():
                dests = COMFY_LAYOUT.get(kind)
                if not dests:
                    continue
                target = webui / rels[0]
                link = comfy / dests[0] / "_webui"
                if not target.is_dir():
                    continue
                try:
                    if create_junction(link, target):
                        notes.append(f"Linked {kind} -> {target} (no copy)")
                        _REMOTE_CACHE.clear()
                except OSError as exc:
                    notes.append(f"Could not link {kind}: {exc}")

        # Embeddings / hypernets / VAE live in different A1111 folders and are
        # often missed even when checkpoints already show up via extra_model_paths.
        for kind in ("embeddings", "hypernetworks", "vae", "controlnet", "upscale_models", "text_encoders"):
            n = link_unseen_files(kind, cfg, client)
            if n:
                notes.append(f"Linked {n} {kind} from A1111 (no copy)")
        n = link_adetailer_models(cfg)
        if n:
            notes.append(f"Linked {n} ADetailer detectors (no copy)")
        anima_notes = link_anima_assets(cfg, client)
        notes.extend(anima_notes)
    else:
        notes.append("ComfyUI root unknown — listings still use the A1111 folder; set paths.comfyui_root to auto-link.")

    _SHARE_NOTES = notes
    _SHARED = True
    return notes


def ensure_available(kind: str, name: str, cfg: WrapperConfig | None, client=None) -> str:
    """Return a filename ComfyUI will accept, linking the A1111 file if needed."""
    raw = str(name or "").replace("\\", "/").strip()
    if not raw:
        return raw
    remote = _remote_list(kind, client)
    by_full = {n.lower(): n for n in remote}
    by_base = {Path(n).name.lower(): n for n in remote}
    if raw.lower() in by_full:
        return by_full[raw.lower()]
    base = Path(raw).name.lower()
    if base in by_base:
        return by_base[base]

    src = find_local(kind, raw, cfg)
    if src is None:
        return raw

    dest_dirs = comfy_dirs(cfg, kind)
    if dest_dirs:
        dest = dest_dirs[0] / src.name
        try:
            _hardlink_or_symlink(src, dest)
            _REMOTE_CACHE.clear()
        except OSError:
            pass
        remote = _remote_list(kind, client)
        by_base = {Path(n).name.lower(): n for n in remote}
        if src.name.lower() in by_base:
            return by_base[src.name.lower()]
        if dest.exists():
            return dest.name
    return src.name
