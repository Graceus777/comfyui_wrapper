"""A1111-style Gradio WebUI for the ComfyUI wrapper."""

from __future__ import annotations

import argparse
import queue
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from comfyui_wrapper.client import ComfyClient, ComfyError
from comfyui_wrapper.combinator import (
    list_named_json,
    load_jobs,
    prep_jobs,
    preview_jobs,
    run_jobs,
    save_jobs,
)
from comfyui_wrapper.config import WrapperConfig, apply_overrides, load_config
from comfyui_wrapper.generate import generate
from comfyui_wrapper.lora import LoRAResolver
from comfyui_wrapper.h3 import (
    match_h3_unet,
    recipe_choices,
    resolve_recipe,
    video_recipe_ui,
)
from comfyui_wrapper.models import (
    infer_webui_root,
    list_adetailer_models,
    list_embeddings,
    list_hypernetworks,
    list_loras,
    list_model_patches,
    list_pdd_acc,
    list_text_encoders,
    list_unets,
    list_upscalers,
    list_vaes,
    share_webui_models,
)
from comfyui_wrapper.video import generate_video, h3_size, target_dimensions

CSS = """
.gradio-container { max-width: 1480px !important; }
#prompt-row { align-items: stretch; }
#generate-btn button {
  background: linear-gradient(#ff8a3d, #e06700) !important;
  color: #fff !important;
  font-size: 1.35rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.02em;
  border: none !important;
  box-shadow: 0 2px 0 #b35300;
  min-height: 5.6rem !important;
  height: 100% !important;
}
#generate-btn button:hover {
  background: linear-gradient(#ffa05a, #f07010) !important;
}
#interrupt-btn button {
  background: #7a1f1f !important;
  color: #fff !important;
  border: none !important;
}
#refresh-btn button { min-width: 2.6rem; }
#status-bar textarea, #infotext textarea, #batch-log textarea {
  font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
  font-size: 0.82rem;
}
#prompt-box textarea, #negative-box textarea { font-size: 1rem; }
#top-status { font-size: 0.9rem; }
"""

DEFAULT_SAMPLERS = [
    "euler_ancestral",
    "euler",
    "er_sde",
    "dpmpp_2m",
    "dpmpp_2m_sde",
    "dpmpp_sde",
    "dpmpp_2s_ancestral",
    "dpm_2",
    "dpm_2_ancestral",
    "heun",
    "lms",
    "ddim",
    "uni_pc",
]
DEFAULT_SCHEDULERS = [
    "normal",
    "karras",
    "exponential",
    "sgm_uniform",
    "simple",
    "ddim_uniform",
    "beta",
]
SAMPLER_DISPLAY = {
    "euler_ancestral": "Euler a",
    "euler": "Euler",
    "er_sde": "ER SDE",
    "dpmpp_2m": "DPM++ 2M",
    "dpmpp_2m_sde": "DPM++ 2M SDE",
    "dpmpp_sde": "DPM++ SDE",
    "dpmpp_2s_ancestral": "DPM++ 2S a",
    "dpm_2": "DPM2",
    "dpm_2_ancestral": "DPM2 a",
    "heun": "Heun",
    "lms": "LMS",
    "ddim": "DDIM",
    "uni_pc": "UniPC",
}
SIZE_PRESETS = [
    "1024 x 1280 (portrait)",
    "1280 x 1024 (landscape)",
    "1024 x 1024 (square)",
    "1024 x 1344 (Anima)",
    "832 x 1216 (portrait)",
    "1216 x 832 (landscape)",
    "896 x 1152",
    "1152 x 896",
    "768 x 1344",
    "1344 x 768",
]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
VIDEO_SUFFIXES = {".mp4", ".webm"}
VAE_FROM_CHECKPOINT = "(from checkpoint)"


def anima_lllite_controls(
    enabled: bool,
    image: str | Path | None,
    model_patch: str | None,
    strength: float,
    start_percent: float,
    end_percent: float,
    preprocess: str | None = "none",
    canny_low: float = 0.4,
    canny_high: float = 0.8,
) -> list[dict[str, Any]]:
    """Build one UI-authored Anima LLLite control with clear validation."""
    if not enabled:
        return []
    image_value = str(image or "").strip()
    patch_value = str(model_patch or "").strip()
    if not image_value:
        raise ValueError("Enable Anima LLLite requires a control image")
    if not patch_value:
        raise ValueError("Enable Anima LLLite requires a model patch")
    start = float(start_percent)
    end = float(end_percent)
    if not 0.0 <= start <= end <= 1.0:
        raise ValueError("Anima LLLite schedule must satisfy 0 <= start <= end <= 1")
    mode = str(preprocess or "none").strip().lower()
    if mode not in ("none", "canny"):
        raise ValueError("Anima preprocessor must be none or canny")
    spec: dict[str, Any] = {
        "image": image_value,
        "model_patch": patch_value,
        "strength": float(strength),
        "start_percent": start,
        "end_percent": end,
    }
    if mode != "none":
        spec["preprocess"] = mode
    if mode == "canny":
        low, high = float(canny_low), float(canny_high)
        if not 0.01 <= low < high <= 0.99:
            raise ValueError("Canny thresholds must satisfy 0.01 <= low < high <= 0.99")
        spec["canny_low"] = low
        spec["canny_high"] = high
    return [spec]

WORKFLOW_PRESETS = {
    "sdxl_txt2img.api.json": {
        "lora_mode": "full",
        "steps": 28,
        "cfg": 5.0,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "vae": VAE_FROM_CHECKPOINT,
    },
    "krea2_txt2img.api.json": {
        "checkpoint": "krea2_turbo_fp8_scaled.safetensors",
        "clip": "qwen3vl_4b_fp8_scaled.safetensors",
        "vae": "qwen_image_vae.safetensors",
        "lora_mode": "model_only",
        "steps": 8,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "width": 1024,
        "height": 1024,
    },
    "anima_txt2img.api.json": {
        "checkpoint": "waiANIMA_v10Base10.safetensors",
        "clip": "qwen_3_06b_base.safetensors",
        "vae": "qwen_image_vae.safetensors",
        "lora_mode": "model_only",
        "steps": 28,
        "cfg": 4.5,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "width": 1024,
        "height": 1344,
    },
    "flux_gguf_txt2img.api.json": {
        "checkpoint": "flux1-dev-Q8_0.gguf",
        "clip": "clip_l.safetensors",
        "clip2": "t5xxl_fp8_e4m3fn.safetensors",
        "vae": "ae.safetensors",
        "lora_mode": "model_only",
        "steps": 20,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "width": 1024,
        "height": 1024,
    },
}


def format_bytes(n: Any) -> str:
    try:
        value = float(n)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} PB"


def sampler_choices(names: Iterable[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for name in names:
        key = str(name)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((SAMPLER_DISPLAY.get(key, key), key))
    return out


def parse_size_preset(label: str | None) -> tuple[int, int] | None:
    if not label:
        return None
    match = re.search(r"(\d+)\s*[xX×]\s*(\d+)", str(label))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def insert_lora_tag(prompt: str, name: str, weight: float = 1.0) -> str:
    """Insert or replace an A1111 ``<lora:name:weight>`` tag."""
    raw = (name or "").strip()
    if not raw:
        return prompt or ""
    tag = f"<lora:{raw}:{float(weight):g}>"
    text = prompt or ""
    stem = Path(raw).stem
    pattern = re.compile(
        rf"<lora:(?:[^:>]*[/\\\\])?{re.escape(stem)}(?:\.[A-Za-z0-9]+)?:[^>]*>",
        re.IGNORECASE,
    )
    if pattern.search(text):
        return pattern.sub(tag, text, count=1)
    text = text.rstrip()
    return f"{text}, {tag}" if text else tag


def insert_embedding_tag(prompt: str, name: str, weight: float = 1.0) -> str:
    """Insert ComfyUI ``embedding:name`` (optional A1111-style attention weight)."""
    raw = (name or "").strip()
    if not raw:
        return prompt or ""
    stem = Path(raw).stem
    token = f"embedding:{stem}"
    if abs(float(weight) - 1.0) > 1e-6:
        token = f"({token}:{float(weight):g})"
    text = prompt or ""
    pattern = re.compile(
        rf"\(?embedding:{re.escape(stem)}(?::[0-9.]+)?\)?",
        re.IGNORECASE,
    )
    if pattern.search(text):
        return pattern.sub(token, text, count=1)
    text = text.rstrip()
    return f"{text}, {token}" if text else token


def parse_slot_text(text: str | None) -> dict[str, list[str]]:
    """Parse ``pose=a,b`` lines (or semicolon-separated) into extra_slots."""
    slots: dict[str, list[str]] = {}
    if not text or not str(text).strip():
        return slots
    raw = str(text).replace(";", "\n")
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Slot override must be name=a,b (got {line!r})")
        key, rest = line.split("=", 1)
        values = [x.strip() for x in rest.split(",") if x.strip()]
        if key.strip() and values:
            slots[key.strip()] = values
    return slots


def split_csv(text: str | None) -> list[str] | None:
    if not text or not str(text).strip():
        return None
    items = [x.strip() for x in str(text).split(",") if x.strip()]
    return items or None


def filter_choices(query: str | None, items: Iterable[str]) -> list[str]:
    q = (query or "").strip().lower()
    values = [str(x) for x in items]
    if not q:
        return values
    return [x for x in values if q in x.lower()]


def selected_vae(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {VAE_FROM_CHECKPOINT, "none", "automatic", "auto"}:
        return None
    return raw


def vae_dropdown_choices(vaes: Iterable[str], current: str | None = None) -> list[str]:
    return prefer_first([VAE_FROM_CHECKPOINT, *[str(x) for x in vaes if x]], current or VAE_FROM_CHECKPOINT)


def prefer_first(items: Iterable[str], preferred: str | None) -> list[str]:
    values = [str(x) for x in items if str(x)]
    pref = (preferred or "").strip()
    if pref:
        values = [x for x in values if x != pref]
        values.insert(0, pref)
    return values


def match_choice(choices: Iterable[str], *wanted: str) -> str | None:
    """Pick the listed file that matches a preferred name (exact, basename, or unique stem)."""
    values = [str(x) for x in choices if x]
    by_lower = {v.lower(): v for v in values}
    by_name = {Path(v).name.lower(): v for v in values}
    for raw in wanted:
        w = (raw or "").strip()
        if not w:
            continue
        if w in values:
            return w
        if w.lower() in by_lower:
            return by_lower[w.lower()]
        name = Path(w).name.lower()
        if name in by_name:
            return by_name[name]
        stem = Path(w).stem.lower()
        hits = [v for v in values if stem in Path(v).name.lower()]
        if len(hits) == 1:
            return hits[0]
        if hits:
            hits.sort(key=lambda v: (len(Path(v).name), len(v)))
            return hits[0]
    return values[0] if values else None


def format_progress(message: dict | None) -> str:
    if not message:
        return "Working..."
    kind = str(message.get("type") or "")
    data = message.get("data") or {}
    if kind == "progress":
        value = data.get("value")
        max_v = data.get("max")
        if max_v:
            return f"Sampling {value}/{max_v}"
        return f"Progress {value}"
    if kind == "executing":
        node = data.get("node")
        if node is None:
            return "Finishing..."
        if str(node) == "rtx_vsr":
            return "RTX VSR upscale..."
        title = ""
        if isinstance(data.get("display_node"), dict):
            title = (data.get("display_node") or {}).get("title") or ""
        return f"Executing node {node}" + (f" ({title})" if title else "")
    if kind == "execution_cached":
        return "Using cached nodes..."
    if kind == "status":
        remaining = ((data.get("status") or {}).get("exec_info") or {}).get("queue_remaining")
        if remaining is not None:
            return f"Queue remaining: {remaining}"
    if kind == "poll":
        return "Waiting for ComfyUI..."
    if kind == "ws_fallback":
        return "WebSocket unavailable, polling..."
    if kind:
        return kind.replace("_", " ")
    return "Working..."


def format_infotext(
    *,
    positive: str,
    negative: str,
    steps: int,
    sampler: str,
    scheduler: str,
    cfg: float,
    seed: int,
    width: int,
    height: int,
    checkpoint: str,
    loras: list[dict] | None = None,
    hires: dict | None = None,
    adetailer: dict | None = None,
    img2img: dict | None = None,
) -> str:
    """A1111-style generation info block."""
    lines = [positive or ""]
    if negative:
        lines.append(f"Negative prompt: {negative}")
    sampler_label = SAMPLER_DISPLAY.get(sampler, sampler)
    bits = [
        f"Steps: {steps}",
        f"Sampler: {sampler_label}",
        f"Schedule type: {scheduler}",
        f"CFG scale: {cfg}",
        f"Seed: {seed}",
        f"Size: {width}x{height}",
        f"Model: {checkpoint}",
    ]
    if loras:
        lora_s = ", ".join(
            f"{item.get('name') or item.get('file')}:{item.get('weight', 1)}"
            for item in loras
        )
        bits.append(f"LoRAs: {lora_s}")
    if hires and hires.get("enable"):
        bits.append(
            f"Hires upscale: {hires.get('scale')} ({hires.get('upscaler')}, "
            f"{hires.get('steps')} steps, denoise {hires.get('denoise')})"
        )
    if adetailer and adetailer.get("enable"):
        bits.append(f"ADetailer: {adetailer.get('model')}")
    if img2img and img2img.get("enable"):
        bits.append(f"img2img: denoise {img2img.get('denoise')}")
    lines.append(", ".join(str(x) for x in bits))
    return "\n".join(lines)


def list_workflows(cfg: WrapperConfig) -> list[str]:
    folder = cfg.paths.workflows
    if not folder.exists():
        name = Path(cfg.workflow_file).name
        return [name] if name else []
    names = sorted(p.name for p in folder.glob("*.json") if p.is_file())
    return prefer_first(names, Path(cfg.workflow_file).name)


def list_job_files(cfg: WrapperConfig) -> list[str]:
    folder = cfg.paths.jobs
    if not folder.exists():
        return []
    files = sorted(folder.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in files]


def list_output_images(cfg: WrapperConfig, limit: int = 80) -> list[str]:
    folder = cfg.paths.output
    if not folder.exists():
        return []
    files = [
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(p) for p in files[:limit]]


def ksampler_options(client: ComfyClient) -> tuple[list[str], list[str]]:
    try:
        info = client.get("/object_info/KSampler")
        node = (info or {}).get("KSampler") or {}
        required = ((node.get("input") or {}).get("required") or {})
        samplers = list((required.get("sampler_name") or [[]])[0] or [])
        schedulers = list((required.get("scheduler") or [[]])[0] or [])
        if samplers and schedulers:
            return [str(x) for x in samplers], [str(x) for x in schedulers]
    except Exception:
        pass
    return list(DEFAULT_SAMPLERS), list(DEFAULT_SCHEDULERS)


def gallery_items(paths: Iterable[str]) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for path in paths:
        if not path:
            continue
        p = Path(path)
        items.append((str(p), p.name))
    return items


def collect_saved(result, client: ComfyClient, output_dir: Path) -> list[str]:
    if result.saved_files:
        return [str(p) for p in result.saved_files]
    if not result.images:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for i, image in enumerate(result.images, start=1):
        filename = image.get("filename") or f"output_{i}.png"
        data = client.get_image_bytes(
            filename,
            subfolder=image.get("subfolder") or "",
            folder_type=image.get("type") or "output",
        )
        dest = output_dir / Path(filename).name
        if dest.exists():
            dest = output_dir / f"{dest.stem}_{i}{dest.suffix}"
        dest.write_bytes(data)
        saved.append(str(dest))
    return saved


def server_status_text(cfg: WrapperConfig, client: ComfyClient | None = None) -> str:
    client = client or ComfyClient(cfg=cfg)
    if not client.ping():
        return f"ComfyUI offline at {cfg.comfyui.base_url} — start it, then hit Refresh."
    try:
        stats = client.system_stats()
        q = client.queue()
    except ComfyError as exc:
        return f"ComfyUI error: {exc}"
    devices = stats.get("devices") or [{}]
    device = devices[0] if devices else {}
    name = device.get("name") or "GPU"
    vram = device.get("vram_total")
    free = device.get("vram_free")
    running = len(q.get("queue_running") or [])
    pending = len(q.get("queue_pending") or [])
    vram_s = f"{format_bytes(free)} free / {format_bytes(vram)}" if vram else "VRAM n/a"
    webui = infer_webui_root(cfg)
    share = f"  ·  A1111 models {webui / 'models'}" if webui else ""
    return f"Connected to {cfg.comfyui.base_url}  ·  {name}  ·  {vram_s}  ·  queue {running} running, {pending} pending{share}"


def _import_gradio():
    try:
        import gradio as gr  # type: ignore
    except ImportError as exc:
        raise ImportError(
            'Gradio is required for the WebUI. Install with: pip install -e ".[web]"'
        ) from exc
    return gr


def build_ui(cfg: WrapperConfig):
    gr = _import_gradio()
    ctx: dict[str, Any] = {
        "cfg": cfg,
        "stop": threading.Event(),
        "busy": threading.Lock(),
        "client": None,
        "loras": [],
        "last_seed": cfg.generation.seed,
        "lora_mode": cfg.generation.lora_mode,
    }

    def current_cfg(**overrides: Any) -> WrapperConfig:
        return apply_overrides(ctx["cfg"], **overrides)

    def live_client() -> ComfyClient:
        client = ComfyClient(cfg=ctx["cfg"])
        ctx["client"] = client
        return client

    def load_options():
        cfg_now = ctx["cfg"]
        gen = cfg_now.generation
        client = live_client()
        connected = client.ping()
        live = client if connected else None
        share_notes = share_webui_models(cfg_now, live, force=True)
        checkpoints = list_unets(cfg_now, live)
        clips = list_text_encoders(cfg_now, live)
        vaes = list_vaes(cfg_now, live)
        loras = list_loras(cfg_now, live)
        embeddings = list_embeddings(cfg_now, live)
        hypernets = list_hypernetworks(cfg_now, live)
        upscalers = list_upscalers(cfg_now, live)
        ad_models = list_adetailer_models(cfg_now, live)
        model_patches = list_model_patches(cfg_now, live)
        samplers = list(DEFAULT_SAMPLERS)
        schedulers = list(DEFAULT_SCHEDULERS)
        status = server_status_text(cfg_now, client)
        extra = [n for n in share_notes if n.lower().startswith("wrote") or n.lower().startswith("linked")]
        if extra:
            status = status + "  \n" + "  \n".join(extra)
        if connected:
            samplers, schedulers = ksampler_options(client)
        ctx["loras"] = loras
        ctx["embeddings"] = embeddings
        ctx["hypernetworks"] = hypernets
        checkpoints = prefer_first(checkpoints, gen.checkpoint)
        sampler_dd = sampler_choices(prefer_first(samplers, gen.sampler))
        schedulers = prefer_first(schedulers, gen.scheduler)
        workflows = list_workflows(cfg_now)
        zones = list_named_json(cfg_now.paths.zone_dirs())
        prompts = list_named_json(cfg_now.paths.prompt_dirs())
        jobs = list_job_files(cfg_now)
        vid = cfg_now.video
        acc_files = list_pdd_acc(cfg_now, live)
        rec = resolve_recipe(vid.recipe, acc_file=vid.acc_file, acc_files=acc_files)
        v_unet = match_h3_unet(checkpoints, rec, vid.unet) or match_choice(
            checkpoints, vid.unet, "minimax_h3_fl2va", "minimax_h3"
        )
        v_clip = match_choice(clips, vid.clip, "qwen3vl_32b_minimax", "minimax_h3")
        v_vae = match_choice(vaes, vid.vae, "minimax_h3_video_vae")
        v_audio_vae = match_choice(vaes, vid.audio_vae, "minimax_h3_audio_vae")
        rec, v_steps, v_acc = video_recipe_ui(rec, v_unet, acc_files, vid.acc_file)
        anima_specs = [item for item in gen.anima_lllite if isinstance(item, dict)]
        anima_spec = anima_specs[0] if anima_specs else {}
        anima_image_value = str(anima_spec.get("image") or "").strip()
        if anima_image_value:
            image_path = Path(anima_image_value)
            if not image_path.is_absolute():
                image_path = cfg_now.paths.root / image_path
            anima_image_value = str(image_path) if image_path.is_file() else ""
        anima_model = str(
            anima_spec.get("model_patch") or anima_spec.get("file") or ""
        ).strip()
        if not anima_model:
            anima_model = match_choice(model_patches, "", "anima-lllite-depth") or (
                model_patches[0] if model_patches else None
            )
        ctx["acc_files"] = acc_files
        init_value = str(getattr(gen, "init_image", "") or "").strip()
        init_path_value: str | None = None
        if init_value:
            init_path = Path(init_value)
            if not init_path.is_absolute():
                init_path = cfg_now.paths.root / init_path
            init_path_value = str(init_path) if init_path.is_file() else None
        try:
            init_denoise = float(getattr(gen, "denoise", 1.0) or 1.0)
        except (TypeError, ValueError):
            init_denoise = 1.0
        if not 0.0 < init_denoise <= 1.0:
            init_denoise = 0.6
        elif not init_path_value:
            # txt2img default: keep a useful img2img strength ready.
            init_denoise = 0.6 if init_denoise >= 1.0 else init_denoise
        return {
            "status": status,
            "img2img_image": init_path_value,
            "img2img_denoise": float(init_denoise),
            "checkpoints": checkpoints,
            "checkpoint": gen.checkpoint if gen.checkpoint in checkpoints or gen.checkpoint else (checkpoints[0] if checkpoints else None),
            "clips": clips,
            "clip": gen.clip if gen.clip in clips or gen.clip else (clips[0] if clips else None),
            "clip2s": clips,
            "clip2": gen.clip2 if gen.clip2 in clips or gen.clip2 else None,
            "vaes": vae_dropdown_choices(vaes, gen.vae),
            "vae": gen.vae if gen.vae else VAE_FROM_CHECKPOINT,
            "loras": loras,
            "lora": loras[0] if loras else None,
            "embeddings": embeddings,
            "embedding": embeddings[0] if embeddings else None,
            "hypernetworks": hypernets,
            "hypernetwork": hypernets[0] if hypernets else None,
            "upscalers": upscalers,
            "upscaler": cfg_now.generation.hires_upscaler if cfg_now.generation.hires_upscaler in upscalers else (upscalers[0] if upscalers else "latent"),
            "ad_models": ad_models,
            "ad_model": (
                cfg_now.generation.adetailer_model
                if cfg_now.generation.adetailer_model in ad_models
                else (ad_models[0] if ad_models else None)
            ),
            "anima_enabled": bool(anima_specs),
            "anima_image": anima_image_value or None,
            "anima_model_patches": prefer_first(model_patches, anima_model),
            "anima_model_patch": anima_model,
            "anima_strength": float(anima_spec.get("strength", 1.0)),
            "anima_start_percent": float(anima_spec.get("start_percent", 0.0)),
            "anima_end_percent": float(anima_spec.get("end_percent", 1.0)),
            "anima_preprocess": str(anima_spec.get("preprocess") or "none").strip().lower(),
            "anima_canny_low": float(anima_spec.get("canny_low", 0.4)),
            "anima_canny_high": float(anima_spec.get("canny_high", 0.8)),
            "samplers": sampler_dd,
            "sampler": gen.sampler,
            "schedulers": schedulers,
            "scheduler": gen.scheduler,
            "workflows": workflows,
            "workflow": Path(cfg_now.workflow_file).name,
            "zones": zones,
            "zone": zones[0] if zones else None,
            "prompts": prompts,
            "prompt_config": prompts[0] if prompts else None,
            "jobs": jobs,
            "job_file": jobs[0] if jobs else None,
            "v_unets": prefer_first(checkpoints, v_unet),
            "v_unet": v_unet,
            "v_clips": prefer_first(clips, v_clip),
            "v_clip": v_clip,
            "v_vaes": prefer_first(vaes, v_vae),
            "v_vae": v_vae,
            "v_audio_vaes": prefer_first(vaes, v_audio_vae),
            "v_audio_vae": v_audio_vae,
            "v_recipes": recipe_choices(acc_files),
            "v_recipe": rec.key,
            "v_acc_files": prefer_first(acc_files, v_acc) if rec.acc else ([""] + prefer_first(acc_files, None)),
            "v_acc": v_acc,
            "v_steps": v_steps,
            "v_latent_upscale": bool(vid.latent_upscale),
            "v_rtx_vsr": bool(vid.rtx_vsr),
        }

    initial = load_options()

    with gr.Blocks(title="ComfyUI Wrapper") as demo:
        gr.Markdown("## ComfyUI Wrapper")
        status_md = gr.Markdown(initial["status"], elem_id="top-status")

        with gr.Row():
            workflow = gr.Dropdown(
                label="Workflow",
                choices=initial["workflows"],
                value=initial["workflow"],
                scale=3,
                allow_custom_value=True,
            )
            checkpoint = gr.Dropdown(
                label="Model (checkpoint / UNET / GGUF)",
                choices=initial["checkpoints"],
                value=initial["checkpoint"],
                scale=4,
                allow_custom_value=True,
            )
            refresh_btn = gr.Button("Refresh", elem_id="refresh-btn", scale=1)
        with gr.Row():
            clip = gr.Dropdown(
                label="Text encoder / CLIP",
                choices=initial["clips"],
                value=initial["clip"],
                scale=3,
                allow_custom_value=True,
            )
            clip2 = gr.Dropdown(
                label="CLIP 2 (Flux T5)",
                choices=initial["clip2s"],
                value=initial["clip2"],
                scale=3,
                allow_custom_value=True,
            )
            vae = gr.Dropdown(
                label="VAE",
                choices=initial["vaes"],
                value=initial["vae"],
                scale=3,
                allow_custom_value=True,
            )

        with gr.Tabs():
            with gr.Tab("txt2img"):
                with gr.Row():
                    with gr.Column(scale=6):
                        with gr.Row(elem_id="prompt-row", equal_height=True):
                            with gr.Column(scale=5):
                                prompt = gr.Textbox(
                                    label="Prompt",
                                    value=cfg.generation.positive,
                                    lines=3,
                                    elem_id="prompt-box",
                                    placeholder="masterpiece, best quality, 1girl",
                                )
                                negative = gr.Textbox(
                                    label="Negative prompt",
                                    value=cfg.generation.negative,
                                    lines=2,
                                    elem_id="negative-box",
                                )
                            with gr.Column(scale=1, min_width=150):
                                generate_btn = gr.Button(
                                    "Generate",
                                    variant="primary",
                                    elem_id="generate-btn",
                                )
                                interrupt_btn = gr.Button(
                                    "Interrupt",
                                    elem_id="interrupt-btn",
                                )

                        with gr.Accordion("Extra networks", open=False):
                            with gr.Tabs():
                                with gr.Tab("LoRA"):
                                    gr.Markdown(
                                        "Inserts `<lora:name:weight>`. Tags are stripped from the "
                                        "prompt and applied as ComfyUI LoRA nodes."
                                    )
                                    with gr.Row():
                                        lora_search = gr.Textbox(label="Search", scale=2)
                                        lora_pick = gr.Dropdown(
                                            label="LoRA",
                                            choices=initial["loras"],
                                            value=initial["lora"],
                                            scale=4,
                                            allow_custom_value=True,
                                        )
                                        lora_weight = gr.Slider(
                                            label="Weight",
                                            minimum=0,
                                            maximum=2,
                                            step=0.05,
                                            value=0.8,
                                            scale=2,
                                        )
                                        lora_add = gr.Button("Add to prompt", scale=1)
                                with gr.Tab("Textual inversion"):
                                    gr.Markdown(
                                        "Inserts ComfyUI `embedding:name` from your A1111 "
                                        "`embeddings/` folder (no copy)."
                                    )
                                    with gr.Row():
                                        emb_search = gr.Textbox(label="Search", scale=2)
                                        emb_pick = gr.Dropdown(
                                            label="Embedding",
                                            choices=initial["embeddings"],
                                            value=initial["embedding"],
                                            scale=4,
                                            allow_custom_value=True,
                                        )
                                        emb_weight = gr.Slider(
                                            label="Weight",
                                            minimum=0,
                                            maximum=2,
                                            step=0.05,
                                            value=1.0,
                                            scale=2,
                                        )
                                        emb_add = gr.Button("Add to prompt", scale=1)
                                        emb_neg = gr.Button("Add to negative", scale=1)
                                with gr.Tab("Hypernetworks"):
                                    gr.Markdown(
                                        "Shared from A1111 `models/hypernetworks`. "
                                        "Inserts `<hypernet:name:weight>` for reference."
                                    )
                                    with gr.Row():
                                        hn_search = gr.Textbox(label="Search", scale=2)
                                        hn_pick = gr.Dropdown(
                                            label="Hypernetwork",
                                            choices=initial["hypernetworks"],
                                            value=initial["hypernetwork"],
                                            scale=4,
                                            allow_custom_value=True,
                                        )

                        with gr.Row():
                            sampler = gr.Dropdown(
                                label="Sampling method",
                                choices=initial["samplers"],
                                value=initial["sampler"],
                                scale=2,
                            )
                            scheduler = gr.Dropdown(
                                label="Schedule type",
                                choices=initial["schedulers"],
                                value=initial["scheduler"],
                                scale=2,
                            )
                            steps = gr.Slider(
                                label="Sampling steps",
                                minimum=1,
                                maximum=150,
                                step=1,
                                value=int(cfg.generation.steps),
                                scale=2,
                            )

                        with gr.Row():
                            width = gr.Slider(
                                label="Width",
                                minimum=64,
                                maximum=2048,
                                step=8,
                                value=int(cfg.generation.width),
                            )
                            height = gr.Slider(
                                label="Height",
                                minimum=64,
                                maximum=2048,
                                step=8,
                                value=int(cfg.generation.height),
                            )
                            swap_wh = gr.Button("⇄", min_width=40)
                            size_preset = gr.Dropdown(
                                label="Size preset",
                                choices=SIZE_PRESETS,
                                value=None,
                            )

                        with gr.Row():
                            cfg_scale = gr.Slider(
                                label="CFG Scale",
                                minimum=1,
                                maximum=20,
                                step=0.5,
                                value=float(cfg.generation.cfg),
                            )
                            batch_count = gr.Slider(
                                label="Batch count",
                                minimum=1,
                                maximum=16,
                                step=1,
                                value=1,
                            )
                            batch_size = gr.Slider(
                                label="Batch size",
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=int(cfg.generation.batch_size),
                            )

                        with gr.Row():
                            seed = gr.Number(
                                label="Seed (-1 = random)",
                                value=int(cfg.generation.seed),
                                precision=0,
                            )
                            dice_btn = gr.Button("Random", min_width=80)
                            recycle_btn = gr.Button("Recycle last", min_width=80)

                        with gr.Accordion("Advanced", open=False):
                            with gr.Row():
                                denoise = gr.Slider(
                                    label="Denoise",
                                    minimum=0,
                                    maximum=1,
                                    step=0.01,
                                    value=float(cfg.generation.denoise),
                                )
                                prefix = gr.Textbox(
                                    label="Filename prefix",
                                    value=cfg.generation.filename_prefix,
                                )

                        with gr.Accordion("img2img", open=False):
                            gr.Markdown(
                                "Drop an init image to rework it instead of txt2img. "
                                "It is resized to Width × Height (lanczos), VAE-encoded, "
                                "and fed to the first KSampler. Works on SDXL, Anima, "
                                "Krea 2, and Flux GGUF. Empty = txt2img."
                            )
                            img2img_image = gr.Image(
                                label="Init image (empty = txt2img)",
                                type="filepath",
                                value=initial.get("img2img_image"),
                            )
                            img2img_denoise = gr.Slider(
                                label="Denoise (0.3 light to 0.75 heavy rework)",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                value=float(initial.get("img2img_denoise", 0.6)),
                            )

                        with gr.Accordion("Hires. fix", open=False):
                            hires_enable = gr.Checkbox(
                                label="Enable hires. fix",
                                value=bool(cfg.generation.hires_enable),
                            )
                            with gr.Row():
                                hires_upscaler = gr.Dropdown(
                                    label="Upscaler",
                                    choices=initial["upscalers"],
                                    value=initial["upscaler"],
                                    allow_custom_value=True,
                                )
                                hires_scale = gr.Slider(
                                    label="Upscale by",
                                    minimum=1.0,
                                    maximum=4.0,
                                    step=0.05,
                                    value=float(cfg.generation.hires_scale),
                                )
                            with gr.Row():
                                hires_steps = gr.Slider(
                                    label="Hires steps (0 = same)",
                                    minimum=0,
                                    maximum=150,
                                    step=1,
                                    value=int(cfg.generation.hires_steps),
                                )
                                hires_denoise = gr.Slider(
                                    label="Hires denoise",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=float(cfg.generation.hires_denoise),
                                )

                        with gr.Accordion("Anima LLLite control", open=True):
                            gr.Markdown(
                                "Control image + LLLite patch. Use a preprocessed depth, "
                                "pose, or lineart map as-is (preprocessor None), or hand "
                                "a raw photo to Canny for A1111-style edge extraction "
                                "in-graph (pairs with the lineart patch). The image is "
                                "uploaded to ComfyUI when you generate."
                            )
                            anima_enable = gr.Checkbox(
                                label="Enable Anima LLLite",
                                value=bool(initial["anima_enabled"]),
                            )
                            anima_image = gr.Image(
                                label="Control image (preprocessed map, or raw photo with Canny)",
                                type="filepath",
                                value=initial["anima_image"],
                            )
                            anima_preprocess = gr.Dropdown(
                                label="Preprocessor",
                                choices=[("None (already preprocessed)", "none"), ("Canny edges (raw photo)", "canny")],
                                value=initial.get("anima_preprocess", "none"),
                            )
                            with gr.Row():
                                anima_canny_low = gr.Slider(
                                    label="Canny low",
                                    minimum=0.01,
                                    maximum=0.99,
                                    step=0.01,
                                    value=float(initial.get("anima_canny_low", 0.4)),
                                )
                                anima_canny_high = gr.Slider(
                                    label="Canny high",
                                    minimum=0.01,
                                    maximum=0.99,
                                    step=0.01,
                                    value=float(initial.get("anima_canny_high", 0.8)),
                                )
                            anima_model_patch = gr.Dropdown(
                                label="LLLite model patch",
                                choices=initial["anima_model_patches"],
                                value=initial["anima_model_patch"],
                                allow_custom_value=True,
                            )
                            with gr.Row():
                                anima_strength = gr.Slider(
                                    label="Strength",
                                    minimum=-2.0,
                                    maximum=2.0,
                                    step=0.05,
                                    value=float(initial["anima_strength"]),
                                )
                                anima_start = gr.Slider(
                                    label="Start percent",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=float(initial["anima_start_percent"]),
                                )
                                anima_end = gr.Slider(
                                    label="End percent",
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    value=float(initial["anima_end_percent"]),
                                )

                        with gr.Accordion("ADetailer", open=False):
                            ad_enable = gr.Checkbox(
                                label="Enable ADetailer (Impact FaceDetailer)",
                                value=bool(cfg.generation.adetailer_enable),
                            )
                            ad_model = gr.Dropdown(
                                label="Detector model",
                                choices=initial["ad_models"],
                                value=initial["ad_model"],
                                allow_custom_value=True,
                            )
                            with gr.Row():
                                ad_steps = gr.Slider(
                                    label="ADetailer steps",
                                    minimum=1,
                                    maximum=80,
                                    step=1,
                                    value=int(cfg.generation.adetailer_steps),
                                )
                                ad_cfg = gr.Slider(
                                    label="ADetailer CFG",
                                    minimum=1,
                                    maximum=20,
                                    step=0.5,
                                    value=float(cfg.generation.adetailer_cfg),
                                )
                                ad_denoise = gr.Slider(
                                    label="ADetailer denoise",
                                    minimum=0.05,
                                    maximum=1.0,
                                    step=0.01,
                                    value=float(cfg.generation.adetailer_denoise),
                                )

                    with gr.Column(scale=5):
                        gallery = gr.Gallery(
                            label="Output",
                            columns=2,
                            height=560,
                            object_fit="contain",
                            preview=True,
                        )
                        gen_status = gr.Textbox(
                            label="Status",
                            value="Ready",
                            interactive=False,
                            elem_id="status-bar",
                        )
                        infotext = gr.Textbox(
                            label="Generation info",
                            lines=6,
                            elem_id="infotext",
                        )

            with gr.Tab("Video / H3 loop"):
                gr.Markdown(
                    "MiniMax H3 image-to-video with native audio mux "
                    "(audio VAE + VAEDecodeAudio → CreateVideo), plus optional "
                    "cyclic crossfade. Native is 12-step `res_multistep` (Acc file cleared). "
                    "Acc uses `MiniMaxH3PDDAccApply` (not Load LoRA) with euler + Acc sigmas. "
                    "**Size from image** keeps the still's aspect on H3's 32px grid "
                    "(1024×1280 → 640×800) and writes that size to both `ImageScale` and "
                    "the H3 latent. 3D latent upscaler is decode-only 2x; RTX VSR is a post pass. "
                    "**R2V-lite** reuses your FL2VA UNET with the native `MiniMaxH3ReferenceToVideo` node "
                    "(`<Picture i>` / `<Video k>` / `<Audio j>`); weaker refs than true Ref2VA, zero new models. "
                    "I2V accepts an optional last frame for first+last-frame interpolation."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            v_mode = gr.Dropdown(
                                label="Mode",
                                choices=[("Image-to-video", "i2v"), ("Reference-lite (FL2VA)", "r2v")],
                                value="i2v",
                            )
                            v_ref_size = gr.Dropdown(
                                label="Ref size",
                                choices=["match", "max"],
                                value=str(getattr(cfg.video, "ref_image_size", "match") or "match"),
                            )
                        v_image = gr.Image(label="First frame (I2V; doubles as <Picture 1> in R2V)", type="filepath")
                        v_last = gr.Image(label="Last frame (optional FLF2V, I2V only)", type="filepath")
                        v_ref_images = gr.File(label="Ref stills (<Picture i>, max 9)", file_count="multiple", file_types=["image"])
                        v_ref_video = gr.File(label="Ref clip (<Video k>, mp4, max 3)", file_count="multiple", file_types=["video"])
                        v_ref_audio = gr.File(label="Ref audio (<Audio j>, max 3)", file_count="multiple", file_types=["audio"])
                        v_prompt = gr.Textbox(
                            label="Motion prompt",
                            lines=6,
                            value=cfg.video.prompt or "",
                        )
                        with gr.Row():
                            v_width = gr.Number(label="Width", value=int(cfg.video.width), precision=0)
                            v_height = gr.Number(label="Height", value=int(cfg.video.height), precision=0)
                            v_auto_size = gr.Checkbox(
                                label="Size from image (H3 32px grid)",
                                value=True,
                            )
                        with gr.Row():
                            v_recipe = gr.Dropdown(
                                label="Recipe",
                                choices=initial["v_recipes"],
                                value=initial["v_recipe"],
                            )
                            v_frames = gr.Slider(label="Frames (24 fps)", minimum=29, maximum=362, step=17, value=int(cfg.video.frames))
                            v_steps = gr.Slider(label="Steps", minimum=4, maximum=40, step=1, value=int(initial["v_steps"]))
                            v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                        with gr.Row():
                            v_unet = gr.Dropdown(
                                label="H3 UNET",
                                choices=initial["v_unets"],
                                value=initial["v_unet"],
                                allow_custom_value=True,
                            )
                            v_clip = gr.Dropdown(
                                label="H3 CLIP",
                                choices=initial["v_clips"],
                                value=initial["v_clip"],
                                allow_custom_value=True,
                            )
                            v_vae = gr.Dropdown(
                                label="Video VAE",
                                choices=initial["v_vaes"],
                                value=initial["v_vae"],
                                allow_custom_value=True,
                            )
                            v_audio_vae = gr.Dropdown(
                                label="Audio VAE",
                                choices=initial["v_audio_vaes"],
                                value=initial["v_audio_vae"],
                                allow_custom_value=True,
                            )
                        v_acc = gr.Dropdown(
                            label="Acc file (PDD, not LoRA)",
                            choices=initial["v_acc_files"],
                            value=initial["v_acc"],
                            allow_custom_value=True,
                        )
                        with gr.Row():
                            v_loop = gr.Checkbox(label="Close loop (ffmpeg crossfade)", value=bool(cfg.video.loop))
                            v_latent_up = gr.Checkbox(
                                label="3D latent upscaler (2x)",
                                value=bool(initial["v_latent_upscale"]),
                            )
                            v_rtx = gr.Checkbox(
                                label="RTX VSR (2x post)",
                                value=bool(initial["v_rtx_vsr"]),
                            )
                        v_generate = gr.Button("Generate loop", variant="primary", elem_id="generate-btn")
                    with gr.Column(scale=5):
                        v_video = gr.Video(label="Output")
                        v_status = gr.Textbox(label="Status", value="Ready", interactive=False)

            with gr.Tab("Batch"):
                gr.Markdown(
                    "Combinator batch: expand a zone (Always / Loop / Random) plus "
                    "prompt slots, inspect the job list, then run it."
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        with gr.Row():
                            zone = gr.Dropdown(
                                label="Zone",
                                choices=initial["zones"],
                                value=initial["zone"],
                                allow_custom_value=True,
                            )
                            prompt_config = gr.Dropdown(
                                label="Prompt config",
                                choices=initial["prompts"],
                                value=initial["prompt_config"],
                                allow_custom_value=True,
                            )
                        with gr.Row():
                            random_count = gr.Slider(
                                label="Random count",
                                minimum=0,
                                maximum=8,
                                step=1,
                                value=int(cfg.batch.random_count),
                            )
                            batch_count_b = gr.Slider(
                                label="Batch count",
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=int(cfg.batch.batch_count),
                            )
                            limit = gr.Number(label="Limit (0 = all)", value=0, precision=0)
                        with gr.Row():
                            skip_exists = gr.Checkbox(
                                label="Skip exists (output file or history hash)",
                                value=bool(cfg.batch.skip_exists),
                            )
                            least_used = gr.Checkbox(
                                label="Least-used loop first",
                                value=bool(cfg.batch.least_used_first),
                            )
                        only = gr.Textbox(label="Only loop LoRAs (comma-separated)")
                        exclude = gr.Textbox(label="Exclude loop LoRAs (comma-separated)")
                        slot_text = gr.Textbox(
                            label="Slot overrides",
                            lines=2,
                            placeholder="pose=standing,kneeling\nview=from side,from behind",
                        )
                        with gr.Accordion("Generation settings", open=True):
                            with gr.Row():
                                b_sampler = gr.Dropdown(
                                    label="Sampling method",
                                    choices=initial["samplers"],
                                    value=initial["sampler"],
                                )
                                b_scheduler = gr.Dropdown(
                                    label="Schedule type",
                                    choices=initial["schedulers"],
                                    value=initial["scheduler"],
                                )
                                b_steps = gr.Slider(
                                    label="Steps",
                                    minimum=1,
                                    maximum=150,
                                    step=1,
                                    value=int(cfg.generation.steps),
                                )
                            with gr.Row():
                                b_width = gr.Slider(
                                    label="Width",
                                    minimum=64,
                                    maximum=2048,
                                    step=8,
                                    value=int(cfg.generation.width),
                                )
                                b_height = gr.Slider(
                                    label="Height",
                                    minimum=64,
                                    maximum=2048,
                                    step=8,
                                    value=int(cfg.generation.height),
                                )
                                b_cfg = gr.Slider(
                                    label="CFG Scale",
                                    minimum=1,
                                    maximum=20,
                                    step=0.5,
                                    value=float(cfg.generation.cfg),
                                )
                        with gr.Accordion("ADetailer", open=False):
                            b_ad_enable = gr.Checkbox(
                                label="Enable ADetailer",
                                value=bool(cfg.generation.adetailer_enable),
                            )
                            b_ad_model = gr.Dropdown(
                                label="Detector model",
                                choices=initial["ad_models"],
                                value=initial["ad_model"],
                                allow_custom_value=True,
                            )
                            with gr.Row():
                                b_ad_steps = gr.Slider(
                                    label="Steps",
                                    minimum=1,
                                    maximum=80,
                                    step=1,
                                    value=int(cfg.generation.adetailer_steps),
                                )
                                b_ad_cfg = gr.Slider(
                                    label="CFG",
                                    minimum=1,
                                    maximum=20,
                                    step=0.5,
                                    value=float(cfg.generation.adetailer_cfg),
                                )
                                b_ad_denoise = gr.Slider(
                                    label="Denoise",
                                    minimum=0.05,
                                    maximum=1.0,
                                    step=0.01,
                                    value=float(cfg.generation.adetailer_denoise),
                                )
                        with gr.Row():
                            preview_btn = gr.Button("Preview")
                            prep_btn = gr.Button("Prep jobs")
                            run_zone_btn = gr.Button("Prep + Run", variant="primary")
                            stop_batch_btn = gr.Button("Stop")
                        with gr.Row():
                            job_file = gr.Dropdown(
                                label="Existing job file",
                                choices=initial["jobs"],
                                value=initial["job_file"],
                            )
                            run_jobs_btn = gr.Button("Run job file")
                            resume_errors = gr.Checkbox(label="Retry errors", value=True)
                    with gr.Column(scale=6):
                        batch_gallery = gr.Gallery(
                            label="Batch output",
                            columns=3,
                            height=380,
                            object_fit="contain",
                        )
                        batch_log = gr.Textbox(
                            label="Jobs / log",
                            lines=22,
                            elem_id="batch-log",
                        )

            with gr.Tab("History"):
                gr.Markdown(f"Recent images in `{cfg.paths.output}`")
                history_refresh = gr.Button("Refresh history")
                history_gallery = gr.Gallery(
                    label="outputs/",
                    columns=4,
                    height=640,
                    object_fit="contain",
                    value=gallery_items(list_output_images(cfg)),
                )

        last_seed = gr.State(int(cfg.generation.seed))

        def refresh_all():
            old = ctx["cfg"]
            source = old.source
            reloaded = load_config(source) if source else load_config()
            ctx["cfg"] = apply_overrides(
                reloaded,
                host=old.comfyui.host,
                port=old.comfyui.port,
            )
            opt = load_options()
            return (
                opt["status"],
                gr.update(choices=opt["checkpoints"], value=opt["checkpoint"]),
                gr.update(choices=opt["workflows"], value=opt["workflow"]),
                gr.update(choices=opt["clips"], value=opt["clip"]),
                gr.update(choices=opt["clip2s"], value=opt["clip2"]),
                gr.update(choices=opt["vaes"], value=opt["vae"]),
                gr.update(choices=opt["loras"], value=opt["lora"]),
                gr.update(choices=opt["embeddings"], value=opt["embedding"]),
                gr.update(choices=opt["hypernetworks"], value=opt["hypernetwork"]),
                gr.update(choices=opt["samplers"], value=opt["sampler"]),
                gr.update(choices=opt["schedulers"], value=opt["scheduler"]),
                gr.update(choices=opt["upscalers"], value=opt["upscaler"]),
                gr.update(choices=opt["ad_models"], value=opt["ad_model"]),
                gr.update(
                    choices=opt["anima_model_patches"],
                    value=opt["anima_model_patch"],
                ),
                gr.update(choices=opt["zones"], value=opt["zone"]),
                gr.update(choices=opt["prompts"], value=opt["prompt_config"]),
                gr.update(choices=opt["jobs"], value=opt["job_file"]),
                gr.update(choices=opt["samplers"], value=opt["sampler"]),
                gr.update(choices=opt["schedulers"], value=opt["scheduler"]),
                gr.update(choices=opt["ad_models"], value=opt["ad_model"]),
                gr.update(choices=opt["v_unets"], value=opt["v_unet"]),
                gr.update(choices=opt["v_clips"], value=opt["v_clip"]),
                gr.update(choices=opt["v_vaes"], value=opt["v_vae"]),
                gr.update(choices=opt["v_audio_vaes"], value=opt["v_audio_vae"]),
                gr.update(choices=opt["v_recipes"], value=opt["v_recipe"]),
                gr.update(choices=opt["v_acc_files"], value=opt["v_acc"]),
                opt["v_steps"],
                opt["v_latent_upscale"],
                opt["v_rtx_vsr"],
            )

        refresh_btn.click(
            refresh_all,
            outputs=[
                status_md, checkpoint, workflow, clip, clip2, vae, lora_pick, emb_pick, hn_pick,
                sampler, scheduler, hires_upscaler, ad_model, anima_model_patch,
                zone, prompt_config, job_file,
                b_sampler, b_scheduler, b_ad_model, v_unet, v_clip, v_vae, v_audio_vae,
                v_recipe, v_acc, v_steps, v_latent_up, v_rtx,
            ],
        )

        def on_lora_search(query):
            choices = filter_choices(query, ctx.get("loras") or [])
            return gr.update(choices=choices, value=(choices[0] if choices else None))

        def on_emb_search(query):
            choices = filter_choices(query, ctx.get("embeddings") or [])
            return gr.update(choices=choices, value=(choices[0] if choices else None))

        def on_hn_search(query):
            choices = filter_choices(query, ctx.get("hypernetworks") or [])
            return gr.update(choices=choices, value=(choices[0] if choices else None))

        lora_search.change(on_lora_search, inputs=[lora_search], outputs=[lora_pick])
        lora_add.click(insert_lora_tag, inputs=[prompt, lora_pick, lora_weight], outputs=[prompt])
        emb_search.change(on_emb_search, inputs=[emb_search], outputs=[emb_pick])
        emb_add.click(insert_embedding_tag, inputs=[prompt, emb_pick, emb_weight], outputs=[prompt])
        emb_neg.click(insert_embedding_tag, inputs=[negative, emb_pick, emb_weight], outputs=[negative])
        hn_search.change(on_hn_search, inputs=[hn_search], outputs=[hn_pick])

        def on_size_preset(label):
            parsed = parse_size_preset(label)
            if not parsed:
                return gr.update(), gr.update()
            w, h = parsed
            return w, h

        size_preset.change(on_size_preset, inputs=[size_preset], outputs=[width, height])
        swap_wh.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height])
        dice_btn.click(lambda: -1, outputs=[seed])
        recycle_btn.click(lambda s: s, inputs=[last_seed], outputs=[seed])

        def interrupt():
            ctx["stop"].set()
            client = ctx.get("client")
            try:
                (client or live_client()).interrupt()
            except Exception as exc:
                return f"Interrupt failed: {exc}"
            return "Interrupt sent"

        interrupt_btn.click(interrupt, outputs=[gen_status])
        stop_batch_btn.click(interrupt, outputs=[batch_log])

        def run_txt2img(
            prompt_s, negative_s, checkpoint_s, clip_s, clip2_s, vae_s, workflow_s, sampler_s, scheduler_s,
            steps_v, width_v, height_v, cfg_v, batch_count_v, batch_size_v,
            seed_v, denoise_v, prefix_s,
            img2img_img, img2img_dn,
            hires_on, hires_up, hires_sc, hires_st, hires_dn,
            anima_on, anima_img, anima_pre, anima_c_low, anima_c_high,
            anima_patch, anima_weight, anima_start_v, anima_end_v,
            ad_on, ad_mod, ad_st, ad_cf, ad_dn,
        ):
            if not ctx["busy"].acquire(blocking=False):
                yield gallery_items([]), "Already running — wait or Interrupt.", "", ctx["last_seed"]
                return
            ctx["stop"].clear()
            result_q: queue.Queue = queue.Queue()
            images: list[str] = []
            info = ""
            used_seed = int(seed_v) if seed_v is not None else -1

            def worker():
                nonlocal images, info, used_seed
                client = live_client()
                try:
                    if not client.ping():
                        raise ComfyError(f"ComfyUI is not reachable at {ctx['cfg'].comfyui.base_url}")
                    count = max(1, int(batch_count_v or 1))
                    seed_i = int(seed_v) if seed_v is not None else -1
                    resolver = LoRAResolver.from_client(client, cfg=ctx["cfg"])
                    anima_controls = anima_lllite_controls(
                        bool(anima_on),
                        anima_img,
                        anima_patch,
                        float(anima_weight),
                        float(anima_start_v),
                        float(anima_end_v),
                        anima_pre,
                        float(anima_c_low),
                        float(anima_c_high),
                    )
                    init_path = str(img2img_img or "").strip()
                    use_img2img = bool(init_path)
                    try:
                        eff_denoise = float(img2img_dn) if use_img2img else float(denoise_v)
                    except (TypeError, ValueError):
                        raise ValueError("Denoise must be a number")
                    if use_img2img and not 0.0 < eff_denoise <= 1.0:
                        raise ValueError("img2img Denoise must satisfy 0 < denoise <= 1")
                    for i in range(count):
                        if ctx["stop"].is_set():
                            result_q.put(("status", "Stopped"))
                            break
                        this_seed = -1 if seed_i < 0 else seed_i + i
                        cfg_run = current_cfg(
                            checkpoint=checkpoint_s or None,
                            clip=clip_s or None,
                            clip2=clip2_s or None,
                            vae=selected_vae(vae_s),
                            lora_mode=ctx.get("lora_mode"),
                            workflow_file=workflow_s or None,
                            sampler=sampler_s or None,
                            scheduler=scheduler_s or None,
                            steps=int(steps_v),
                            width=int(width_v),
                            height=int(height_v),
                            cfg=float(cfg_v),
                            batch_size=int(batch_size_v),
                            denoise=float(eff_denoise),
                            init_image=init_path if use_img2img else "",
                            filename_prefix=prefix_s or None,
                            seed=this_seed,
                            save_locally=True,
                            hires_enable=bool(hires_on),
                            hires_upscaler=hires_up or "latent",
                            hires_scale=float(hires_sc),
                            hires_steps=int(hires_st),
                            hires_denoise=float(hires_dn),
                            anima_lllite=anima_controls,
                            adetailer_enable=bool(ad_on),
                            adetailer_model=ad_mod or "",
                            adetailer_steps=int(ad_st),
                            adetailer_cfg=float(ad_cf),
                            adetailer_denoise=float(ad_dn),
                        )

                        def on_progress(msg):
                            result_q.put(("status", format_progress(msg)))

                        result_q.put(("status", f"Queueing {i + 1}/{count}..."))
                        result = generate(
                            prompt=prompt_s,
                            negative=negative_s,
                            seed=this_seed,
                            cfg=cfg_run,
                            client=client,
                            resolver=resolver,
                            wait=True,
                            on_progress=on_progress,
                        )
                        images.extend(collect_saved(result, client, cfg_run.paths.output))
                        used_seed = result.seed
                        ctx["last_seed"] = result.seed
                        info = format_infotext(
                            positive=result.positive,
                            negative=result.negative,
                            steps=int(steps_v),
                            sampler=str(sampler_s or cfg_run.generation.sampler),
                            scheduler=str(scheduler_s or cfg_run.generation.scheduler),
                            cfg=float(cfg_v),
                            seed=result.seed,
                            width=int(width_v),
                            height=int(height_v),
                            checkpoint=str(checkpoint_s or cfg_run.generation.checkpoint),
                            loras=result.loras,
                            hires={
                                "enable": bool(hires_on),
                                "scale": hires_sc,
                                "upscaler": hires_up,
                                "steps": hires_st,
                                "denoise": hires_dn,
                            },
                            adetailer={"enable": bool(ad_on), "model": ad_mod},
                            img2img={"enable": bool(use_img2img), "denoise": eff_denoise},
                        )
                        if count > 1 and seed_i < 0:
                            info += f"\nBatch {i + 1}/{count}"
                        if use_img2img:
                            info += f"\nimg2img: {Path(init_path).name}, denoise {eff_denoise}"
                        if anima_controls:
                            control = anima_controls[0]
                            pre = str(control.get("preprocess") or "none")
                            pre_note = (
                                f", Canny {control.get('canny_low')}-{control.get('canny_high')}"
                                if pre == "canny" else ", preprocessed map"
                            )
                            info += (
                                f"\nAnima LLLite: {control['model_patch']}, "
                                f"Weight: {control['strength']}, "
                                f"Schedule: {control['start_percent']}-{control['end_percent']}"
                                f"{pre_note}"
                            )
                        result_q.put(("partial", (list(images), info, used_seed)))
                    result_q.put(("done", (list(images), info, used_seed)))
                except Exception as exc:
                    result_q.put(("error", str(exc)))
                finally:
                    ctx["busy"].release()

            threading.Thread(target=worker, daemon=True).start()
            out_images: list[str] = []
            out_info = ""
            out_seed = used_seed
            yield gallery_items(out_images), "Starting...", out_info, out_seed
            while True:
                kind, payload = result_q.get()
                if kind == "status":
                    yield gallery_items(out_images), payload, out_info, out_seed
                elif kind == "partial":
                    out_images, out_info, out_seed = payload
                    yield gallery_items(out_images), f"Generated {len(out_images)} image(s)", out_info, out_seed
                elif kind == "error":
                    yield gallery_items(out_images), f"Error: {payload}", payload, out_seed
                    break
                elif kind == "done":
                    out_images, out_info, out_seed = payload
                    yield gallery_items(out_images), "Done", out_info, out_seed
                    break

        txt2img_inputs = [
            prompt, negative, checkpoint, clip, clip2, vae, workflow, sampler, scheduler,
            steps, width, height, cfg_scale, batch_count, batch_size,
            seed, denoise, prefix,
            img2img_image, img2img_denoise,
            hires_enable, hires_upscaler, hires_scale, hires_steps, hires_denoise,
            anima_enable, anima_image, anima_preprocess, anima_canny_low, anima_canny_high,
            anima_model_patch, anima_strength, anima_start, anima_end,
            ad_enable, ad_model, ad_steps, ad_cfg, ad_denoise,
        ]
        txt2img_outputs = [gallery, gen_status, infotext, last_seed]
        generate_btn.click(run_txt2img, inputs=txt2img_inputs, outputs=txt2img_outputs)

        def on_workflow_change(name):
            preset = WORKFLOW_PRESETS.get(Path(str(name or "")).name, {})
            ctx["lora_mode"] = preset.get("lora_mode", "full")
            ck = preset.get("checkpoint")
            cl = preset.get("clip")
            cl2 = preset.get("clip2")
            va = preset.get("vae")
            return (
                ck if ck else gr.update(),
                cl if cl else gr.update(),
                cl2 if cl2 else gr.update(),
                va if va else gr.update(),
                preset.get("steps", gr.update()),
                preset.get("cfg", gr.update()),
                preset.get("sampler", gr.update()),
                preset.get("scheduler", gr.update()),
                preset.get("width", gr.update()),
                preset.get("height", gr.update()),
            )

        workflow.change(
            on_workflow_change,
            inputs=[workflow],
            outputs=[checkpoint, clip, clip2, vae, steps, cfg_scale, sampler, scheduler, width, height],
        )

        def run_video(image_path, last_path, ref_img_files, ref_vid_files, ref_aud_files, mode_s, ref_size_s, prompt_s, width_v, height_v, auto_size, frames_v, steps_v, seed_v, unet_s, clip_s, vae_s, audio_vae_s, recipe_s, acc_s, loop_on, latent_on, rtx_on):
            if not ctx["busy"].acquire(blocking=False):
                yield None, "Already running"
                return
            ctx["stop"].clear()
            result_q: queue.Queue = queue.Queue()

            def _paths(files):
                if not files:
                    return []
                if isinstance(files, (str, Path)):
                    return [str(files)]
                return [str(getattr(f, "name", f)) for f in files if str(getattr(f, "name", f) or "").strip()]

            def worker():
                try:
                    is_r2v = str(mode_s or "i2v").lower() == "r2v"
                    ref_imgs = _paths(ref_img_files)
                    ref_vids = _paths(ref_vid_files)
                    ref_auds = _paths(ref_aud_files)
                    if is_r2v and image_path and not ref_imgs and not ref_vids:
                        ref_imgs = [str(image_path)]
                    if not is_r2v and not image_path:
                        raise ValueError("Upload a first-frame still")
                    if is_r2v and not image_path and not ref_imgs and not ref_vids and not (prompt_s or "").strip():
                        raise ValueError("R2V needs a prompt or at least one reference")
                    mp = ctx["cfg"].video.size_megapixels
                    size_src = image_path or (ref_imgs[0] if ref_imgs else None)
                    if auto_size and size_src:
                        w, h = target_dimensions(size_src, megapixels=mp)
                    else:
                        w, h = h3_size(width_v or 0, height_v or 0, fit=False)
                    client = live_client()
                    if not client.ping():
                        raise ComfyError(f"ComfyUI is not reachable at {ctx['cfg'].comfyui.base_url}")

                    def on_progress(msg):
                        result_q.put(("status", format_progress(msg)))

                    result_q.put(("status", f"Queueing MiniMax H3 {'R2V-lite' if is_r2v else 'I2V'} at {w}×{h}..."))
                    result = generate_video(
                        prompt_s,
                        image=None if is_r2v else image_path,
                        last_frame=None if is_r2v else last_path,
                        mode="r2v" if is_r2v else "i2v",
                        ref_images=ref_imgs if is_r2v else None,
                        ref_videos=ref_vids if is_r2v else None,
                        ref_audios=ref_auds if is_r2v else None,
                        ref_image_size=ref_size_s if is_r2v else None,
                        seed=int(seed_v) if seed_v is not None else -1,
                        steps=int(steps_v),
                        width=w,
                        height=h,
                        frames=int(frames_v),
                        unet=unet_s,
                        clip=clip_s,
                        vae=vae_s,
                        audio_vae=audio_vae_s,
                        recipe=recipe_s,
                        acc_file=acc_s or "",
                        loop=bool(loop_on),
                        latent_upscale=bool(latent_on),
                        rtx_vsr=bool(rtx_on),
                        cfg=ctx["cfg"],
                        client=client,
                        on_progress=on_progress,
                    )
                    clip_path = None
                    for path in reversed(result.saved_files or []):
                        if Path(path).suffix.lower() in VIDEO_SUFFIXES:
                            clip_path = path
                            break
                    note = "Done"
                    if result.saved_files:
                        note = "Saved: " + ", ".join(Path(p).name for p in result.saved_files)
                    result_q.put(("done", (clip_path, note)))
                except Exception as exc:
                    result_q.put(("error", str(exc)))
                finally:
                    ctx["busy"].release()

            threading.Thread(target=worker, daemon=True).start()
            video_path = None
            yield None, "Starting..."
            while True:
                kind, payload = result_q.get()
                if kind == "status":
                    yield video_path, payload
                elif kind == "error":
                    yield video_path, f"Error: {payload}"
                    break
                elif kind == "done":
                    video_path, note = payload
                    yield video_path, note
                    break

        def on_video_recipe(recipe_key, unet_s):
            files = ctx.get("acc_files") or []
            rec, steps, acc = video_recipe_ui(
                recipe_key, unet_s, files, ctx["cfg"].video.acc_file,
            )
            choices = prefer_first(files, acc) if rec.acc else ([""] + prefer_first(files, None))
            return steps, gr.update(choices=choices, value=acc)

        v_recipe.change(
            on_video_recipe,
            inputs=[v_recipe, v_unet],
            outputs=[v_steps, v_acc],
        )

        def on_video_unet(unet_s, recipe_key):
            files = ctx.get("acc_files") or []
            rec, _steps, acc = video_recipe_ui(
                recipe_key, unet_s, files, ctx["cfg"].video.acc_file,
            )
            if not rec.acc:
                return ""
            return acc

        v_unet.change(on_video_unet, inputs=[v_unet, v_recipe], outputs=[v_acc])

        def on_video_size_from_image(image_path, auto):
            if not auto or not image_path:
                return gr.update(), gr.update()
            try:
                w, h = target_dimensions(image_path, megapixels=ctx["cfg"].video.size_megapixels)
            except Exception:
                return gr.update(), gr.update()
            return w, h

        v_image.change(
            on_video_size_from_image,
            inputs=[v_image, v_auto_size],
            outputs=[v_width, v_height],
        )
        v_auto_size.change(
            on_video_size_from_image,
            inputs=[v_image, v_auto_size],
            outputs=[v_width, v_height],
        )

        v_generate.click(
            run_video,
            inputs=[
                v_image, v_last, v_ref_images, v_ref_video, v_ref_audio, v_mode, v_ref_size,
                v_prompt, v_width, v_height, v_auto_size, v_frames, v_steps, v_seed,
                v_unet, v_clip, v_vae, v_audio_vae, v_recipe, v_acc, v_loop, v_latent_up, v_rtx,
            ],
            outputs=[v_video, v_status],
        )

        def _batch_cfg(checkpoint_s, workflow_s, sampler_s, scheduler_s, steps_v,
                       width_v, height_v, cfg_v, batch_size_v, denoise_v, prefix_s,
                       random_count_v, batch_count_v, skip_exists_v, least_used_v,
                       ad_on=False, ad_mod="", ad_st=20, ad_cf=5.0, ad_dn=0.4,
                       vae_s=None):
            return current_cfg(
                checkpoint=checkpoint_s or None,
                vae=selected_vae(vae_s),
                workflow_file=workflow_s or None,
                sampler=sampler_s or None,
                scheduler=scheduler_s or None,
                steps=int(steps_v),
                width=int(width_v),
                height=int(height_v),
                cfg=float(cfg_v),
                batch_size=int(batch_size_v),
                denoise=float(denoise_v),
                filename_prefix=prefix_s or None,
                random_count=int(random_count_v),
                batch_count=int(batch_count_v),
                skip_exists=bool(skip_exists_v),
                least_used_first=bool(least_used_v),
                save_locally=True,
                hires_enable=False,
                adetailer_enable=bool(ad_on),
                adetailer_model=ad_mod or "",
                adetailer_steps=int(ad_st),
                adetailer_cfg=float(ad_cf),
                adetailer_denoise=float(ad_dn),
            )

        def _resolver_for(cfg_run: WrapperConfig, offline: bool = False) -> LoRAResolver:
            if offline:
                return LoRAResolver(cfg=cfg_run)
            client = live_client()
            try:
                return LoRAResolver.from_client(client, cfg=cfg_run)
            except ComfyError:
                return LoRAResolver(cfg=cfg_run)

        def do_preview(
            zone_s, prompt_s, checkpoint_s, workflow_s, sampler_s, scheduler_s,
            steps_v, width_v, height_v, cfg_v, batch_size_v, denoise_v, prefix_s,
            random_count_v, batch_count_v, skip_exists_v, least_used_v,
            only_s, exclude_s, slot_s, limit_v,
            ad_on, ad_mod, ad_st, ad_cf, ad_dn, vae_s,
        ):
            try:
                cfg_run = _batch_cfg(
                    checkpoint_s, workflow_s, sampler_s, scheduler_s, steps_v,
                    width_v, height_v, cfg_v, batch_size_v, denoise_v, prefix_s,
                    random_count_v, batch_count_v, skip_exists_v, least_used_v,
                    ad_on, ad_mod, ad_st, ad_cf, ad_dn, vae_s,
                )
                extra_slots = parse_slot_text(slot_s)
                lim = int(limit_v or 0)
                jobs = prep_jobs(
                    cfg_run,
                    zone_name=zone_s or None,
                    prompt_name=prompt_s or None,
                    resolver=_resolver_for(cfg_run, offline=True),
                    extra_slots=extra_slots or None,
                    only=split_csv(only_s),
                    exclude=split_csv(exclude_s),
                    limit=lim or None,
                )
                return preview_jobs(jobs)
            except Exception as exc:
                return f"error: {exc}"

        batch_shared = [
            zone, prompt_config, checkpoint, workflow, b_sampler, b_scheduler,
            b_steps, b_width, b_height, b_cfg, batch_size, denoise, prefix,
            random_count, batch_count_b, skip_exists, least_used,
            only, exclude, slot_text, limit,
            b_ad_enable, b_ad_model, b_ad_steps, b_ad_cfg, b_ad_denoise,
            vae,
        ]
        preview_btn.click(do_preview, inputs=batch_shared, outputs=[batch_log])

        def do_prep(*args):
            try:
                (
                    zone_s, prompt_s, checkpoint_s, workflow_s, sampler_s, scheduler_s,
                    steps_v, width_v, height_v, cfg_v, batch_size_v, denoise_v, prefix_s,
                    random_count_v, batch_count_v, skip_exists_v, least_used_v,
                    only_s, exclude_s, slot_s, limit_v,
                    ad_on, ad_mod, ad_st, ad_cf, ad_dn, vae_s,
                ) = args
                cfg_run = _batch_cfg(
                    checkpoint_s, workflow_s, sampler_s, scheduler_s, steps_v,
                    width_v, height_v, cfg_v, batch_size_v, denoise_v, prefix_s,
                    random_count_v, batch_count_v, skip_exists_v, least_used_v,
                    ad_on, ad_mod, ad_st, ad_cf, ad_dn, vae_s,
                )
                extra_slots = parse_slot_text(slot_s)
                lim = int(limit_v or 0)
                jobs = prep_jobs(
                    cfg_run,
                    zone_name=zone_s or None,
                    prompt_name=prompt_s or None,
                    resolver=_resolver_for(cfg_run),
                    extra_slots=extra_slots or None,
                    only=split_csv(only_s),
                    exclude=split_csv(exclude_s),
                    limit=lim or None,
                )
                out = cfg_run.paths.jobs / f"jobs-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
                save_jobs(
                    jobs,
                    out,
                    extra={"zone": zone_s, "prompt": prompt_s, "workflow": cfg_run.workflow_file},
                )
                names = list_job_files(cfg_run)
                pending = sum(1 for j in jobs if j.status == "pending")
                skipped = sum(1 for j in jobs if j.status == "skipped")
                text = f"wrote {out}\n{len(jobs)} jobs, {pending} pending, {skipped} skipped\n\n" + preview_jobs(jobs)
                return text, gr.update(choices=names, value=out.name)
            except Exception as exc:
                return f"error: {exc}", gr.update()

        prep_btn.click(do_prep, inputs=batch_shared, outputs=[batch_log, job_file])

        def run_batch_worker(jobs, cfg_run, resume, images_start=None):
            result_q: queue.Queue = queue.Queue()
            images: list[str] = list(images_start or [])
            logs: list[str] = []

            def worker():
                client = live_client()
                try:
                    if not client.ping():
                        raise ComfyError(f"ComfyUI is not reachable at {cfg_run.comfyui.base_url}")
                    resolver = LoRAResolver.from_client(client, cfg=cfg_run)

                    def on_log(msg: str):
                        logs.append(msg)
                        result_q.put(("log", "\n".join(logs)))

                    def on_job(job):
                        for path in job.files or []:
                            if path and path not in images:
                                images.append(path)
                        result_q.put(("images", list(images)))

                    run_jobs(
                        jobs,
                        cfg_run,
                        client=client,
                        resolver=resolver,
                        on_log=on_log,
                        on_job=on_job,
                        resume=resume,
                        stop_flag=lambda: ctx["stop"].is_set(),
                    )
                    result_q.put(("done", ("\n".join(logs), list(images))))
                except Exception as exc:
                    logs.append(f"error: {exc}")
                    result_q.put(("done", ("\n".join(logs), list(images))))
                finally:
                    ctx["busy"].release()

            threading.Thread(target=worker, daemon=True).start()
            log_text = ""
            out_images: list[str] = list(images)
            yield gallery_items(out_images), "Starting batch..."
            while True:
                kind, payload = result_q.get()
                if kind == "log":
                    log_text = payload
                    yield gallery_items(out_images), log_text
                elif kind == "images":
                    out_images = payload
                    yield gallery_items(out_images), log_text
                elif kind == "done":
                    log_text, out_images = payload
                    yield gallery_items(out_images), log_text or "Done"
                    break

        def run_from_zone(*args):
            if not ctx["busy"].acquire(blocking=False):
                yield [], "Already running — wait or Stop."
                return
            ctx["stop"].clear()
            try:
                (
                    zone_s, prompt_s, checkpoint_s, workflow_s, sampler_s, scheduler_s,
                    steps_v, width_v, height_v, cfg_v, batch_size_v, denoise_v, prefix_s,
                    random_count_v, batch_count_v, skip_exists_v, least_used_v,
                    only_s, exclude_s, slot_s, limit_v,
                    ad_on, ad_mod, ad_st, ad_cf, ad_dn, vae_s,
                ) = args
                cfg_run = _batch_cfg(
                    checkpoint_s, workflow_s, sampler_s, scheduler_s, steps_v,
                    width_v, height_v, cfg_v, batch_size_v, denoise_v, prefix_s,
                    random_count_v, batch_count_v, skip_exists_v, least_used_v,
                    ad_on, ad_mod, ad_st, ad_cf, ad_dn, vae_s,
                )
                jobs = prep_jobs(
                    cfg_run,
                    zone_name=zone_s or None,
                    prompt_name=prompt_s or None,
                    resolver=_resolver_for(cfg_run),
                    extra_slots=parse_slot_text(slot_s) or None,
                    only=split_csv(only_s),
                    exclude=split_csv(exclude_s),
                    limit=int(limit_v or 0) or None,
                )
                out = cfg_run.paths.jobs / f"jobs-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
                save_jobs(jobs, out, extra={"zone": zone_s, "prompt": prompt_s})
            except Exception as exc:
                ctx["busy"].release()
                yield [], f"error: {exc}"
                return
            yield from run_batch_worker(jobs, cfg_run, resume=True)

        def run_from_file(job_name, resume, checkpoint_s, workflow_s, sampler_s, scheduler_s,
                          steps_v, width_v, height_v, cfg_v, batch_size_v, denoise_v, prefix_s,
                          random_count_v, batch_count_v, skip_exists_v, least_used_v,
                          ad_on, ad_mod, ad_st, ad_cf, ad_dn, vae_s):
            if not ctx["busy"].acquire(blocking=False):
                yield [], "Already running — wait or Stop."
                return
            ctx["stop"].clear()
            try:
                if not job_name:
                    raise ValueError("Pick a job file (or Prep one first)")
                cfg_run = _batch_cfg(
                    checkpoint_s, workflow_s, sampler_s, scheduler_s, steps_v,
                    width_v, height_v, cfg_v, batch_size_v, denoise_v, prefix_s,
                    random_count_v, batch_count_v, skip_exists_v, least_used_v,
                    ad_on, ad_mod, ad_st, ad_cf, ad_dn, vae_s,
                )
                path = Path(job_name)
                if not path.is_file():
                    path = cfg_run.paths.jobs / job_name
                jobs, _meta = load_jobs(path)
                if resume:
                    for job in jobs:
                        if job.status == "error":
                            job.status = "pending"
                            job.error = ""
            except Exception as exc:
                ctx["busy"].release()
                yield [], f"error: {exc}"
                return
            yield from run_batch_worker(jobs, cfg_run, resume=True)

        run_zone_btn.click(run_from_zone, inputs=batch_shared, outputs=[batch_gallery, batch_log])
        run_jobs_btn.click(
            run_from_file,
            inputs=[
                job_file, resume_errors, checkpoint, workflow, b_sampler, b_scheduler,
                b_steps, b_width, b_height, b_cfg, batch_size, denoise, prefix,
                random_count, batch_count_b, skip_exists, least_used,
                b_ad_enable, b_ad_model, b_ad_steps, b_ad_cfg, b_ad_denoise,
                vae,
            ],
            outputs=[batch_gallery, batch_log],
        )

        def refresh_history():
            return gallery_items(list_output_images(ctx["cfg"]))

        history_refresh.click(refresh_history, outputs=[history_gallery])

    return demo


def launch(
    *,
    cfg: WrapperConfig | None = None,
    config: str | Path | None = None,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
    share: bool = False,
    inbrowser: bool = True,
) -> None:
    if cfg is None:
        cfg = load_config(config)
    gr = _import_gradio()
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate",
    )
    demo = build_ui(cfg)
    if hasattr(demo, "queue"):
        try:
            demo.queue()
        except Exception:
            pass
    allowed = [str(cfg.paths.root), str(cfg.paths.output)]
    launch_kwargs = {
        "server_name": server_name,
        "server_port": int(server_port),
        "share": bool(share),
        "inbrowser": bool(inbrowser),
        "allowed_paths": allowed,
        "show_error": True,
        "css": CSS,
        "theme": theme,
    }
    try:
        demo.launch(**launch_kwargs)
    except TypeError:
        launch_kwargs.pop("css", None)
        launch_kwargs.pop("theme", None)
        demo.launch(**launch_kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comfywrap-ui",
        description="A1111-style Gradio WebUI for the ComfyUI wrapper.",
    )
    parser.add_argument("--config", "-c", help="Path to config.yaml")
    parser.add_argument("--host", help="ComfyUI host (override config)")
    parser.add_argument("--comfy-port", type=int, dest="comfy_port", help="ComfyUI port")
    parser.add_argument("--listen", action="store_true", help="Listen on 0.0.0.0")
    parser.add_argument("--server-name", default=None, help="WebUI bind address")
    parser.add_argument("--port", "-p", type=int, default=7860, help="WebUI port (default 7860)")
    parser.add_argument("--share", action="store_true", help="Gradio share tunnel")
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    return parser


def launch_from_args(args) -> int:
    cfg = load_config(getattr(args, "config", None))
    cfg = apply_overrides(
        cfg,
        host=getattr(args, "host", None),
        port=getattr(args, "comfy_port", None),
    )
    listen = bool(getattr(args, "listen", False))
    server_name = getattr(args, "server_name", None) or ("0.0.0.0" if listen else "127.0.0.1")
    try:
        launch(
            cfg=cfg,
            server_name=server_name,
            server_port=int(getattr(args, "port", None) or 7860),
            share=bool(getattr(args, "share", False)),
            inbrowser=not bool(getattr(args, "no_browser", False)),
        )
    except ImportError as exc:
        print(f"error: {exc}", flush=True)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return launch_from_args(args)


if __name__ == "__main__":
    raise SystemExit(main())
