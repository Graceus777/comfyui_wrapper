"""Load and merge the wrapper YAML config."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from comfyui_wrapper.paths import project_root, resolve_path
from comfyui_wrapper.h3 import normalize_recipe

DEFAULTS: dict[str, Any] = {
    "comfyui": {
        "host": "127.0.0.1",
        "port": 8188,
        "timeout_s": 600,
        "poll_interval_s": 0.5,
    },
    "paths": {
        "workflows": "workflows",
        "zones": "configs/zones",
        "prompts": "configs/prompts",
        "jobs": "configs/jobs",
        "output": "outputs",
        "history": "generation_history.jsonl",
        "combinator_root": "",
        "webui_root": "",
        "comfyui_root": "",
    },
    "workflow": {
        "file": "sdxl_txt2img.api.json",
        "map": {},
    },
    "generation": {
        "checkpoint": "",
        "positive": "",
        "negative": "",
        "steps": 28,
        "cfg": 5.0,
        "sampler": "euler_ancestral",
        "scheduler": "normal",
        "width": 1024,
        "height": 1280,
        "seed": -1,
        "batch_size": 1,
        "denoise": 1.0,
        "filename_prefix": "wrapper",
        "loras": [],
        "lora_mode": "full",
        "include_lora_tags": False,
        "hires": {
            "enable": False,
            "scale": 1.5,
            "steps": 0,
            "denoise": 0.45,
            "upscaler": "latent",
        },
        "adetailer": {
            "enable": False,
            "model": "",
            "denoise": 0.4,
            "steps": 20,
            "cfg": 5.0,
            "guide_size": 512,
        },
    },
    "batch": {
        "random_count": 1,
        "batch_count": 1,
        "cooldown_s": 0,
        "skip_exists": True,
        "least_used_first": True,
        "sample_random_at": "prep",
        "save_locally": True,
        "in_flight": 1,
    },
    "video": {
        "workflow": "minimax_h3_i2v_loop.api.json",
        "recipe": "acc",
        "unet": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
        "clip": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "vae": "minimax_h3_video_vae_fp16.safetensors",
        "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        "acc_file": "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        "frames": 124,
        "fps": 24,
        "steps": 8,
        "sampler": "euler",
        "scheduler": "simple",
        "shift_video": 12.0,
        "shift_audio": 3.0,
        "width": 832,
        "height": 480,
        "size_megapixels": 0.52,
        "loop": True,
        "crossfade_frames": 12,
        "latent_upscale": False,
        "latent_upscale_model": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
        "latent_upscale_scale": 2.0,
        "rtx_vsr": False,
        "rtx_vsr_scale": 2.0,
        "rtx_vsr_quality": "ULTRA",
        "prompt": (
            "integrated_multimodal_description: A single continuous locked-off shot "
            "from the reference image. Natural continuous environmental motion only; "
            "no camera cut, zoom, reversal, or end freeze."
        ),
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


@dataclass
class ComfyUISettings:
    host: str = "127.0.0.1"
    port: int = 8188
    timeout_s: float = 600
    poll_interval_s: float = 0.5

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        return f"ws://{self.host}:{self.port}"


@dataclass
class PathSettings:
    root: Path = field(default_factory=project_root)
    workflows: Path = Path("workflows")
    zones: Path = Path("configs/zones")
    prompts: Path = Path("configs/prompts")
    jobs: Path = Path("configs/jobs")
    output: Path = Path("outputs")
    history: Path = Path("generation_history.jsonl")
    combinator_root: Path | None = None
    webui_root: Path | None = None
    comfyui_root: Path | None = None

    def zone_dirs(self) -> list[Path]:
        dirs = [self.zones]
        if self.combinator_root:
            dirs.append(self.combinator_root / "configs")
        return dirs

    def prompt_dirs(self) -> list[Path]:
        dirs = [self.prompts]
        if self.combinator_root:
            dirs.append(self.combinator_root / "configs" / "prompts")
        return dirs

    def lora_text_dirs(self) -> list[Path]:
        dirs = [self.root / "lora_texts"]
        if self.combinator_root:
            dirs.append(self.combinator_root / "lora_texts")
        webui = self.webui_root
        if webui is None and self.combinator_root and self.combinator_root.parent.name.lower() == "extensions":
            webui = self.combinator_root.parent.parent
        if webui:
            dirs.append(webui / "models" / "Lora")
        return dirs


@dataclass
class GenerationSettings:
    checkpoint: str = ""
    positive: str = ""
    negative: str = ""
    steps: int = 28
    cfg: float = 5.0
    sampler: str = "euler_ancestral"
    scheduler: str = "normal"
    width: int = 1024
    height: int = 1280
    seed: int = -1
    batch_size: int = 1
    denoise: float = 1.0
    filename_prefix: str = "wrapper"
    clip: str = ""
    clip2: str = ""
    vae: str = ""
    loras: list = field(default_factory=list)
    lora_mode: str = "full"
    include_lora_tags: bool = False
    hires_enable: bool = False
    hires_scale: float = 1.5
    hires_steps: int = 0
    hires_denoise: float = 0.45
    hires_upscaler: str = "latent"
    adetailer_enable: bool = False
    adetailer_model: str = ""
    adetailer_denoise: float = 0.4
    adetailer_steps: int = 20
    adetailer_cfg: float = 5.0
    adetailer_guide_size: int = 512


@dataclass
class VideoSettings:
    workflow_file: str = "minimax_h3_i2v_loop.api.json"
    recipe: str = "acc"
    unet: str = "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
    clip: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
    vae: str = "minimax_h3_video_vae_fp16.safetensors"
    audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors"
    acc_file: str = "MiniMax-H3-FL2VA-Acc-8Step.safetensors"
    frames: int = 124
    fps: int = 24
    steps: int = 8
    sampler: str = "euler"
    scheduler: str = "simple"
    shift_video: float = 12.0
    shift_audio: float = 3.0
    width: int = 832
    height: int = 480
    size_megapixels: float = 0.52
    loop: bool = True
    crossfade_frames: int = 12
    latent_upscale: bool = False
    latent_upscale_model: str = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
    latent_upscale_scale: float = 2.0
    rtx_vsr: bool = False
    rtx_vsr_scale: float = 2.0
    rtx_vsr_quality: str = "ULTRA"
    prompt: str = ""


@dataclass
class BatchSettings:
    random_count: int = 1
    batch_count: int = 1
    cooldown_s: float = 0
    skip_exists: bool = True
    least_used_first: bool = True
    sample_random_at: str = "prep"
    save_locally: bool = True
    in_flight: int = 1


@dataclass
class WrapperConfig:
    raw: dict = field(default_factory=dict)
    source: Path | None = None
    comfyui: ComfyUISettings = field(default_factory=ComfyUISettings)
    paths: PathSettings = field(default_factory=PathSettings)
    workflow_file: str = "sdxl_txt2img.api.json"
    workflow_map: dict = field(default_factory=dict)
    generation: GenerationSettings = field(default_factory=GenerationSettings)
    batch: BatchSettings = field(default_factory=BatchSettings)
    video: VideoSettings = field(default_factory=VideoSettings)

    def video_workflow_path(self) -> Path:
        given = Path(self.video.workflow_file)
        if given.is_absolute() and given.exists():
            return given
        return self.paths.workflows / given.name

    def workflow_path(self) -> Path:
        given = Path(self.workflow_file)
        if given.is_absolute() and given.exists():
            return given
        direct = self.paths.workflows / given.name if given.parent == Path(".") else self.paths.root / given
        if direct.exists():
            return direct
        nested = self.paths.workflows / given
        return nested


def _as_path(root: Path, value: str | Path) -> Path:
    return resolve_path(value, root=root)


def from_dict(data: dict, *, source: Path | None = None, root: Path | None = None) -> WrapperConfig:
    merged = deep_merge(DEFAULTS, data or {})
    root = root or (source.parent if source else project_root())
    p = merged["paths"]
    combinator = str(p.get("combinator_root") or "").strip()
    webui_root = str(p.get("webui_root") or "").strip()
    comfyui_root = str(p.get("comfyui_root") or "").strip()
    paths = PathSettings(
        root=root,
        workflows=_as_path(root, p["workflows"]),
        zones=_as_path(root, p["zones"]),
        prompts=_as_path(root, p["prompts"]),
        jobs=_as_path(root, p["jobs"]),
        output=_as_path(root, p["output"]),
        history=_as_path(root, p["history"]),
        combinator_root=Path(combinator) if combinator else None,
        webui_root=Path(webui_root) if webui_root else None,
        comfyui_root=Path(comfyui_root) if comfyui_root else None,
    )
    g = merged["generation"]
    v = merged.get("video") if isinstance(merged.get("video"), dict) else {}
    hires = g.get("hires") if isinstance(g.get("hires"), dict) else {}
    ad = g.get("adetailer") if isinstance(g.get("adetailer"), dict) else {}
    b = merged["batch"]
    c = merged["comfyui"]
    w = merged["workflow"]
    return WrapperConfig(
        raw=merged,
        source=source,
        comfyui=ComfyUISettings(
            host=str(c["host"]),
            port=int(c["port"]),
            timeout_s=float(c["timeout_s"]),
            poll_interval_s=float(c["poll_interval_s"]),
        ),
        paths=paths,
        workflow_file=str(w.get("file") or "sdxl_txt2img.api.json"),
        workflow_map=dict(w.get("map") or {}),
        generation=GenerationSettings(
            checkpoint=str(g.get("checkpoint") or ""),
            positive=str(g.get("positive") or ""),
            negative=str(g.get("negative") or ""),
            steps=int(g["steps"]),
            cfg=float(g["cfg"]),
            sampler=str(g["sampler"]),
            scheduler=str(g["scheduler"]),
            width=int(g["width"]),
            height=int(g["height"]),
            seed=int(g["seed"]),
            batch_size=int(g["batch_size"]),
            denoise=float(g["denoise"]),
            filename_prefix=str(g.get("filename_prefix") or "wrapper"),
            clip=str(g.get("clip") or ""),
            clip2=str(g.get("clip2") or ""),
            vae=str(g.get("vae") or ""),
            loras=list(g.get("loras") or []),
            lora_mode=str(g.get("lora_mode") or "full"),
            include_lora_tags=bool(g.get("include_lora_tags") or False),
            hires_enable=bool(hires.get("enable", g.get("hires_enable") or False)),
            hires_scale=float(hires.get("scale") if hires.get("scale") is not None else 1.5),
            hires_steps=int(hires.get("steps") if hires.get("steps") is not None else 0),
            hires_denoise=float(hires.get("denoise") if hires.get("denoise") is not None else 0.45),
            hires_upscaler=str(hires.get("upscaler") or "latent"),
            adetailer_enable=bool(ad.get("enable", g.get("adetailer_enable") or False)),
            adetailer_model=str(ad.get("model") or ""),
            adetailer_denoise=float(ad.get("denoise") if ad.get("denoise") is not None else 0.4),
            adetailer_steps=int(ad.get("steps") if ad.get("steps") is not None else 20),
            adetailer_cfg=float(ad.get("cfg") if ad.get("cfg") is not None else 5.0),
            adetailer_guide_size=int(ad.get("guide_size") if ad.get("guide_size") is not None else 512),
        ),
        batch=BatchSettings(
            random_count=int(b["random_count"]),
            batch_count=int(b["batch_count"]),
            cooldown_s=float(b["cooldown_s"]),
            skip_exists=bool(b["skip_exists"]),
            least_used_first=bool(b["least_used_first"]),
            sample_random_at=str(b.get("sample_random_at") or "prep"),
            save_locally=bool(b["save_locally"]),
            in_flight=max(1, int(b.get("in_flight") or 1)),
        ),
        video=VideoSettings(
            workflow_file=str(v.get("workflow") or "minimax_h3_i2v_loop.api.json"),
            recipe=normalize_recipe(v.get("recipe") or "acc"),
            unet=str(v.get("unet") or "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"),
            clip=str(v.get("clip") or "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"),
            vae=str(v.get("vae") or "minimax_h3_video_vae_fp16.safetensors"),
            audio_vae=str(v.get("audio_vae") or "minimax_h3_audio_vae_fp32.safetensors"),
            acc_file=str(v.get("acc_file") or "MiniMax-H3-FL2VA-Acc-8Step.safetensors"),
            frames=int(v.get("frames") or 124),
            fps=int(v.get("fps") or 24),
            steps=int(v.get("steps") or 8),
            sampler=str(v.get("sampler") or "euler"),
            scheduler=str(v.get("scheduler") or "simple"),
            shift_video=float(v.get("shift_video") if v.get("shift_video") is not None else 12.0),
            shift_audio=float(v.get("shift_audio") if v.get("shift_audio") is not None else 3.0),
            width=int(v.get("width") or 832),
            height=int(v.get("height") or 480),
            size_megapixels=float(
                v.get("size_megapixels") if v.get("size_megapixels") is not None else 0.52
            ),
            loop=bool(v.get("loop") if v.get("loop") is not None else True),
            crossfade_frames=int(v.get("crossfade_frames") or 12),
            latent_upscale=bool(v.get("latent_upscale") or False),
            latent_upscale_model=str(
                v.get("latent_upscale_model") or "minimax_h3_latent_upscaler_3d_fp16.safetensors"
            ),
            latent_upscale_scale=float(
                v.get("latent_upscale_scale") if v.get("latent_upscale_scale") is not None else 2.0
            ),
            rtx_vsr=bool(v.get("rtx_vsr") or False),
            rtx_vsr_scale=float(v.get("rtx_vsr_scale") if v.get("rtx_vsr_scale") is not None else 2.0),
            rtx_vsr_quality=str(v.get("rtx_vsr_quality") or "ULTRA"),
            prompt=str(v.get("prompt") or ""),
        ),
    )


def load_config(path: str | Path | None = None) -> WrapperConfig:
    """Load config.yaml from an explicit path, or project_root/config.yaml."""
    if path is None:
        path = project_root() / "config.yaml"
    path = Path(path)
    if not path.exists():
        return from_dict({}, source=None, root=project_root())
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return from_dict(data, source=path, root=path.parent)


def apply_overrides(config: WrapperConfig, **kwargs: Any) -> WrapperConfig:
    """Return a copy with generation/batch/workflow fields overridden.

    Unknown keys are ignored. Nested dicts merge into the raw config.
    """
    if kwargs.get("cfg_scale") is not None:
        kwargs["cfg"] = kwargs.pop("cfg_scale")
    else:
        kwargs.pop("cfg_scale", None)
    raw = deepcopy(config.raw)
    gen_keys = {
        "checkpoint", "positive", "negative", "steps", "cfg", "sampler",
        "scheduler", "width", "height", "seed", "batch_size", "denoise",
        "filename_prefix", "loras", "lora_mode", "include_lora_tags",
        "clip", "clip2", "vae",
    }
    video_keys = {
        "video_workflow", "video_unet", "video_clip", "video_vae", "video_audio_vae",
        "video_recipe", "video_acc_file", "frames", "fps", "length", "loop",
        "crossfade_frames", "shift_video", "shift_audio", "size_megapixels",
        "latent_upscale", "latent_upscale_model", "latent_upscale_scale",
        "rtx_vsr", "rtx_vsr_scale", "rtx_vsr_quality",
    }
    nested_keys = {
        "hires_enable": ("hires", "enable"),
        "hires_scale": ("hires", "scale"),
        "hires_steps": ("hires", "steps"),
        "hires_denoise": ("hires", "denoise"),
        "hires_upscaler": ("hires", "upscaler"),
        "adetailer_enable": ("adetailer", "enable"),
        "adetailer_model": ("adetailer", "model"),
        "adetailer_denoise": ("adetailer", "denoise"),
        "adetailer_steps": ("adetailer", "steps"),
        "adetailer_cfg": ("adetailer", "cfg"),
        "adetailer_guide_size": ("adetailer", "guide_size"),
    }
    batch_keys = {
        "random_count", "batch_count", "cooldown_s", "skip_exists",
        "least_used_first", "sample_random_at", "save_locally", "in_flight",
    }
    for key, value in kwargs.items():
        if value is None:
            continue
        if key in gen_keys:
            raw.setdefault("generation", {})[key] = value
        elif key in nested_keys:
            section, sub = nested_keys[key]
            raw.setdefault("generation", {}).setdefault(section, {})[sub] = value
        elif key in {"hires", "adetailer"} and isinstance(value, dict):
            dest = raw.setdefault("generation", {}).setdefault(key, {})
            dest.update(value)
        elif key in batch_keys:
            raw.setdefault("batch", {})[key] = value
        elif key in video_keys or key.startswith("video_"):
            field = {
                "video_workflow": "workflow",
                "video_unet": "unet",
                "video_clip": "clip",
                "video_vae": "vae",
                "video_audio_vae": "audio_vae",
                "video_recipe": "recipe",
                "video_acc_file": "acc_file",
                "length": "frames",
            }.get(key, key)
            raw.setdefault("video", {})[field] = value
        elif key == "workflow_file":
            raw.setdefault("workflow", {})["file"] = value
        elif key == "workflow_map":
            raw.setdefault("workflow", {})["map"] = value
        elif key in {"host", "port", "timeout_s", "poll_interval_s"}:
            raw.setdefault("comfyui", {})[key] = value
        elif key in {"webui_root", "comfyui_root", "combinator_root"}:
            raw.setdefault("paths", {})[key] = str(value)
    return from_dict(raw, source=config.source, root=config.paths.root)
