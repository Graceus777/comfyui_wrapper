"""Qwen Image 2.1 text-to-image and reference-image edits.

The bundled graph matches the Q6 low-VRAM workflow that already runs in the
portable ComfyUI install: UnetLoaderGGUF ``qwen_image_2.1-Q6_K.gguf``,
CLIPLoader ``qwen3vl_8b_w4a8.safetensors`` (type ``qwen_image``), and
``qwen_image_2.1_vae_bf16.safetensors``. Sampling is Euler / simple / CFG 1.

Edits attach reference images to ``TextEncodeQwenImage21`` (``images.image_N``
plus the VAE). The encoder's latent output is the first reference's size;
sampling any other size shifts the edit, so that is opt-in.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from comfyui_wrapper.client import ComfyClient
from comfyui_wrapper.config import WrapperConfig, load_config
from comfyui_wrapper.generate import GenerateResult, _save_locally
from comfyui_wrapper.history import append_history, compute_generation_hash
from comfyui_wrapper.models import ensure_available, list_text_encoders, list_unets, list_vaes
from comfyui_wrapper.workflow import (
    WorkflowError,
    dump_workflow,
    load_workflow,
    nodes_of_type,
    normalize_sampler,
)

MAX_REFERENCES = 10
CACHE_DEVICES = ("auto", "gpu", "cpu", "off")
CACHE_DTYPES = ("default", "int8", "int4")
_ENCODE = "TextEncodeQwenImage21"
_CACHE_ID = "qwen_cache"
_REPEAT_ID = "qwen_repeat"


def align_qwen_size(value: int | float) -> int:
    """Qwen Image 2.1 latents are a 16px grid."""
    snapped = int(value) // 16 * 16
    return max(16, snapped)


def align_qwen_resolution(value: int | float) -> int:
    """Reference resize target. 0 keeps each image, rounded to 32."""
    raw = int(value)
    if raw <= 0:
        return 0
    snapped = raw // 32 * 32
    return min(4096, max(32, snapped))


def normalize_qwen_mode(mode: str | None, images: Iterable[Any] | None = None) -> str:
    key = str(mode or "").strip().lower().replace("-", "").replace("_", "")
    if key in {"edit", "imageedit", "i2i"}:
        return "edit"
    if key in {"txt2img", "t2i", "text", "texttoimage"}:
        return "txt2img"
    return "edit" if _clean_keys(images) else "txt2img"


def collect_reference_paths(primary: Any = None, extras: Any = None) -> list[str]:
    """Flatten a Gradio image plus a multi-file picker into path strings."""
    paths: list[str] = []

    def add(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (str, Path)):
            text = str(item).strip()
            if text:
                paths.append(text)
            return
        name = getattr(item, "name", None)
        if name:
            text = str(name).strip()
            if text:
                paths.append(text)

    add(primary)
    if isinstance(extras, (str, Path)):
        add(extras)
    elif isinstance(extras, (list, tuple)):
        for item in extras:
            add(item)
    elif extras is not None:
        add(extras)
    return paths


def resolve_qwen_asset(choices: Iterable[str], preferred: str, *hints: str) -> str:
    """Pick a listed file that matches a Qwen filename, otherwise keep the requested name."""
    values = [str(item) for item in choices if str(item)]
    by_lower = {value.lower(): value for value in values}
    by_name = {Path(value).name.lower(): value for value in values}
    for raw in (preferred, *hints):
        wanted = (raw or "").strip()
        if not wanted:
            continue
        if wanted in values:
            return wanted
        if wanted.lower() in by_lower:
            return by_lower[wanted.lower()]
        name = Path(wanted).name.lower()
        if name in by_name:
            return by_name[name]
        stem = Path(wanted).stem.lower()
        if len(stem) < 8:
            continue
        hits = [value for value in values if stem in Path(value).name.lower()]
        if len(hits) == 1:
            return hits[0]
        if hits:
            hits.sort(key=lambda value: (len(Path(value).name), len(value)))
            return hits[0]
    return preferred


def prepare_qwen_graph(
    workflow: dict,
    *,
    mode: str = "txt2img",
    prompt: str = "",
    negative: str = "",
    seed: int = 0,
    steps: int = 25,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    width: int = 1024,
    height: int = 1024,
    batch_size: int = 1,
    unet: str = "",
    clip: str = "",
    vae: str = "",
    resolution: int = 1024,
    image_keys: list[str] | None = None,
    match_reference: bool = True,
    cache_device: str = "auto",
    cache_dtype: str = "default",
    filename_prefix: str = "qwen_image_2_1",
) -> dict:
    """Return a patched copy. Text-to-image keeps the Q6 smoke graph."""
    from copy import deepcopy

    wf = deepcopy(workflow)
    mode_n = normalize_qwen_mode(mode, image_keys)
    keys = _clean_keys(image_keys) if mode_n == "edit" else []
    if mode_n == "edit" and not keys:
        raise ValueError("Qwen Image edit needs at least one reference image")
    if len(keys) > MAX_REFERENCES:
        raise ValueError(f"Qwen Image edit accepts at most {MAX_REFERENCES} reference images")
    steps_n = int(steps)
    if steps_n < 1:
        raise ValueError("Steps must be at least 1")
    batch_n = max(1, int(batch_size))
    width_n = align_qwen_size(width)
    height_n = align_qwen_size(height)
    resolution_n = align_qwen_resolution(resolution)
    device_n = str(cache_device or "auto").strip().lower()
    dtype_n = str(cache_dtype or "default").strip().lower()
    if device_n not in CACHE_DEVICES:
        raise ValueError(f"Cache device must be one of {', '.join(CACHE_DEVICES)}")
    if dtype_n not in CACHE_DTYPES:
        raise ValueError(f"Cache dtype must be one of {', '.join(CACHE_DTYPES)}")

    unet_id, unet_node = _require(wf, ["UnetLoaderGGUF", "UNETLoader"], "UNET loader")
    _set_unet(unet_node, unet)
    clip_id, clip_node = _require(wf, ["CLIPLoader"], "CLIP loader")
    clip_inputs = clip_node.setdefault("inputs", {})
    if clip:
        clip_inputs["clip_name"] = clip
    clip_inputs["type"] = "qwen_image"
    clip_inputs.setdefault("device", "default")
    _vae_id, vae_node = _require(wf, ["VAELoader"], "VAE loader")
    if vae:
        vae_node.setdefault("inputs", {})["vae_name"] = vae

    enc_id, enc_node = _require(wf, [_ENCODE], _ENCODE)
    enc_inputs = enc_node.setdefault("inputs", {})
    _clear_references(wf, enc_inputs)
    enc_inputs["clip"] = [clip_id, 0]
    enc_inputs["prompt"] = prompt or ""
    enc_inputs["negative_prompt"] = negative or ""
    enc_inputs["resolution"] = resolution_n

    _latent_id, latent_node = _require(wf, ["EmptyLatentImage"], "EmptyLatentImage")
    latent_inputs = latent_node.setdefault("inputs", {})
    latent_inputs["width"] = width_n
    latent_inputs["height"] = height_n
    latent_inputs["batch_size"] = batch_n

    _sampler_id, sampler_node = _require(wf, ["KSampler", "KSamplerAdvanced"], "KSampler")
    sampler_inputs = sampler_node.setdefault("inputs", {})
    sampler_inputs["seed"] = int(seed)
    sampler_inputs["steps"] = steps_n
    sampler_inputs["cfg"] = float(cfg)
    sampler_inputs["sampler_name"] = normalize_sampler(sampler or "euler")
    sampler_inputs["scheduler"] = scheduler or "simple"
    sampler_inputs["denoise"] = 1.0
    sampler_inputs["positive"] = [enc_id, 0]
    sampler_inputs["negative"] = [enc_id, 1]
    sampler_inputs["model"] = [unet_id, 0]
    sampler_inputs["latent_image"] = [_latent_id, 0]

    for _nid, node in nodes_of_type(wf, ["SaveImage", "SaveImageWebsocket"]):
        node.setdefault("inputs", {})["filename_prefix"] = filename_prefix or "qwen_image_2_1"
        break

    if keys:
        for index, key in enumerate(keys, start=1):
            nid = f"qwen_ref_{index}"
            wf[nid] = {
                "class_type": "LoadImage",
                "_meta": {"title": f"Qwen reference {index}"},
                "inputs": {"image": key},
            }
            enc_inputs[f"images.image_{index}"] = [nid, 0]
        enc_inputs["vae"] = [_vae_id, 0]
        if match_reference:
            latent_link: list[Any] = [enc_id, 2]
            if batch_n > 1:
                wf[_REPEAT_ID] = {
                    "class_type": "RepeatLatentBatch",
                    "_meta": {"title": "Qwen edit batch"},
                    "inputs": {"samples": [enc_id, 2], "amount": batch_n},
                }
                latent_link = [_REPEAT_ID, 0]
            sampler_inputs["latent_image"] = latent_link

    # Official edit graphs always insert the cache node. Text-to-image keeps
    # the smoke graph unless the user moves the cache off auto/default.
    if keys or device_n != "auto" or dtype_n != "default":
        model_link = sampler_inputs.get("model")
        wf[_CACHE_ID] = {
            "class_type": "QwenImage21Cache",
            "_meta": {"title": "Qwen Image 2.1 Cache"},
            "inputs": {
                "model": model_link,
                "device": device_n,
                "dtype": dtype_n,
            },
        }
        sampler_inputs["model"] = [_CACHE_ID, 0]
    return wf


def generate_qwen(
    prompt: str | None = None,
    *,
    negative: str | None = None,
    images: list[str | Path] | None = None,
    mode: str | None = None,
    seed: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    batch_size: int | None = None,
    unet: str | None = None,
    clip: str | None = None,
    vae: str | None = None,
    resolution: int | None = None,
    match_reference: bool | None = None,
    cache_device: str | None = None,
    cache_dtype: str | None = None,
    filename_prefix: str | None = None,
    cfg: WrapperConfig | None = None,
    config: WrapperConfig | str | Path | None = None,
    client: ComfyClient | None = None,
    wait: bool = True,
    dump_path: str | Path | None = None,
    record_history: bool = True,
    on_progress: Any = None,
) -> GenerateResult:
    """Queue one Qwen Image 2.1 txt2img or reference edit."""
    if cfg is None:
        if isinstance(config, WrapperConfig):
            cfg = config
        else:
            cfg = load_config(config)
    settings = cfg.qwen
    raw_images = list(images or [])
    mode_n = normalize_qwen_mode(mode, raw_images)
    if mode_n == "edit" and not _clean_keys(raw_images):
        raise ValueError("Qwen Image edit needs at least one reference image")
    if mode_n == "edit" and len(_clean_keys(raw_images)) > MAX_REFERENCES:
        raise ValueError(f"Qwen Image edit accepts at most {MAX_REFERENCES} reference images")

    used_seed = settings.seed if seed is None else seed
    if used_seed is None or int(used_seed) < 0:
        used_seed = random.randint(0, 2**32 - 1)
    used_seed = int(used_seed)
    positive = settings.prompt if prompt is None else prompt
    neg = settings.negative if negative is None else negative
    prefix = filename_prefix or settings.filename_prefix or "qwen_image_2_1"

    client = client or ComfyClient(cfg=cfg)
    live = client if client.ping() else None
    unet_n = resolve_qwen_asset(
        list_unets(cfg, live),
        unet or settings.unet,
        "qwen_image_2.1",
    )
    clip_n = resolve_qwen_asset(
        list_text_encoders(cfg, live),
        clip or settings.clip,
        "qwen3vl_8b",
    )
    vae_n = resolve_qwen_asset(
        list_vaes(cfg, live),
        vae or settings.vae,
        "qwen_image_2.1_vae",
    )
    if live is not None:
        unet_n = ensure_available("diffusion_models", unet_n, cfg, live) or unet_n
        clip_n = ensure_available("text_encoders", clip_n, cfg, live) or clip_n
        vae_n = ensure_available("vae", vae_n, cfg, live) or vae_n

    image_keys: list[str] = []
    if mode_n == "edit":
        image_keys = [_as_input_key(client, cfg, item) for item in _clean_keys(raw_images)]

    wf = prepare_qwen_graph(
        load_workflow(cfg.qwen_workflow_path()),
        mode=mode_n,
        prompt=positive or "",
        negative=neg or "",
        seed=used_seed,
        steps=int(steps if steps is not None else settings.steps),
        cfg=float(cfg_scale if cfg_scale is not None else settings.cfg),
        sampler=sampler or settings.sampler,
        scheduler=scheduler or settings.scheduler,
        width=int(width if width is not None else settings.width),
        height=int(height if height is not None else settings.height),
        batch_size=int(batch_size if batch_size is not None else settings.batch_size),
        unet=unet_n,
        clip=clip_n,
        vae=vae_n,
        resolution=int(resolution if resolution is not None else settings.resolution),
        image_keys=image_keys,
        match_reference=settings.match_reference if match_reference is None else bool(match_reference),
        cache_device=cache_device or settings.cache_device,
        cache_dtype=cache_dtype or settings.cache_dtype,
        filename_prefix=prefix,
    )
    if dump_path:
        dump_workflow(wf, dump_path)

    gen_hash = compute_generation_hash(
        positive or "",
        neg or "",
        steps=int(steps if steps is not None else settings.steps),
        sampler=sampler or settings.sampler,
        cfg=float(cfg_scale if cfg_scale is not None else settings.cfg),
        width=align_qwen_size(width if width is not None else settings.width),
        height=align_qwen_size(height if height is not None else settings.height),
        checkpoint=unet_n,
        extra={
            "kind": "qwen_image",
            "mode": mode_n,
            "references": image_keys,
            "resolution": int(resolution if resolution is not None else settings.resolution),
            "match_reference": settings.match_reference if match_reference is None else bool(match_reference),
        },
    )
    prompt_id = client.queue_prompt(wf, prompt_id=str(uuid.uuid4()))
    images_out: list[dict] = []
    saved: list[str] = []
    if wait:
        result = client.wait(prompt_id, on_progress=on_progress)
        images_out = result.images
        if cfg.batch.save_locally and images_out:
            saved = _save_locally(client, images_out, cfg.paths.output, prefix)
        if record_history:
            append_history(
                cfg.paths.history,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "prompt_id": prompt_id,
                    "kind": "qwen_image",
                    "mode": mode_n,
                    "gen_hash": gen_hash,
                    "seed": used_seed,
                    "prompt_preview": (positive or "")[:200],
                    "filename_prefix": prefix,
                    "references": len(image_keys),
                    "files": [Path(path).name for path in saved] or [img.get("filename") for img in images_out],
                },
            )
    return GenerateResult(
        prompt_id=prompt_id,
        workflow=wf,
        images=images_out,
        saved_files=saved,
        seed=used_seed,
        positive=positive or "",
        negative=neg or "",
        gen_hash=gen_hash,
    )


def _clean_keys(images: Iterable[Any] | None) -> list[str]:
    keys: list[str] = []
    for item in images or []:
        text = str(item or "").strip()
        if text:
            keys.append(text)
    return keys


def _require(workflow: dict, class_types: list[str], label: str) -> tuple[str, dict]:
    found = nodes_of_type(workflow, class_types)
    if not found:
        raise WorkflowError(f"Qwen Image workflow is missing {label}")
    return found[0]


def _set_unet(node: dict, name: str) -> None:
    if not name:
        return
    inputs = node.setdefault("inputs", {})
    inputs["unet_name"] = name
    if str(name).lower().endswith(".gguf"):
        node["class_type"] = "UnetLoaderGGUF"
        inputs.pop("weight_dtype", None)
    else:
        node["class_type"] = "UNETLoader"
        inputs["weight_dtype"] = inputs.get("weight_dtype") or "default"


def _clear_references(workflow: dict, enc_inputs: dict) -> None:
    for nid in list(workflow):
        if str(nid).startswith("qwen_ref_") or nid in {_CACHE_ID, _REPEAT_ID}:
            del workflow[nid]
    for key in list(enc_inputs):
        if key == "vae" or str(key).startswith("images."):
            enc_inputs.pop(key, None)


def _as_input_key(client: ComfyClient, cfg: WrapperConfig, value: str) -> str:
    candidate = Path(value)
    if not candidate.is_absolute():
        rooted = cfg.paths.root / candidate
        if rooted.is_file():
            candidate = rooted
    if candidate.is_file():
        return client.upload_image(candidate, "wrapper/qwen")
    if candidate.is_absolute():
        raise FileNotFoundError(f"Qwen reference image not found: {candidate}")
    return value
