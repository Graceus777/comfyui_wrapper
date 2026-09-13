"""Single-prompt generation against a running ComfyUI instance."""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from comfyui_wrapper.client import ComfyClient, JobResult
from comfyui_wrapper.config import WrapperConfig, apply_overrides, load_config
from comfyui_wrapper.history import append_history, compute_generation_hash
from comfyui_wrapper.lora import LoRARef, LoRAResolver, extract_lora_tags, refs_from_config_list
from comfyui_wrapper.models import ensure_available, share_webui_models
from comfyui_wrapper.workflow import (
    apply_settings,
    apply_vae_loader,
    dump_workflow,
    inject_adetailer,
    inject_hires,
    inject_loras,
    load_workflow,
    merge_bindings,
)


@dataclass
class GenerateResult:
    prompt_id: str
    workflow: dict
    images: list[dict] = field(default_factory=list)
    saved_files: list[str] = field(default_factory=list)
    seed: int = 0
    positive: str = ""
    negative: str = ""
    loras: list[dict] = field(default_factory=list)
    gen_hash: str = ""


def _normalize_loras(loras: Any, resolver: LoRAResolver, extra: list | None = None) -> list[LoRARef]:
    items = list(extra or [])
    if loras:
        if isinstance(loras, list):
            items.extend(loras)
        else:
            items.append(loras)
    return refs_from_config_list(items, resolver)


def merge_prompt_tags(prompt: str, refs: list[LoRARef], resolver: LoRAResolver) -> tuple[str, list[LoRARef]]:
    """Strip A1111 LoRA tags from the prompt and resolve them into LoRARef entries."""
    clean, tagged = extract_lora_tags(prompt)
    out = list(refs)
    have = {r.name.lower() for r in out}
    for name, weight in tagged:
        if name.lower() in have:
            continue
        out.append(resolver.resolve(name, weight))
        have.add(name.lower())
    return clean, out


def build_workflow(
    cfg: WrapperConfig,
    *,
    prompt: str | None = None,
    negative: str | None = None,
    seed: int | None = None,
    loras: list[LoRARef] | None = None,
    filename_prefix: str | None = None,
) -> tuple[dict, int, str, str]:
    gen = cfg.generation
    positive, _ = extract_lora_tags(gen.positive if prompt is None else prompt)
    neg = gen.negative if negative is None else negative
    used_seed = gen.seed if seed is None else seed
    if used_seed is None or int(used_seed) < 0:
        used_seed = random.randint(0, 2**32 - 1)
    used_seed = int(used_seed)

    wf = load_workflow(cfg.workflow_path())
    bindings = merge_bindings(wf, cfg.workflow_map)
    settings = {
        "positive": positive,
        "negative": neg,
        "seed": used_seed,
        "steps": gen.steps,
        "cfg": gen.cfg,
        "sampler": gen.sampler,
        "scheduler": gen.scheduler,
        "denoise": gen.denoise,
        "width": gen.width,
        "height": gen.height,
        "batch_size": gen.batch_size,
        "checkpoint": gen.checkpoint or None,
        "clip": gen.clip or None,
        "clip2": gen.clip2 or None,
        "vae": gen.vae or None,
        "filename_prefix": filename_prefix or gen.filename_prefix,
    }
    wf = apply_settings(wf, settings, bindings=bindings)
    if loras:
        wf = inject_loras(
            wf,
            [r.to_dict() for r in loras],
            mode=gen.lora_mode,
        )
    if gen.hires_enable:
        wf = inject_hires(
            wf,
            scale=gen.hires_scale,
            steps=gen.hires_steps,
            denoise=gen.hires_denoise,
            upscaler=gen.hires_upscaler,
            width=gen.width,
            height=gen.height,
        )
    if gen.adetailer_enable:
        wf = inject_adetailer(
            wf,
            model=gen.adetailer_model,
            denoise=gen.adetailer_denoise,
            steps=gen.adetailer_steps,
            cfg=gen.adetailer_cfg,
            guide_size=gen.adetailer_guide_size,
            seed=used_seed,
            sampler=gen.sampler,
            scheduler=gen.scheduler,
        )
    if gen.vae:
        wf = apply_vae_loader(wf, gen.vae)
    return wf, used_seed, positive, neg


def generate(
    prompt: str | None = None,
    *,
    negative: str | None = None,
    seed: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    checkpoint: str | None = None,
    loras: list | None = None,
    filename_prefix: str | None = None,
    config: WrapperConfig | str | Path | None = None,
    cfg: WrapperConfig | None = None,
    client: ComfyClient | None = None,
    resolver: LoRAResolver | None = None,
    wait: bool = True,
    dump_path: str | Path | None = None,
    record_history: bool = True,
    on_progress: Callable[[dict], None] | None = None,
    **overrides: Any,
) -> GenerateResult:
    """Queue one prompt. Settings come from config.yaml; kwargs override them."""
    if cfg is None:
        if isinstance(config, WrapperConfig):
            cfg = config
        else:
            cfg = load_config(config)
    if cfg_scale is not None:
        overrides["cfg"] = cfg_scale
    cfg = apply_overrides(
        cfg,
        steps=steps,
        width=width,
        height=height,
        sampler=sampler,
        scheduler=scheduler,
        checkpoint=checkpoint,
        seed=seed,
        filename_prefix=filename_prefix,
        **overrides,
    )

    client = client or ComfyClient(cfg=cfg)
    share_webui_models(cfg, client)
    if cfg.generation.checkpoint:
        ckpt = ensure_available("checkpoints", cfg.generation.checkpoint, cfg, client)
        unet = ensure_available("diffusion_models", cfg.generation.checkpoint, cfg, client)
        chosen = unet or ckpt
        if chosen and chosen != cfg.generation.checkpoint:
            cfg = apply_overrides(cfg, checkpoint=chosen)
    if cfg.generation.clip:
        clip = ensure_available("text_encoders", cfg.generation.clip, cfg, client)
        if clip != cfg.generation.clip:
            cfg = apply_overrides(cfg, clip=clip)
    if cfg.generation.clip2:
        clip2 = ensure_available("text_encoders", cfg.generation.clip2, cfg, client)
        if clip2 != cfg.generation.clip2:
            cfg = apply_overrides(cfg, clip2=clip2)
    if cfg.generation.vae:
        vae = ensure_available("vae", cfg.generation.vae, cfg, client)
        if vae != cfg.generation.vae:
            cfg = apply_overrides(cfg, vae=vae)
    resolver = resolver or LoRAResolver.from_client(client, cfg=cfg)
    refs = _normalize_loras(loras, resolver, extra=cfg.generation.loras)
    raw_positive = cfg.generation.positive if prompt is None else prompt
    raw_negative = cfg.generation.negative if negative is None else negative
    positive, refs = merge_prompt_tags(raw_positive, refs, resolver)

    workflow, used_seed, positive, neg = build_workflow(
        cfg,
        prompt=positive,
        negative=raw_negative,
        seed=seed,
        loras=refs,
        filename_prefix=filename_prefix,
    )
    if dump_path:
        dump_workflow(workflow, dump_path)

    gen_hash = compute_generation_hash(
        positive,
        neg,
        steps=cfg.generation.steps,
        sampler=cfg.generation.sampler,
        cfg=cfg.generation.cfg,
        width=cfg.generation.width,
        height=cfg.generation.height,
        checkpoint=cfg.generation.checkpoint,
        loras=[r.name for r in refs],
    )

    prompt_id = str(uuid.uuid4())
    prompt_id = client.queue_prompt(workflow, prompt_id=prompt_id)
    images: list[dict] = []
    saved: list[str] = []
    if wait:
        result: JobResult = client.wait(prompt_id, on_progress=on_progress)
        images = result.images
        if cfg.batch.save_locally and images:
            saved = _save_locally(client, images, cfg.paths.output, filename_prefix or cfg.generation.filename_prefix)
        if record_history:
            append_history(
                cfg.paths.history,
                {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "prompt_id": prompt_id,
                    "loras": [r.name for r in refs],
                    "gen_hash": gen_hash,
                    "seed": used_seed,
                    "prompt_preview": positive[:200],
                    "filename_prefix": filename_prefix or cfg.generation.filename_prefix,
                    "files": [Path(p).name for p in saved] or [img.get("filename") for img in images],
                },
            )

    return GenerateResult(
        prompt_id=prompt_id,
        workflow=workflow,
        images=images,
        saved_files=saved,
        seed=used_seed,
        positive=positive,
        negative=neg,
        loras=[r.to_dict() for r in refs],
        gen_hash=gen_hash,
    )


def _save_locally(
    client: ComfyClient,
    images: list[dict],
    output_dir: Path,
    prefix: str,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    for i, image in enumerate(images, start=1):
        filename = image.get("filename") or f"{prefix}_{i}.png"
        data = client.get_image_bytes(
            filename,
            subfolder=image.get("subfolder") or "",
            folder_type=image.get("type") or "output",
        )
        dest_name = filename
        dest = output_dir / dest_name
        if dest.exists():
            dest = output_dir / f"{Path(filename).stem}_{i}{Path(filename).suffix}"
        dest.write_bytes(data)
        saved.append(str(dest))
    return saved
