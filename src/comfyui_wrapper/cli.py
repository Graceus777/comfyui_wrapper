"""Command-line interface: generate, prep, preview, batch, loras, models."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

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
from comfyui_wrapper.workflow import load_workflow, merge_bindings


def _cfg(args) -> WrapperConfig:
    cfg = load_config(getattr(args, "config", None))
    control_image = getattr(args, "anima_control_image", None)
    control_model = getattr(args, "anima_lllite_model", None)
    anima_lllite = None
    if control_image is not None or control_model is not None:
        if not control_image or not control_model:
            raise SystemExit(
                "--anima-control-image and --anima-lllite-model must be supplied together"
            )
        anima_lllite = [{
            "image": control_image,
            "model_patch": control_model,
            "strength": getattr(args, "anima_control_strength", 1.0),
            "start_percent": getattr(args, "anima_control_start_percent", 0.0),
            "end_percent": getattr(args, "anima_control_end_percent", 1.0),
            "preprocess": getattr(args, "anima_preprocess", "none") or "none",
            "canny_low": getattr(args, "anima_canny_low", 0.4),
            "canny_high": getattr(args, "anima_canny_high", 0.8),
        }]
    inpaint_enable = getattr(args, "inpaint_enable", None)
    if getattr(args, "inpaint_mask", None) or getattr(args, "inpaint_image", None):
        inpaint_enable = True if inpaint_enable is None else inpaint_enable
    return apply_overrides(
        cfg,
        host=getattr(args, "host", None),
        port=getattr(args, "port", None),
        checkpoint=getattr(args, "checkpoint", None),
        steps=getattr(args, "steps", None),
        cfg_scale=getattr(args, "cfg_scale", None),
        sampler=getattr(args, "sampler", None),
        scheduler=getattr(args, "scheduler", None),
        width=getattr(args, "width", None),
        height=getattr(args, "height", None),
        seed=getattr(args, "seed", None),
        batch_size=getattr(args, "batch_size", None),
        denoise=getattr(args, "denoise", None),
        init_image=getattr(args, "init_image", None),
        filename_prefix=getattr(args, "prefix", None),
        workflow_file=getattr(args, "workflow", None),
        anima_lllite=anima_lllite,
        inpaint_enable=inpaint_enable,
        inpaint_image=getattr(args, "inpaint_image", None),
        inpaint_mask=getattr(args, "inpaint_mask", None),
        inpaint_denoise=getattr(args, "inpaint_denoise", None),
        inpaint_grow=getattr(args, "inpaint_grow", None),
        inpaint_blur=getattr(args, "inpaint_blur", None),
        inpaint_invert=getattr(args, "inpaint_invert", None),
        skip_exists=getattr(args, "skip_exists", None),
        random_count=getattr(args, "random_count", None),
        batch_count=getattr(args, "batch_count", None),
        cooldown_s=getattr(args, "cooldown", None),
        in_flight=getattr(args, "in_flight", None),
        save_locally=None if not getattr(args, "no_save", False) else False,
    )


def _add_global(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", help="Path to config.yaml")
    p.add_argument("--host", help="ComfyUI host")
    p.add_argument("--port", type=int, help="ComfyUI port")


def _add_gen(p: argparse.ArgumentParser) -> None:
    p.add_argument("--workflow", "-w", help="API-format workflow JSON (overrides config)")
    p.add_argument("--checkpoint", help="Checkpoint / UNET filename")
    p.add_argument("--steps", type=int)
    p.add_argument("--cfg-scale", dest="cfg_scale", type=float)
    p.add_argument("--sampler")
    p.add_argument("--scheduler")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--seed", type=int, help="Seed; -1 for random")
    p.add_argument("--batch-size", dest="batch_size", type=int)
    p.add_argument("--denoise", type=float, help="Denoise 0..1 (img2img strength; 1.0 = txt2img)")
    p.add_argument(
        "--init-image",
        dest="init_image",
        help="Init image for img2img (local path or ComfyUI input key)",
    )
    p.add_argument(
        "--inpaint-image",
        dest="inpaint_image",
        default=None,
        help="Init image for inpaint (defaults to --init-image when empty)",
    )
    p.add_argument(
        "--inpaint-mask",
        dest="inpaint_mask",
        default=None,
        help="White-on-black mask for inpaint (white = repaint; enables inpaint)",
    )
    p.add_argument("--inpaint-denoise", dest="inpaint_denoise", type=float, default=None)
    p.add_argument("--inpaint-grow", dest="inpaint_grow", type=int, default=None)
    p.add_argument("--inpaint-blur", dest="inpaint_blur", type=int, default=None)
    p.add_argument("--inpaint-invert", dest="inpaint_invert", action="store_true", default=None)
    p.add_argument("--no-inpaint", dest="inpaint_enable", action="store_false", default=None)
    p.add_argument("--prefix", help="SaveImage filename_prefix")
    p.add_argument("--anima-control-image", help="Preprocessed Anima control map")
    p.add_argument(
        "--anima-lllite-model",
        help="LLLite filename in ComfyUI models/model_patches",
    )
    p.add_argument("--anima-control-strength", type=float, default=1.0)
    p.add_argument("--anima-control-start-percent", type=float, default=0.0)
    p.add_argument("--anima-control-end-percent", type=float, default=1.0)
    p.add_argument(
        "--anima-preprocess",
        choices=["none", "canny"],
        default="none",
        help="A1111-style preprocessor for the control image (none = already preprocessed; canny = raw photo to edges in-graph)",
    )
    p.add_argument("--anima-canny-low", type=float, default=0.4)
    p.add_argument("--anima-canny-high", type=float, default=0.8)


def _add_lora_opt(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="NAME[:WEIGHT]",
        help="LoRA to apply (repeatable). Example: --lora AsheIL-21:0.8",
    )


def _parse_slots(values: list[str] | None) -> dict[str, list[str]]:
    slots: dict[str, list[str]] = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--slot must be name=a,b,c  (got {item!r})")
        key, raw = item.split("=", 1)
        slots[key.strip()] = [x.strip() for x in raw.split(",") if x.strip()]
    return slots


def cmd_generate(args) -> int:
    cfg = _cfg(args)
    loras = list(args.lora or [])
    try:
        result = generate(
            prompt=args.prompt,
            negative=args.negative,
            loras=loras or None,
            cfg=cfg,
            wait=not args.no_wait,
            dump_path=args.dump_workflow,
        )
    except (ComfyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"prompt_id: {result.prompt_id}")
    print(f"seed:      {result.seed}")
    print(f"hash:      {result.gen_hash}")
    if result.loras:
        print("loras:     " + ", ".join(f"{x['name']}:{x['weight']}" for x in result.loras))
    for path in result.saved_files:
        print(f"saved:     {path}")
    for image in result.images:
        print(f"output:    {image.get('filename')} ({image.get('subfolder') or 'root'})")
    if args.no_wait:
        print("queued (not waiting)")
    return 0


def _resolver(cfg: WrapperConfig, offline: bool) -> LoRAResolver:
    if offline:
        return LoRAResolver(cfg=cfg)
    client = ComfyClient(cfg=cfg)
    try:
        return LoRAResolver.from_client(client, cfg=cfg)
    except ComfyError as exc:
        print(f"warning: ComfyUI unreachable ({exc}); resolving LoRAs offline", file=sys.stderr)
        return LoRAResolver(cfg=cfg)


def _prep_from_args(args, cfg: WrapperConfig):
    extra_slots = _parse_slots(getattr(args, "slot", None))
    only = [x.strip() for x in (args.only or "").split(",") if x.strip()] if getattr(args, "only", None) else None
    exclude = [x.strip() for x in (args.exclude or "").split(",") if x.strip()] if getattr(args, "exclude", None) else None
    resolver = _resolver(cfg, offline=getattr(args, "offline", False))
    return prep_jobs(
        cfg,
        zone_name=args.zone,
        prompt_name=args.prompt_config,
        resolver=resolver,
        extra_slots=extra_slots,
        only=only,
        exclude=exclude,
        limit=getattr(args, "limit", None),
        offset=getattr(args, "offset", 0) or 0,
        seed=getattr(args, "seed", None),
    )


def cmd_preview(args) -> int:
    cfg = _cfg(args)
    try:
        jobs = _prep_from_args(args, cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(preview_jobs(jobs))
    return 0


def cmd_prep(args) -> int:
    cfg = _cfg(args)
    try:
        jobs = _prep_from_args(args, cfg)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    out = Path(args.out) if args.out else cfg.paths.jobs / f"jobs-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    save_jobs(
        jobs,
        out,
        extra={
            "zone": args.zone,
            "prompt": args.prompt_config,
            "workflow": cfg.workflow_file,
            "checkpoint": cfg.generation.checkpoint,
        },
    )
    pending = sum(1 for j in jobs if j.status == "pending")
    skipped = sum(1 for j in jobs if j.status == "skipped")
    print(f"wrote {out}  ({len(jobs)} jobs, {pending} pending, {skipped} skipped)")
    if args.show:
        print(preview_jobs(jobs))
    return 0


def cmd_batch(args) -> int:
    cfg = _cfg(args)
    jobs_path = Path(args.jobs) if args.jobs else None
    meta: dict = {}
    if jobs_path:
        try:
            jobs, meta = load_jobs(jobs_path)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if args.resume:
            for job in jobs:
                if job.status == "error":
                    job.status = "pending"
                    job.error = ""
    else:
        if not args.zone:
            print("error: pass --jobs FILE or --zone NAME", file=sys.stderr)
            return 1
        try:
            jobs = _prep_from_args(args, cfg)
        except (FileNotFoundError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        jobs_path = cfg.paths.jobs / f"jobs-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
        save_jobs(jobs, jobs_path, extra={"zone": args.zone, "prompt": args.prompt_config})
        print(f"prepared {jobs_path}")

    if args.dry_run:
        print(preview_jobs(jobs))
        return 0

    client = ComfyClient(cfg=cfg)
    if not client.ping():
        print(f"error: ComfyUI is not reachable at {cfg.comfyui.base_url}", file=sys.stderr)
        return 1
    resolver = LoRAResolver.from_client(client, cfg=cfg)
    stopped = {"flag": False}

    def handle_sigint(*_a):
        stopped["flag"] = True
        print("\ninterrupt requested — finishing current job then stopping", file=sys.stderr)
        try:
            client.interrupt()
        except Exception:
            pass

    try:
        import signal
        signal.signal(signal.SIGINT, handle_sigint)
    except Exception:
        pass

    def log(msg: str) -> None:
        print(msg, flush=True)

    run_jobs(
        jobs,
        cfg,
        client=client,
        resolver=resolver,
        on_log=log,
        resume=True,
        stop_flag=lambda: stopped["flag"],
    )
    save_jobs(jobs, jobs_path, extra=meta.get("meta") or meta)
    done = sum(1 for j in jobs if j.status == "done")
    err = sum(1 for j in jobs if j.status == "error")
    skip = sum(1 for j in jobs if j.status == "skipped")
    print(f"\n=== complete: {done} done, {err} error, {skip} skipped ===")
    return 1 if err else 0


def cmd_loras(args) -> int:
    cfg = _cfg(args)
    from comfyui_wrapper.models import list_loras, share_webui_models

    client = ComfyClient(cfg=cfg)
    share_webui_models(cfg, client if client.ping() else None)
    try:
        files = list_loras(cfg, client if client.ping() else None)
    except ComfyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    q = (args.search or "").lower()
    shown = 0
    for name in files:
        if q and q not in name.lower():
            continue
        print(name)
        shown += 1
    print(f"# {shown}/{len(files)} LoRAs", file=sys.stderr)
    return 0


def cmd_models(args) -> int:
    cfg = _cfg(args)
    from comfyui_wrapper.models import list_kind, share_webui_models

    client = ComfyClient(cfg=cfg)
    live = client if client.ping() else None
    share_webui_models(cfg, live)
    folder = args.folder or "checkpoints"
    try:
        files = list_kind(folder, cfg, live)
    except ComfyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name in files:
        print(name)
    print(f"# {len(files)} {folder}", file=sys.stderr)
    return 0


def cmd_status(args) -> int:
    cfg = _cfg(args)
    client = ComfyClient(cfg=cfg)
    try:
        stats = client.system_stats()
        queue = client.queue()
    except ComfyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    devices = (stats.get("devices") or [{}])
    if devices:
        d = devices[0]
        vram = d.get("vram_total")
        free = d.get("vram_free")
        print(f"device: {d.get('name', '?')}")
        if vram:
            print(f"vram:   {free}/{vram}")
    running = queue.get("queue_running") or []
    pending = queue.get("queue_pending") or []
    print(f"queue:  {len(running)} running, {len(pending)} pending")
    return 0


def cmd_interrupt(args) -> int:
    cfg = _cfg(args)
    try:
        ComfyClient(cfg=cfg).interrupt()
    except ComfyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("interrupt sent")
    return 0


def cmd_bindings(args) -> int:
    cfg = _cfg(args)
    path = Path(args.workflow) if args.workflow else cfg.workflow_path()
    try:
        wf = load_workflow(path)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    bindings = merge_bindings(wf, cfg.workflow_map)
    print(f"workflow: {path}")
    if not bindings:
        print("no bindings detected")
        return 1
    for key, (nid, field) in bindings.items():
        node = wf.get(nid) or {}
        current = (node.get("inputs") or {}).get(field)
        print(f"  {key:18} -> {nid}.{field}  ({node.get('class_type')})  = {current!r}")
    if "init_image" not in bindings:
        from comfyui_wrapper.workflow import nodes_of_type

        has_sampler = bool(nodes_of_type(wf, ["KSampler", "KSamplerAdvanced"]))
        has_latent = bool(
            nodes_of_type(wf, ["EmptyLatentImage", "EmptySD3LatentImage", "EmptyHunyuanLatentVideo"])
        )
        has_vae = bool(
            nodes_of_type(wf, ["VAELoader", "CheckpointLoaderSimple", "CheckpointLoader", "VAEDecode"])
        )
        if has_sampler and has_latent and has_vae:
            print("  init_image         -> img2img_loader.image  (LoadImage, injected when --init-image is set)")
            print("                       img2img_scale.width/height follow Width/Height; KSampler.denoise = strength")
            print("  inpaint_mask       -> inpaint_mask_loader.image  (LoadImage, injected when --inpaint-mask is set)")
            print("                       white = repaint via SetLatentNoiseMask; --inpaint-grow/--inpaint-blur feather it")
    return 0


def cmd_video(args) -> int:
    from comfyui_wrapper.video import generate_video

    cfg = _cfg(args)
    try:
        result = generate_video(
            prompt=args.prompt,
            image=args.image,
            last_frame=getattr(args, "last_frame", None),
            mode=getattr(args, "mode", None),
            ref_images=getattr(args, "ref_image", None),
            ref_videos=getattr(args, "ref_video", None),
            ref_audios=getattr(args, "ref_audio", None),
            ref_image_size=getattr(args, "ref_image_size", None),
            include_reference_video_audio=False if getattr(args, "no_ref_video_audio", False) else None,
            seed=getattr(args, "seed", None),
            steps=getattr(args, "steps", None),
            width=getattr(args, "width", None),
            height=getattr(args, "height", None),
            frames=getattr(args, "frames", None),
            fps=getattr(args, "fps", None),
            sampler=getattr(args, "sampler", None),
            scheduler=getattr(args, "scheduler", None),
            unet=getattr(args, "checkpoint", None),
            audio_vae=getattr(args, "audio_vae", None),
            recipe=getattr(args, "recipe", None),
            acc_file=getattr(args, "acc_file", None),
            size_megapixels=getattr(args, "size_megapixels", None),
            loop=False if getattr(args, "no_loop", False) else None,
            sam_target=getattr(args, "sam_target", None),
            sam3_checkpoint=getattr(args, "sam3_checkpoint", None),
            source_start=getattr(args, "source_start", None),
            source_end=getattr(args, "source_end", None),
            reference_description=getattr(args, "reference_description", None),
            latent_upscale=True if getattr(args, "latent_upscale", False) else None,
            rtx_vsr=True if getattr(args, "rtx_vsr", False) else None,
            save_latents=False if getattr(args, "no_save_latents", False) else (
                True if getattr(args, "save_latents", False) else None
            ),
            filename_prefix=getattr(args, "prefix", None),
            cfg=cfg,
            wait=not args.no_wait,
            dump_path=getattr(args, "dump_workflow", None),
        )
    except (ComfyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"prompt_id: {result.prompt_id}")
    print(f"seed:      {result.seed}")
    for path in result.saved_files:
        print(f"saved:     {path}")
    return 0


def cmd_upscale(args) -> int:
    from pathlib import Path as _Path

    from comfyui_wrapper.video import upscale_dropped_file
    from comfyui_wrapper.workflow import dump_workflow as _dump

    cfg = _cfg(args)
    client = ComfyClient(cfg=cfg)
    if not client.ping():
        print(f"error: ComfyUI is not reachable at {cfg.comfyui.base_url}", file=sys.stderr)
        return 1
    source = _Path(args.source)
    dest = _Path(args.out) if getattr(args, "out", None) else None
    if getattr(args, "dump_workflow", None) and getattr(args, "no_wait", False):
        # Dump the graph without queueing: build via the same helpers.
        from comfyui_wrapper.h3 import (
            detect_upscale_kind,
            latent_upscale_graph,
            resolve_dropped_latent_pair,
            rtx_vsr_graph,
        )
        from comfyui_wrapper.video import stage_latent_file, stage_input_video

        kind = args.mode if args.mode != "auto" else detect_upscale_kind(source)
        if kind == "latent":
            explicit_audio = getattr(args, "audio_latent", None)
            if explicit_audio:
                video_p, audio_p = source, _Path(explicit_audio)
            else:
                video_p, audio_p = resolve_dropped_latent_pair([source])
            if video_p is None:
                print(f"error: Latent not found: {source}", file=sys.stderr)
                return 1
            key = stage_latent_file(video_p, client=client, cfg=cfg)
            audio_key = (
                stage_latent_file(audio_p, client=client, cfg=cfg)
                if audio_p is not None and _Path(audio_p).is_file()
                else None
            )
            wf = latent_upscale_graph(
                key,
                (dest.stem if dest else source.stem + "__latent2x"),
                audio_latent_key=audio_key,
                vae=cfg.video.vae,
                audio_vae=cfg.video.audio_vae,
                model_name=cfg.video.latent_upscale_model,
                scale=cfg.video.latent_upscale_scale,
                fps=cfg.video.fps,
            )
        else:
            key = stage_input_video(source, client=client, cfg=cfg)
            wf = rtx_vsr_graph(
                key,
                (dest.stem if dest else source.stem + "__rtx"),
                scale=cfg.video.rtx_vsr_scale,
                quality=cfg.video.rtx_vsr_quality,
            )
        _dump(wf, args.dump_workflow)
        print(f"wrote {args.dump_workflow} ({kind})")
        return 0
    try:
        out, kind = upscale_dropped_file(
            source,
            dest,
            mode=getattr(args, "mode", "auto"),
            audio_source=getattr(args, "audio_latent", None),
            client=client,
            cfg=cfg,
        )
    except (ComfyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"kind:      {kind}")
    print(f"saved:     {out}")
    return 0


def cmd_qwen(args) -> int:
    from comfyui_wrapper.qwen_image import generate_qwen

    cfg = _cfg(args)
    images = list(getattr(args, "image", None) or [])
    try:
        result = generate_qwen(
            prompt=getattr(args, "prompt", None),
            negative=getattr(args, "negative", None),
            images=images or None,
            mode=getattr(args, "mode", None),
            seed=getattr(args, "seed", None),
            steps=getattr(args, "steps", None),
            cfg_scale=getattr(args, "cfg_scale", None),
            width=getattr(args, "width", None),
            height=getattr(args, "height", None),
            sampler=getattr(args, "sampler", None),
            scheduler=getattr(args, "scheduler", None),
            batch_size=getattr(args, "batch_size", None),
            unet=getattr(args, "unet", None),
            clip=getattr(args, "clip", None),
            vae=getattr(args, "vae", None),
            resolution=getattr(args, "resolution", None),
            match_reference=getattr(args, "match_reference", None),
            cache_device=getattr(args, "cache_device", None),
            cache_dtype=getattr(args, "cache_dtype", None),
            filename_prefix=getattr(args, "prefix", None),
            cfg=cfg,
            wait=not args.no_wait,
            dump_path=getattr(args, "dump_workflow", None),
        )
    except (ComfyError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"prompt_id: {result.prompt_id}")
    print(f"seed:      {result.seed}")
    for path in result.saved_files:
        print(f"saved:     {path}")
    return 0


def cmd_ui(args) -> int:
    from comfyui_wrapper.webui import launch_from_args
    return launch_from_args(args)


def cmd_tts(args) -> int:
    from comfyui_wrapper.tts import VoiceCloneError, clone_voice, installation_status

    cfg = _cfg(args)
    if args.check:
        status = installation_status(cfg)
        print(status)
        return 0 if status.startswith("Ready") else 1
    try:
        result = clone_voice(
            args.text, args.reference, cfg=cfg, reference_text=args.reference_text,
            mode="voice_only" if args.voice_only else "continuation",
            language=args.language, steps=args.steps, seed=args.seed,
            guidance_scale=args.guidance_scale,
            max_seconds=args.max_seconds, output=args.output,
            on_progress=lambda message: print(message, flush=True),
        )
    except (VoiceCloneError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Voice clone stopped.", file=sys.stderr)
        return 130
    print(f"saved: {result.audio}")
    print(f"details: {result.metadata}")
    print(f"seed: {result.seed}; duration: {result.duration:.2f}s; peak model VRAM: {result.peak_vram_gb:.2f} GB")
    return 0


def cmd_list_configs(args) -> int:
    cfg = _cfg(args)
    print("zones:")
    for name in list_named_json(cfg.paths.zone_dirs()):
        print(f"  {name}")
    print("prompts:")
    for name in list_named_json(cfg.paths.prompt_dirs()):
        print(f"  {name}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="comfywrap",
        description="ComfyUI API wrapper — single-prompt generate and combinator batch runs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Queue one prompt using config.yaml settings")
    _add_global(g)
    _add_gen(g)
    _add_lora_opt(g)
    g.add_argument("--prompt", "-p", help="Positive prompt (overrides config)")
    g.add_argument("--negative", "-n", help="Negative prompt")
    g.add_argument("--no-wait", action="store_true", help="Queue and return immediately")
    g.add_argument("--dump-workflow", help="Write the patched API workflow JSON to this path")
    g.set_defaults(func=cmd_generate)

    vid = sub.add_parser("video", help="MiniMax H3 frame animation, appearance references, or source-video swaps")
    _add_global(vid)
    vid.add_argument("--checkpoint", help="Compatible H3 checkpoint; the workflow chooses its model family")
    vid.add_argument("--steps", type=int, help="Base quality steps; acceleration and swap presets fix their own step count")
    vid.add_argument("--width", type=int)
    vid.add_argument("--height", type=int)
    vid.add_argument("--seed", type=int, help="Seed; -1 for random")
    vid.add_argument("--prefix", help="Output filename prefix")
    vid.add_argument("--image", required=False, default=None, help="Start frame for i2v/r2v_frames; primary appearance picture for r2v; replacement face/character for swaps")
    vid.add_argument("--last-frame", dest="last_frame", default=None, help="End-frame anchor for i2v or r2v_frames")
    vid.add_argument(
        "--mode",
        choices=["i2v", "r2v", "r2v_frames", "headswap", "charswap", "t2v"],
        default=None,
        help="i2v = FL2VA start/end frames; r2v = Ref2VA appearance/motion/audio references; r2v_frames = references plus separate start/end guides; swaps = one replacement and one source clip; t2v = text alone",
    )
    vid.add_argument("--sam-target", dest="sam_target", default=None, help="Character swap: who SAM3 should replace, for example \"the person\"")
    vid.add_argument(
        "--sam3-checkpoint",
        dest="sam3_checkpoint",
        default=None,
        help="Character swap: SAM3.1 checkpoint filename in models/checkpoints",
    )
    vid.add_argument("--ref-image", dest="ref_image", action="append", default=[], help="Reference still (<Picture i>, repeatable, max 9)")
    vid.add_argument("--ref-video", dest="ref_video", action="append", default=[], help="Reference clip, mp4 (<Video k>, repeatable, max 3)")
    vid.add_argument("--ref-audio", dest="ref_audio", action="append", default=[], help="Reference audio clip (<Audio j>, repeatable, max 3)")
    vid.add_argument("--no-ref-video-audio", action="store_true", help="Use reference video visuals without their soundtrack")
    vid.add_argument("--source-start", type=float, default=None, help="Swap source trim start in seconds")
    vid.add_argument("--source-end", type=float, default=None, help="Swap source trim end in seconds; 0 uses the whole clip")
    vid.add_argument("--reference-description", default=None, help="Character appearance, separate from the scene and motion prompt")
    vid.add_argument("--ref-image-size", dest="ref_image_size", choices=["match", "max"], default=None)
    vid.add_argument("--prompt", "-p", help="Motion prompt")
    vid.add_argument("--frames", type=int, help="Frame count at 24 fps; swap modes interpret this as trim duration unless --source-end is set")
    vid.add_argument("--fps", type=int, choices=[24], help="H3's fixed generation frame rate")
    vid.add_argument("--size-megapixels", type=float, help="Generation resolution when matching input aspect ratio")
    vid.add_argument("--audio-vae", help="MiniMax H3 audio VAE filename")
    vid.add_argument(
        "--recipe",
        choices=["native", "acc", "acc4", "ref_native", "ref_turbo4", "ref_turbo8"],
        help="native/ref_native = adjustable quality steps; Acc and reference Turbo presets use fixed steps and matching weights; swaps select their preset automatically",
    )
    vid.add_argument("--acc-file", dest="acc_file", help="PDD Acc file in models/pdd_acc (not a LoRA)")
    vid.add_argument("--no-loop", action="store_true", help="Skip cyclic crossfade")
    vid.add_argument(
        "--latent-upscale",
        action="store_true",
        help="Decode-only 2x via MinimaxH3LatentUpscaler3D (splits AV latent first)",
    )
    vid.add_argument(
        "--rtx-vsr",
        action="store_true",
        help="Post-process 2x via NVIDIA RTXVideoSuperResolution",
    )
    vid.add_argument(
        "--save-latents",
        dest="save_latents",
        action="store_true",
        default=None,
        help="Save the sampled AV latent (.latent) for decode-only 2x in the Upscale tab",
    )
    vid.add_argument(
        "--no-save-latents",
        dest="no_save_latents",
        action="store_true",
        help="Do not save AV latents",
    )
    vid.add_argument("--no-wait", action="store_true")
    vid.add_argument("--dump-workflow", help="Write the patched API workflow JSON")
    vid.set_defaults(func=cmd_video)

    up = sub.add_parser("upscale", help="Upscale a saved .latent (3D 2x) or rendered clip (RTX VSR)")
    _add_global(up)
    up.add_argument("source", help="Local .latent/.safetensors (3D 2x) or .mp4/.webm/.mov/.mkv (RTX VSR)")
    up.add_argument("--out", "-o", default=None, help="Output mp4 path (default: <stem>__latent2x/__rtx.mp4)")
    up.add_argument(
        "--mode",
        choices=["auto", "latent", "rtx"],
        default="auto",
        help="auto picks from the file suffix (single check); latent = 3D 2x decode-only; rtx = RTX VSR post pass",
    )
    up.add_argument("--no-wait", action="store_true")
    up.add_argument("--dump-workflow", help="Write the upscale API workflow JSON")
    up.add_argument(
        "--audio-latent",
        default=None,
        help="Explicit *_audio.latent half (otherwise the *_audio sibling next to <source> is used)",
    )
    up.set_defaults(func=cmd_upscale)

    def _add_batch_sel(p):
        _add_global(p)
        _add_gen(p)
        p.add_argument("--zone", "-z", help="Zone config name (combinator JSON stem)")
        p.add_argument("--prompt-config", dest="prompt_config", help="Prompt config name")
        p.add_argument("--slot", action="append", default=[], help="Override/add a slot: name=a,b,c")
        p.add_argument("--only", help="Comma-separated loop LoRA names to include")
        p.add_argument("--exclude", help="Comma-separated loop LoRA names to skip")
        p.add_argument("--limit", type=int)
        p.add_argument("--offset", type=int, default=0)
        p.add_argument("--random-count", dest="random_count", type=int)
        p.add_argument("--batch-count", dest="batch_count", type=int)
        p.add_argument("--skip-exists", dest="skip_exists", action="store_const", const=True, default=None)
        p.add_argument("--no-skip-exists", dest="skip_exists", action="store_const", const=False, default=None)
        p.add_argument("--offline", action="store_true", help="Do not query ComfyUI for LoRA filenames")

    prev = sub.add_parser("preview", help="Print the expanded job list without writing or running")
    _add_batch_sel(prev)
    prev.set_defaults(func=cmd_preview)

    prep = sub.add_parser("prep", help="Expand zones/slots into an editable job file")
    _add_batch_sel(prep)
    prep.add_argument("--out", "-o", help="Job JSON output path")
    prep.add_argument("--show", action="store_true", help="Also print the preview")
    prep.set_defaults(func=cmd_prep)

    batch = sub.add_parser("batch", help="Run a job file, or prep+run from --zone")
    _add_batch_sel(batch)
    batch.add_argument("--jobs", "-j", help="Job JSON produced by prep")
    batch.add_argument("--dry-run", action="store_true")
    batch.add_argument("--resume", action="store_true", help="Retry jobs with status=error")
    batch.add_argument("--cooldown", type=float, help="Seconds between jobs")
    batch.add_argument("--no-save", action="store_true", help="Do not copy images into paths.output")
    batch.set_defaults(func=cmd_batch)

    loras = sub.add_parser("loras", help="List LoRAs ComfyUI can see")
    _add_global(loras)
    loras.add_argument("--search", "-s")
    loras.set_defaults(func=cmd_loras)

    models = sub.add_parser("models", help="List files in a ComfyUI model folder")
    _add_global(models)
    models.add_argument("--folder", default="checkpoints", help="checkpoints, loras, vae, ...")
    models.set_defaults(func=cmd_models)

    st = sub.add_parser("status", help="ComfyUI device + queue")
    _add_global(st)
    st.set_defaults(func=cmd_status)

    intr = sub.add_parser("interrupt", help="Ask ComfyUI to stop the current job")
    _add_global(intr)
    intr.set_defaults(func=cmd_interrupt)

    bind = sub.add_parser("bindings", help="Show auto-detected workflow node bindings")
    _add_global(bind)
    bind.add_argument("--workflow", "-w")
    bind.set_defaults(func=cmd_bindings)

    ls = sub.add_parser("configs", help="List zone and prompt config names")
    _add_global(ls)
    ls.set_defaults(func=cmd_list_configs)

    qw = sub.add_parser("qwen", help="Qwen Image 2.1 text-to-image or reference-image edit")
    _add_global(qw)
    qw.add_argument("--prompt", "-p", help="Prompt (edit instruction when --image is set)")
    qw.add_argument("--negative", "-n", help="Negative prompt")
    qw.add_argument(
        "--image",
        action="append",
        default=[],
        help="Reference image for an edit (repeatable, max 10). Implies edit mode.",
    )
    qw.add_argument("--mode", choices=["txt2img", "edit"], default=None)
    qw.add_argument("--steps", type=int)
    qw.add_argument("--cfg-scale", dest="cfg_scale", type=float)
    qw.add_argument("--sampler")
    qw.add_argument("--scheduler")
    qw.add_argument("--width", type=int)
    qw.add_argument("--height", type=int)
    qw.add_argument("--seed", type=int, help="Seed; -1 for random")
    qw.add_argument("--batch-size", dest="batch_size", type=int)
    qw.add_argument("--unet", help="Qwen Image 2.1 UNET or GGUF filename")
    qw.add_argument("--clip", help="Qwen3-VL text encoder filename")
    qw.add_argument("--vae", help="Qwen Image 2.1 VAE filename")
    qw.add_argument("--resolution", type=int, help="Reference resize. 0 keeps each image size.")
    qw.add_argument("--match-reference", dest="match_reference", action="store_const", const=True, default=None)
    qw.add_argument("--no-match-reference", dest="match_reference", action="store_const", const=False, default=None)
    qw.add_argument("--cache-device", dest="cache_device", choices=["auto", "gpu", "cpu", "off"])
    qw.add_argument("--cache-dtype", dest="cache_dtype", choices=["default", "int8", "int4"])
    qw.add_argument("--prefix", help="SaveImage filename_prefix")
    qw.add_argument("--no-wait", action="store_true")
    qw.add_argument("--dump-workflow", help="Write the patched API workflow JSON")
    qw.set_defaults(func=cmd_qwen)

    tts = sub.add_parser("tts", help="Clone a voice locally with dots.tts-SOAR")
    _add_global(tts)
    tts.add_argument("--check", action="store_true", help="Check the local runtime and model files")
    tts.add_argument("--text", help="New words to speak")
    tts.add_argument("--reference", help="Short reference recording (3–10 seconds recommended)")
    tts.add_argument("--reference-text", default="", help="Exact words spoken in the reference")
    tts.add_argument("--voice-only", action="store_true", help="Clone timbre without a reference transcript")
    tts.add_argument("--language", default="EN", help="Language code or auto_detect")
    tts.add_argument("--steps", type=int, default=10, help="Flow sampling steps / NFE (10–32; default: 10)")
    tts.add_argument("--guidance-scale", type=float, default=1.2, help="CFG strength (0–3; default: 1.2)")
    tts.add_argument("--seed", type=int, default=-1)
    tts.add_argument("--max-seconds", type=float, default=25)
    tts.add_argument("--output", type=Path, help="New WAV output path (existing files are preserved)")
    tts.set_defaults(func=cmd_tts)

    ui = sub.add_parser("ui", help="Launch the A1111-style Gradio WebUI")
    ui.add_argument("--config", "-c", help="Path to config.yaml")
    ui.add_argument("--host", help="ComfyUI host (override config)")
    ui.add_argument("--comfy-port", type=int, dest="comfy_port", help="ComfyUI port")
    ui.add_argument("--listen", action="store_true", help="Bind WebUI to 0.0.0.0")
    ui.add_argument("--server-name", default=None, help="WebUI bind address")
    ui.add_argument("--port", "-p", type=int, default=7860, help="WebUI port (default 7860)")
    ui.add_argument("--share", action="store_true", help="Gradio share tunnel")
    ui.add_argument("--no-browser", action="store_true", help="Do not open a browser")
    ui.set_defaults(func=cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
