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
        filename_prefix=getattr(args, "prefix", None),
        workflow_file=getattr(args, "workflow", None),
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
    p.add_argument("--prefix", help="SaveImage filename_prefix")


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
    return 0


def cmd_video(args) -> int:
    from comfyui_wrapper.video import generate_video

    cfg = _cfg(args)
    try:
        result = generate_video(
            prompt=args.prompt,
            image=args.image,
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
            loop=False if getattr(args, "no_loop", False) else None,
            latent_upscale=True if getattr(args, "latent_upscale", False) else None,
            rtx_vsr=True if getattr(args, "rtx_vsr", False) else None,
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

    vid = sub.add_parser("video", help="MiniMax H3 image-to-video loop (sd-comic-ext ambient graph)")
    _add_global(vid)
    _add_gen(vid)
    vid.add_argument("--image", required=True, help="First-frame still")
    vid.add_argument("--prompt", "-p", help="Motion prompt")
    vid.add_argument("--frames", type=int, help="Frame count at 24 fps (124 ≈ 5s)")
    vid.add_argument("--fps", type=int)
    vid.add_argument("--audio-vae", help="MiniMax H3 audio VAE filename")
    vid.add_argument(
        "--recipe",
        choices=["native", "acc"],
        help="native = 12-step res_multistep (no Acc file); acc = 8-step PDD Acc (euler + Acc-8Step weights)",
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
    vid.add_argument("--no-wait", action="store_true")
    vid.add_argument("--dump-workflow", help="Write the patched API workflow JSON")
    vid.set_defaults(func=cmd_video)

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
