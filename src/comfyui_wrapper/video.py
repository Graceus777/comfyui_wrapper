"""MiniMax H3 image-to-video loops (same graph as sd-comic-ext ambient loops)."""

from __future__ import annotations

import random
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable

from comfyui_wrapper.client import ComfyClient, ComfyError
from comfyui_wrapper.config import WrapperConfig, load_config
from comfyui_wrapper.generate import GenerateResult, _save_locally
from comfyui_wrapper.h3 import (
    apply_h3_graph,
    apply_h3_latent_upscale,
    coerce_video_stack,
    normalize_ref_image_size,
    recipe_settings,
    rtx_vsr_graph,
    set_last_frame,
    set_r2v_refs,
)
from comfyui_wrapper.history import append_history, compute_generation_hash
from comfyui_wrapper.models import infer_comfyui_root, list_pdd_acc, list_text_encoders, list_unets, list_vaes
from comfyui_wrapper.workflow import apply_settings, dump_workflow, load_workflow, merge_bindings

# MiniMax H3 sizes are a 32px grid. "Size from image" keeps the still's aspect
# and fits ~0.52 MP (H3 Studio "balanced" 960×544). Native max is 1344×768.
H3_ALIGN = 32
H3_SIZE_MP = 0.52
H3_MAX_MP = 1.03
H3_MIN_MP = 0.15


def align_h3(value: int | float) -> int:
    return max(H3_ALIGN, int(round(float(value) / H3_ALIGN) * H3_ALIGN))


def h3_size(
    width: int | float,
    height: int | float,
    *,
    megapixels: float | None = None,
    fit: bool = True,
) -> tuple[int, int]:
    """Return an H3-legal (width, height).

    ``fit=True`` (Size from image) preserves aspect and fills ``megapixels``.
    ``fit=False`` only snaps the given canvas onto the 32px grid.
    """
    src_w = max(1, int(width))
    src_h = max(1, int(height))
    if not fit:
        return align_h3(src_w), align_h3(src_h)
    ratio = src_w / src_h
    mp = H3_SIZE_MP if megapixels is None else float(megapixels)
    if not (mp > 0):
        mp = H3_SIZE_MP
    mp = min(H3_MAX_MP, max(H3_MIN_MP, mp))
    target = mp * 1_000_000
    out_w = align_h3((target * ratio) ** 0.5)
    out_h = align_h3((target / ratio) ** 0.5)
    area = out_w * out_h
    cap = H3_MAX_MP * 1_000_000
    if area > cap:
        scale = (cap / area) ** 0.5
        out_w = max(H3_ALIGN, int(out_w * scale) // H3_ALIGN * H3_ALIGN)
        out_h = max(H3_ALIGN, int(out_h * scale) // H3_ALIGN * H3_ALIGN)
    return out_w, out_h


def target_dimensions(
    source: str | Path,
    *,
    megapixels: float | None = None,
) -> tuple[int, int]:
    """Derive H3 ImageScale + latent size from the still's aspect ratio."""
    try:
        from PIL import Image
    except ImportError:
        return h3_size(832, 480, megapixels=megapixels)
    with Image.open(source) as image:
        return h3_size(image.width, image.height, megapixels=megapixels)


def ffmpeg_bin() -> str | None:
    return shutil.which("ffmpeg")


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise ComfyError(result.stderr.strip() or result.stdout.strip() or "ffmpeg failed")


def cyclic_crossfade(
    raw_path: str | Path,
    final_path: str | Path,
    *,
    fps: int = 24,
    crossfade_frames: int = 12,
) -> Path:
    """Close a loop: middle = frames[N:-N], seam = fade(last N → first N)."""
    raw_path = Path(raw_path)
    final_path = Path(final_path)
    ff = ffmpeg_bin()
    if not ff:
        raise ComfyError("ffmpeg is not on PATH; cannot close the loop")
    cf = max(1, int(crossfade_frames))
    fade_s = cf / float(fps)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="h3loop_") as tmp:
        tdir = Path(tmp)
        head = tdir / "head.mp4"
        tail = tdir / "tail.mp4"
        mid = tdir / "mid.mp4"
        seam = tdir / "seam.mp4"
        concat = tdir / "list.txt"
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(raw_path),
            "-vf", f"select='lte(n\\,{cf - 1})',setpts=PTS-STARTPTS",
            "-an", str(head),
        ])
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-sseof", f"-{fade_s:.4f}",
            "-i", str(raw_path),
            "-an", str(tail),
        ])
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(raw_path),
            "-vf", f"select='gte(n\\,{cf})',setpts=PTS-STARTPTS",
            "-an", str(mid),
        ])
        # Drop the last N frames of `mid` (they become the fade-from side).
        mid_cut = tdir / "mid_cut.mp4"
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(mid),
            "-vf", f"reverse,select='gte(n\\,{cf})',reverse,setpts=PTS-STARTPTS",
            "-an", str(mid_cut),
        ])
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tail), "-i", str(head),
            "-filter_complex", f"[0][1]xfade=transition=fade:duration={fade_s:.4f}:offset=0[v]",
            "-map", "[v]", "-an", str(seam),
        ])
        concat.write_text(
            f"file '{mid_cut.resolve().as_posix()}'\nfile '{seam.resolve().as_posix()}'\n",
            encoding="utf-8",
        )
        silent = tdir / "silent.mp4"
        _run([
            ff, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "concat", "-safe", "0", "-i", str(concat),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-an",
            str(silent),
        ])
        # Keep native H3 audio from the raw mux, trimmed to the looped picture.
        mux = subprocess.run(
            [
                ff, "-y", "-hide_banner", "-loglevel", "error",
                "-i", str(silent), "-i", str(raw_path),
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-shortest",
                "-movflags", "+faststart",
                str(final_path),
            ],
            capture_output=True,
            text=True,
        )
        if mux.returncode:
            silent.replace(final_path)
    return final_path


def stage_input_video(
    source: str | Path,
    *,
    client: ComfyClient,
    cfg: WrapperConfig,
    subfolder: str = "wrapper/rtx",
) -> str:
    """Put a local mp4 where LoadVideo can see it; return the Comfy input key."""
    source = Path(source)
    try:
        return client.upload_image(source, subfolder)
    except ComfyError:
        root = infer_comfyui_root(cfg)
        if root is None:
            raise
        dest_dir = root / "input" / subfolder
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / source.name
        shutil.copy2(source, dest)
        return f"{subfolder}/{dest.name}".replace("\\", "/")


def apply_rtx_vsr(
    source: str | Path,
    destination: str | Path,
    *,
    client: ComfyClient,
    cfg: WrapperConfig,
    scale: float = 2.0,
    quality: str = "ULTRA",
    filename_prefix: str = "h3_loop_rtx",
    on_progress: Callable[[dict], None] | None = None,
) -> Path:
    """Queue RTXVideoSuperResolution on a saved clip and copy the result locally."""
    source = Path(source)
    destination = Path(destination)
    key = stage_input_video(source, client=client, cfg=cfg)
    wf = rtx_vsr_graph(key, filename_prefix, scale=scale, quality=quality)
    prompt_id = client.queue_prompt(wf)
    result = client.wait(prompt_id, on_progress=on_progress)
    saved = _save_locally(client, result.images, destination.parent, Path(filename_prefix).name)
    if not saved:
        raise ComfyError("RTX VSR completed without a video output")
    produced = Path(saved[0])
    if produced.resolve() != destination.resolve():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(produced, destination)
        if produced.parent == destination.parent and produced.name != destination.name:
            try:
                produced.unlink()
            except OSError:
                pass
    return destination


def generate_video(
    prompt: str | None = None,
    *,
    image: str | Path | None = None,
    last_frame: str | Path | None = None,
    mode: str | None = None,
    ref_images: list[str | Path] | None = None,
    ref_videos: list[str | Path] | None = None,
    ref_audios: list[str | Path] | None = None,
    ref_image_size: str | None = None,
    seed: int | None = None,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    frames: int | None = None,
    fps: int | None = None,
    sampler: str | None = None,
    scheduler: str | None = None,
    unet: str | None = None,
    clip: str | None = None,
    vae: str | None = None,
    audio_vae: str | None = None,
    recipe: str | None = None,
    acc_file: str | None = None,
    loop: bool | None = None,
    crossfade_frames: int | None = None,
    latent_upscale: bool | None = None,
    rtx_vsr: bool | None = None,
    filename_prefix: str | None = None,
    cfg: WrapperConfig | None = None,
    config: WrapperConfig | str | Path | None = None,
    client: ComfyClient | None = None,
    wait: bool = True,
    dump_path: str | Path | None = None,
    on_progress: Callable[[dict], None] | None = None,
) -> GenerateResult:
    """Queue a MiniMax H3 video job: I2V/FLF2V or R2V-lite (FL2VA, no new model).

    R2V-lite reuses the FL2VA UNET with the native
    ``MiniMaxH3ReferenceToVideo`` node: ``<Picture i>`` / ``<Video k>`` /
    ``<Audio j>`` tags steer identity, motion, and voice. Reference
    adherence is weaker than a true Ref2VA checkpoint, but costs zero bytes.
    """
    if cfg is None:
        if isinstance(config, WrapperConfig):
            cfg = config
        else:
            cfg = load_config(config)
    vid = cfg.video
    used_seed = cfg.generation.seed if seed is None else seed
    if used_seed is None or int(used_seed) < 0:
        used_seed = random.randint(0, 2**32 - 1)
    used_seed = int(used_seed)

    ref_img_list = [ref_images] if isinstance(ref_images, (str, Path)) else list(ref_images or [])
    ref_vid_list = [ref_videos] if isinstance(ref_videos, (str, Path)) else list(ref_videos or [])
    ref_aud_list = [ref_audios] if isinstance(ref_audios, (str, Path)) else list(ref_audios or [])
    raw_mode = str(mode if mode is not None else vid.mode or "i2v").strip().lower()
    if raw_mode in {"r2v", "ref", "reference", "r2v-lite", "r2vlite"}:
        use_r2v = True
    elif raw_mode in {"i2v", "flf", "flf2v", "t2v", ""}:
        use_r2v = bool(ref_img_list or ref_vid_list or ref_aud_list)
    else:
        use_r2v = bool(ref_img_list or ref_vid_list or ref_aud_list)
    # In R2V-lite the first-frame still doubles as <Picture 1> when no refs given.
    if use_r2v and image is not None and not ref_img_list and not ref_vid_list:
        ref_img_list = [image]
        image_for_size = image
    else:
        image_for_size = image
    if not use_r2v and image is None:
        raise ValueError("I2V mode needs --image <still>; use --mode r2v for prompt-only or reference-driven generation")
    if image_for_size is not None and width is None and height is None:
        try:
            width, height = target_dimensions(image_for_size, megapixels=vid.size_megapixels)
        except Exception:
            width, height = h3_size(vid.width, vid.height, fit=False)
    else:
        width, height = h3_size(width or vid.width, height or vid.height, fit=False)
    frames_n = int(frames or vid.frames)
    fps_n = int(fps or vid.fps)
    rec = recipe_settings(recipe if recipe is not None else vid.recipe)
    client = client or ComfyClient(cfg=cfg)
    live = client if client.ping() else None
    wanted_acc = (acc_file if acc_file is not None else vid.acc_file) if rec.acc else ""
    stack = coerce_video_stack(
        unet=unet or vid.unet,
        clip=clip or vid.clip,
        vae=vae or vid.vae,
        audio_vae=audio_vae or vid.audio_vae,
        acc_file=wanted_acc,
        recipe=rec,
        defaults=vid,
        unets=list_unets(cfg, live),
        clips=list_text_encoders(cfg, live),
        vaes=list_vaes(cfg, live),
        acc_files=list_pdd_acc(cfg, live),
    )
    rec = stack["recipe"]
    unet_n = stack["unet"]
    clip_n = stack["clip"]
    vae_n = stack["vae"]
    audio_vae_n = stack["audio_vae"]
    acc_n = stack["acc_file"] if rec.acc else ""
    if rec.acc:
        steps_n = int(stack["steps"])
        sampler_n = stack["sampler"]
        scheduler_n = stack["scheduler"]
    else:
        steps_n = int(steps if steps is not None else rec.steps)
        sampler_n = sampler or rec.sampler
        scheduler_n = scheduler or rec.scheduler
    do_loop = vid.loop if loop is None else bool(loop)
    fade_n = int(crossfade_frames if crossfade_frames is not None else vid.crossfade_frames)
    do_latent = vid.latent_upscale if latent_upscale is None else bool(latent_upscale)
    do_rtx = vid.rtx_vsr if rtx_vsr is None else bool(rtx_vsr)
    ref_size_n = normalize_ref_image_size(ref_image_size if ref_image_size is not None else vid.ref_image_size)
    prefix = filename_prefix or ("h3_r2v" if use_r2v else "h3_loop")
    positive = prompt if prompt is not None else (vid.prompt or cfg.generation.positive)

    image_key = client.upload_image(image, "wrapper/h3") if image is not None and not use_r2v else None
    last_key = client.upload_image(last_frame, "wrapper/h3") if last_frame is not None and not use_r2v else None
    ref_image_keys = [client.upload_image(p, "wrapper/h3/refs") for p in ref_img_list] if use_r2v else []
    ref_video_keys = (
        [stage_input_video(p, client=client, cfg=cfg) for p in ref_vid_list] if use_r2v else []
    )
    ref_audio_keys = (
        [stage_input_video(p, client=client, cfg=cfg) for p in ref_aud_list] if use_r2v else []
    )

    wf = load_workflow(cfg.r2v_workflow_path() if use_r2v else cfg.video_workflow_path())
    settings = {
        "positive": positive,
        "seed": used_seed,
        "steps": steps_n,
        "sampler": sampler_n,
        "scheduler": scheduler_n,
        "denoise": 1.0,
        "width": width,
        "height": height,
        "length": frames_n,
        "fps": float(fps_n),
        "checkpoint": unet_n,
        "clip": clip_n,
        "vae": vae_n,
        "audio_vae": audio_vae_n,
        "image": image_key,
        "ref_image_size": ref_size_n,
        "filename_prefix": prefix,
    }
    wf = apply_settings(wf, settings, bindings=merge_bindings(wf, cfg.workflow_map))
    if not use_r2v and last_key:
        wf = set_last_frame(wf, last_key)
    if use_r2v:
        wf = set_r2v_refs(
            wf,
            ref_images=ref_image_keys,
            ref_videos=ref_video_keys,
            ref_audios=ref_audio_keys,
            ref_image_size=ref_size_n,
        )
    wf = apply_h3_graph(
        wf,
        recipe=rec,
        acc_file=acc_n if rec.acc else None,
        unet=unet_n,
        steps=steps_n,
    )
    if do_latent:
        wf = apply_h3_latent_upscale(
            wf,
            model_name=vid.latent_upscale_model,
            scale=vid.latent_upscale_scale,
        )
    if dump_path:
        dump_workflow(wf, dump_path)

    prompt_id = client.queue_prompt(wf)
    images: list[dict] = []
    saved: list[str] = []
    loop_error = ""
    if wait:
        result = client.wait(prompt_id, on_progress=on_progress)
        images = result.images
        if (cfg.batch.save_locally or do_rtx) and images:
            saved = _save_locally(client, images, cfg.paths.output, prefix)
        if do_loop and saved:
            raw = Path(saved[0])
            looped = raw.with_name(raw.stem + "_loop" + raw.suffix)
            try:
                cyclic_crossfade(raw, looped, fps=fps_n, crossfade_frames=fade_n)
                saved.append(str(looped))
            except Exception as exc:
                loop_error = str(exc)
        if do_rtx and saved:
            source = Path(saved[-1])
            rtx_dest = source.with_name(source.stem + "__rtx" + source.suffix)
            try:
                if on_progress:
                    on_progress({"type": "executing", "data": {"node": "rtx_vsr"}})
                apply_rtx_vsr(
                    source,
                    rtx_dest,
                    client=client,
                    cfg=cfg,
                    scale=vid.rtx_vsr_scale,
                    quality=vid.rtx_vsr_quality,
                    filename_prefix=prefix + "_rtx",
                    on_progress=on_progress,
                )
                saved.append(str(rtx_dest))
            except Exception as exc:
                loop_error = (loop_error + "; " if loop_error else "") + f"rtx_vsr: {exc}"
        append_history(
            cfg.paths.history,
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "prompt_id": prompt_id,
                "kind": "video",
                "mode": "r2v" if use_r2v else ("flf2v" if last_key else "i2v"),
                "recipe": rec.key,
                "acc_file": acc_n if rec.acc else "",
                "steps": steps_n,
                "ref_images": len(ref_image_keys),
                "ref_videos": len(ref_video_keys),
                "ref_audios": len(ref_audio_keys),
                "ref_image_size": ref_size_n if use_r2v else "",
                "last_frame": bool(last_key),
                "latent_upscale": do_latent,
                "rtx_vsr": do_rtx,
                "seed": used_seed,
                "prompt_preview": (positive or "")[:200],
                "files": [Path(p).name for p in saved],
                "loop_error": loop_error,
            },
        )

    return GenerateResult(
        prompt_id=prompt_id,
        workflow=wf,
        images=images,
        saved_files=saved,
        seed=used_seed,
        positive=positive or "",
        negative="",
        loras=[],
        gen_hash=compute_generation_hash(
            positive or "",
            "",
            steps=steps_n,
            sampler=sampler_n,
            cfg=1.0,
            width=width,
            height=height,
            checkpoint=unet_n,
        ),
    )
