"""Input staging, preflight and execution for independent workflow containers."""
from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import subprocess
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from comfyui_wrapper.client import ComfyClient, ComfyError
from comfyui_wrapper.config import WrapperConfig
from comfyui_wrapper.generate import _save_locally
from comfyui_wrapper.history import append_history
from comfyui_wrapper.video import h3_size, prepare_swap_video, video_frame_size
from comfyui_wrapper import workflow_containers as recipes

KINDS = (
    "qwen_controlled", "qwen_consistent", "qwen_outfit", "qwen_outpaint",
    "h3_memory_i2v", "h3_lora_swap", "h3_omnichar", "h3_guided_upscale", "wan_keyframes",
)

# Native loaders validate filesystem existence themselves. Their dropdowns
# list top-level input files, while /upload/image also supports subfolders.
_INPUT_FILE_FIELDS = {"LoadImage": "image", "LoadVideo": "file"}


def file_path(value: Any, *, label: str, required: bool = True) -> Path | None:
    if isinstance(value, dict):
        value = value.get("path") or value.get("name")
    elif value is not None and not isinstance(value, (str, Path)):
        value = getattr(value, "path", None) or getattr(value, "name", None)
    if value is None or str(value).strip() == "":
        if required:
            raise ValueError(f"Supply {label}.")
        return None
    path = Path(str(value)).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"{label.capitalize()} is not a readable file: {path}")
    return path


def consecutive(values: list, *, minimum: int, label: str) -> list:
    """Do not silently move an uploaded keyframe into a different slot."""
    result = []
    empty_seen = False
    for value in values:
        if value is None or str(value).strip() == "":
            empty_seen = True
        elif empty_seen:
            raise ValueError(f"Fill {label} in order, without empty slots between files.")
        else:
            result.append(value)
    if len(result) < minimum:
        raise ValueError(f"Supply at least {minimum} {label}.")
    return result


def bounded_image_size(size: tuple[int, int], max_pixels: int = 1024**2) -> tuple[int, int]:
    width, height = size
    factor = min(1.0, math.sqrt(max_pixels / (width * height)))
    # Native Qwen references and their masks must agree on this one resize.
    return max(32, int(width * factor) // 32 * 32), max(32, int(height * factor) // 32 * 32)


def stage_image(source: Path, destination: Path, *, resize: bool = True,
                mask_size: tuple[int, int] | None = None) -> tuple[int, int]:
    from PIL import Image, ImageOps
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        size = image.size
        target = mask_size or (bounded_image_size(size) if resize else size)
        if image.size != target:
            method = Image.Resampling.NEAREST if mask_size else Image.Resampling.LANCZOS
            image = image.resize(target, method)
        image.save(destination, format="PNG")
    return target


def validate_graph(wf: recipes.Graph, info: Mapping[str, Any]) -> list[str]:
    """Check live node IDs, required inputs, link types and model choices.

    This uses object_info only. It never submits a prompt for 'validation'.
    Autogrow sockets are expanded using the server's own template definitions.
    """
    errors = []
    for key, entry in wf.items():
        name = entry["class_type"]
        schema = info.get(name)
        if schema is None:
            errors.append(f"{name}: node is not installed or failed to load")
            continue
        fields, required = {}, set()
        inputs = entry.get("inputs", {})
        for group in ("required", "optional"):
            for field, spec in schema.get("input", {}).get(group, {}).items():
                if spec[0] == "COMFY_AUTOGROW_V3":
                    template = spec[1]["template"]
                    inner = next(iter(template["input"]["required"].values()))
                    names = template.get("names")
                    matched = [f for f in inputs if f.startswith(field + ".")]
                    for full in matched:
                        suffix = full.split(".", 1)[1]
                        prefix = template.get("prefix", "")
                        tail = suffix[len(prefix):] if suffix.startswith(prefix) else ""
                        if (names is not None and suffix not in names) or (
                            names is None and (not tail.isdigit() or int(tail) >= template.get("max", 999))
                        ):
                            errors.append(f"{key}.{full}: unsupported reference slot")
                        fields[full] = inner
                    if len(matched) < template.get("min", 0):
                        errors.append(f"{key}.{field}: too few inputs")
                else:
                    fields[field] = spec
                    if group == "required":
                        required.add(field)
        for field in sorted(required - inputs.keys()):
            errors.append(f"{key}.{field}: missing required input")
        for field, value in inputs.items():
            spec = fields.get(field)
            if spec is None:
                errors.append(f"{key}.{field}: input is unavailable on this node version")
                continue
            expected = spec[0]
            options = spec[1] if len(spec) > 1 else {}
            is_link = isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)
            if is_link:
                source = wf.get(value[0])
                if source is None:
                    errors.append(f"{key}.{field}: dangling link {value[0]}")
                    continue
                outputs = info.get(source["class_type"], {}).get("output", [])
                if not isinstance(value[1], int) or not 0 <= value[1] < len(outputs):
                    errors.append(f"{key}.{field}: unavailable output {value}")
                elif isinstance(expected, str) and expected not in ("COMBO", "*") and outputs[value[1]] != expected:
                    errors.append(f"{key}.{field}: {outputs[value[1]]} cannot feed {expected}")
            else:
                choices = expected if isinstance(expected, list) else options.get("options") if expected == "COMBO" else None
                is_input_file = _INPUT_FILE_FIELDS.get(name) == field
                if choices is not None and value not in choices and not is_input_file:
                    errors.append(f"{key}.{field}: required choice/model is unavailable: {value}")
                if expected in ("INT", "FLOAT"):
                    if not isinstance(value, (int, float)) or not math.isfinite(value):
                        errors.append(f"{key}.{field}: requires a finite number")
                    elif value < options.get("min", -math.inf) or value > options.get("max", math.inf):
                        errors.append(f"{key}.{field}: outside the supported range")
    return errors


def assert_ready(wf: recipes.Graph, client: ComfyClient) -> None:
    errors = validate_graph(wf, client.get("/object_info", timeout=60))
    if errors:
        raise ComfyError("Workflow preflight failed before sampling:\n" + "\n".join(errors))


def inspect_character(source: Path, comfy_root: Path) -> dict:
    """Use ComfyUI's pinned reader, without adding its dependencies to the wrapper."""
    if source.suffix.lower() != ".char":
        raise ValueError("Choose an OmniChar .char file.")
    embedded = comfy_root.parent / "python_embeded" / "python.exe"
    if not embedded.is_file():
        raise ValueError("OmniChar needs the portable ComfyUI Python configured in paths.comfyui_root.")
    script = (
        "import json,sys; from omnichar_sdk import Character,Portability; "
        "c=Character.open(sys.argv[1]); r=c.get_references(limit=4); "
        "a=c.get_lora('minimax-h3'); "
        "print(json.dumps({'name':c.name,'references':len(r),'voice':c.get_voice() is not None,"
        "'adapter':a is not None and a.portability is Portability.PORTABLE,"
        "'adapter_reason':a.reason if a else 'No portable MiniMax H3 adapter in this character.'}))"
    )
    result = subprocess.run([str(embedded), "-s", "-c", script, str(source)],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise ValueError("Could not open this .char file: " + (result.stderr.strip().splitlines()[-1] if result.stderr else "reader failed"))
    data = json.loads(result.stdout)
    if not data["references"]:
        raise ValueError("This character has no original reference pictures.")
    return data


def stage_character(source: Path, comfy_root: Path) -> str:
    if source.suffix.lower() != ".char":
        raise ValueError("Choose an OmniChar .char file.")
    sha = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            sha.update(block)
    key = f"wrapper/{sha.hexdigest()[:24]}.char"
    destination = comfy_root / "models" / "characters" / key
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        part = destination.with_suffix(".char.part")
        shutil.copyfile(source, part)
        part.replace(destination)
    return key


def cancel_owned(client: ComfyClient, prompt_id: str) -> None:
    """Do not interrupt someone else's job if our prompt has already ended."""
    state = client.queue()
    if any(str(item[1]) == prompt_id for item in state.get("queue_running", [])):
        client.interrupt()
    elif any(str(item[1]) == prompt_id for item in state.get("queue_pending", [])):
        client.post("/queue", {"delete": [prompt_id]})


def run_container(kind: str, payload: dict, *, cfg: WrapperConfig,
                  client: ComfyClient | None = None,
                  stop_event: threading.Event | None = None,
                  on_progress: Callable[[dict], None] | None = None,
                  on_queued: Callable[[str], None] | None = None) -> dict:
    if kind not in KINDS:
        raise ValueError(f"Unknown dedicated workflow: {kind}")
    client = client or ComfyClient(cfg=cfg)
    if not client.ping():
        raise ComfyError(f"Start ComfyUI at {client.base} before generating.")
    state = client.queue()
    if state.get("queue_running") or state.get("queue_pending"):
        raise ComfyError("ComfyUI is already running or has queued work. Wait for it to finish first.")
    seed = int(payload.get("seed", -1))
    seed = random.SystemRandom().randrange(2**32) if seed < 0 else seed
    if not 0 <= seed < 2**64:
        raise ValueError("Seed must be -1 (random) or an unsigned 64-bit integer.")
    output = cfg.paths.output / "containers" / kind / uuid.uuid4().hex[:12]
    output.mkdir(parents=True, exist_ok=True)
    notes = []
    with tempfile.TemporaryDirectory(prefix="workflow_container_") as tmp:
        staging = Path(tmp)

        def upload(value, label, *, resize=True, optional=False, mask_size=None):
            source = file_path(value, label=label, required=not optional)
            if source is None:
                return "", None
            destination = staging / f"{uuid.uuid4().hex}.png"
            size = stage_image(source, destination, resize=resize, mask_size=mask_size)
            return client.upload_image(destination, "wrapper/containers"), size

        if kind == "qwen_controlled":
            values = consecutive(payload.get("images", []), minimum=1, label="reference images")
            if len(values) > 3:
                raise ValueError("This panel supports three references.")
            uploaded = [upload(value, f"reference {i + 1}") for i, value in enumerate(values)]
            mask_index = int(payload.get("mask_index", 1))
            mask_value = payload.get("mask")
            if mask_value and not 1 <= mask_index <= len(uploaded):
                raise ValueError("The mask's reference number must refer to a supplied image.")
            mask, _ = upload(mask_value, "reference mask", optional=True,
                             mask_size=uploaded[mask_index - 1][1] if mask_value else None)
            wf = recipes.build_qwen_controlled(
                [row[0] for row in uploaded], payload.get("prompt", ""), seed,
                strengths=[float(n) for n in payload.get("strengths", [1.0] * len(uploaded))][:len(uploaded)],
                mask_key=mask, mask_index=mask_index, mask_strength=float(payload.get("mask_strength", 1.0)),
                isolate_mask=bool(payload.get("isolate_mask", False)))
        elif kind == "qwen_consistent":
            image, _ = upload(payload.get("image"), "source image")
            wf = recipes.build_qwen_consistent(image, payload.get("prompt", ""), seed,
                                               strength=float(payload.get("strength", 1.0)))
        elif kind == "qwen_outfit":
            person, _ = upload(payload.get("image"), "person image")
            outfit, _ = upload(payload.get("outfit"), "outfit reference")
            text = payload.get("prompt") or (
                "Change the clothing of the person in image 1 to match the outfit in image 2. "
                "Keep the person, pose, background and framing of image 1."
            )
            wf = recipes.build_qwen_outfit(person, outfit, text, seed, strength=float(payload.get("strength", 1.0)))
        elif kind == "qwen_outpaint":
            image, size = upload(payload.get("image"), "source image", resize=False)
            wf = recipes.build_qwen_outpaint(image, size, margins=payload.get("margins", (128,) * 4),
                                              scene=payload.get("prompt", ""), seed=seed,
                                              strength=float(payload.get("strength", 1.0)))
        elif kind == "h3_memory_i2v":
            source = file_path(payload.get("image"), label="first frame")
            from comfyui_wrapper.video import target_dimensions
            width, height = target_dimensions(source, megapixels=float(payload.get("megapixels", 0.52)))
            image, _ = upload(source, "first frame")
            last, _ = upload(payload.get("last_image"), "last frame", optional=True)
            wf = recipes.build_h3_memory_i2v(image, payload.get("prompt", ""), seed,
                                             width=width, height=height, last_image_key=last,
                                             frames=round(float(payload.get("seconds", 5)) * 24))
        elif kind in ("h3_lora_swap", "h3_guided_upscale"):
            source = file_path(payload.get("video"), label="source clip")
            size = video_frame_size(source)
            if size is None:
                raise ValueError("Could not read the source clip; ffmpeg and ffprobe must be on PATH.")
            start, end = float(payload.get("start", 0)), float(payload.get("end", 5))
            if not end or end - start > 15:
                raise ValueError("Choose a source start and end spanning at most 15 seconds.")
            width, height = h3_size(*size, megapixels=float(payload.get("megapixels", 0.52))
                                    if kind == "h3_lora_swap" else min(size[0] * size[1] / 1e6, 0.9))
            prepared = staging / "source.mp4"
            delivered, frames = prepare_swap_video(source, prepared, width=width, height=height,
                                                    start_seconds=start, end_seconds=end)
            key = client.upload_image(prepared, "wrapper/containers")
            notes.append(f"Source: {delivered} frames at 24 fps; sampling: {frames} frames; source audio retained.")
            if kind == "h3_lora_swap":
                image, _ = upload(payload.get("image"), "replacement character")
                wf = recipes.build_h3_lora_swap(image, key, payload.get("target", "the main person"), seed,
                                                 width=width, height=height, frames=frames,
                                                 delivered_frames=delivered, instruction=payload.get("prompt", ""),
                                                 strength=float(payload.get("strength", 1.0)),
                                                 turbo=bool(payload.get("turbo", False)),
                                                 steps=payload.get("steps") if payload.get("override_steps", False) else None)
                notes.append(f"Turbo: {'on' if 'turbo' in wf else 'off'}; "
                             f"sampling: {wf['schedule']['inputs']['steps']} steps"
                             f"{' (override)' if payload.get('override_steps', False) else ''}.")
            else:
                wf = recipes.build_h3_guided_upscale(key, megapixels=float(payload.get("output_mp", 2.0)),
                                                      delivered_frames=delivered)
                notes.append(f"Guided upscale from {width} × {height}; output target {payload.get('output_mp', 2.0)} MP.")
        elif kind == "h3_omnichar":
            source = file_path(payload.get("character"), label="OmniChar .char file")
            root = cfg.paths.comfyui_root
            if not root or not (root / "main.py").is_file():
                raise ValueError("Set paths.comfyui_root to the local ComfyUI directory first.")
            metadata = inspect_character(source, root)
            use_voice = bool(payload.get("use_voice", True)) and metadata["voice"]
            use_adapter = bool(payload.get("use_adapter", True)) and metadata["adapter"]
            notes.append(f"{metadata['name']}: {metadata['references']} references; voice {'on' if use_voice else 'off'}; H3 adapter {'on' if use_adapter else 'off'}.")
            if payload.get("use_adapter", True) and not metadata["adapter"]:
                notes.append(metadata["adapter_reason"] + " Using reference conditioning.")
            key = stage_character(source, root)
            wf = recipes.build_h3_omnichar(key, payload.get("prompt", ""), metadata["references"], seed,
                                            frames=round(float(payload.get("seconds", 5)) * 24),
                                            use_voice=use_voice, use_adapter=use_adapter)
        else:
            values = consecutive(payload.get("images", []), minimum=2, label="keyframes")
            if len(values) > 5:
                raise ValueError("This panel supports up to five keyframes.")
            uploaded = [upload(value, f"keyframe {i + 1}") for i, value in enumerate(values)]
            source = file_path(values[0], label="first keyframe")
            from comfyui_wrapper.video import target_dimensions
            width, height = target_dimensions(source, megapixels=0.4)
            count = len(values) - 1
            wf = recipes.build_wan_keyframes([row[0] for row in uploaded],
                                               payload.get("prompts", [])[:count],
                                               payload.get("durations", [])[:count], seed,
                                               width=width, height=height)
            notes.append(f"{len(values)} ordered keyframes at {width} × {height}; Wan 2.2 / AniSora / Lightx2v, four steps per pair.")

    if stop_event and stop_event.is_set():
        raise ComfyError("Stopped before queueing.")
    assert_ready(wf, client)
    workflow_path = output / "workflow.api.json"
    workflow_path.write_text(json.dumps(wf, indent=2), encoding="utf-8")
    if on_progress:
        on_progress({"type": "container_status", "text": "Preflight passed. Queueing..."})
    if stop_event and stop_event.is_set():
        raise ComfyError("Stopped before queueing.")
    pid = client.queue_prompt(wf)
    if on_queued:
        on_queued(pid)
    try:
        if stop_event and stop_event.is_set():
            cancel_owned(client, pid)
            raise ComfyError("Stopped.")
        result = client.wait(pid, timeout_s=max(cfg.comfyui.timeout_s, 6 * 3600), on_progress=on_progress)
    except Exception:
        try:
            cancel_owned(client, pid)
        except Exception:
            pass  # Preserve the generation error if ComfyUI also became unreachable.
        raise
    files = _save_locally(client, result.images, output, kind)
    if not files:
        raise ComfyError(f"ComfyUI finished {pid} without saved media. Inspect its history before retrying.")
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "kind": kind,
              "container": kind, "prompt_id": pid, "seed": seed, "positive": payload.get("prompt", ""),
              "saved_files": files, "workflow_path": str(workflow_path), "notes": notes}
    (output / "result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    append_history(cfg.paths.history, record)
    return {**record, "files": files, "workflow": wf}
