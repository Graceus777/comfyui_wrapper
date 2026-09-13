"""Load API-format ComfyUI workflows, detect bindings, patch settings, inject LoRAs."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

SAMPLER_ALIASES = {
    "euler a": "euler_ancestral",
    "euler_a": "euler_ancestral",
    "ancestral": "euler_ancestral",
    "dpm++ 2m": "dpmpp_2m",
    "dpm++ 2m karras": "dpmpp_2m",
    "dpm++ sde": "dpmpp_sde",
    "dpm++ 2s a": "dpmpp_2s_ancestral",
    "dpm++ 2m sde": "dpmpp_2m_sde",
    "ddim": "ddim",
    "uni_pc": "uni_pc",
    "lms": "lms",
    "heun": "heun",
    "euler": "euler",
}

SETTING_KEYS = (
    "positive",
    "negative",
    "seed",
    "steps",
    "cfg",
    "sampler",
    "scheduler",
    "denoise",
    "width",
    "height",
    "batch_size",
    "checkpoint",
    "filename_prefix",
    "clip",
    "clip2",
    "vae",
    "length",
    "fps",
    "image",
    "audio_vae",
)

# class_type candidates used when walking the graph.
_MODEL_LOADERS = {
    "CheckpointLoaderSimple",
    "CheckpointLoader",
    "UNETLoader",
    "UnetLoaderGGUF",
    "ImageOnlyCheckpointLoader",
}
_CLIP_LOADERS = {
    "CheckpointLoaderSimple",
    "CheckpointLoader",
    "CLIPLoader",
    "DualCLIPLoader",
    "DualCLIPLoaderGGUF",
    "TripleCLIPLoader",
    "CLIPLoaderGGUF",
}
_VIDEO_CONDITIONING = {
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
}
_LORA_NODES = {"LoraLoader", "LoraLoaderModelOnly"}
_LATENT_NODES = {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyHunyuanLatentVideo"}
_SAMPLER_NODES = {"KSampler", "KSamplerAdvanced"}


class WorkflowError(ValueError):
    pass


def load_workflow(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise WorkflowError(f"Workflow not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if is_ui_format(data):
        raise WorkflowError(
            f"{path.name} looks like a ComfyUI UI graph (has nodes/links). "
            "Export it with File → Export (API) and point config.workflow.file at that JSON."
        )
    if not isinstance(data, dict) or not any(
        isinstance(v, dict) and "class_type" in v for v in data.values()
    ):
        raise WorkflowError(f"{path.name} is not a ComfyUI API-format workflow")
    return data


def is_ui_format(data: Any) -> bool:
    return isinstance(data, dict) and "nodes" in data and "links" in data and "class_type" not in next(
        (v for v in data.values() if isinstance(v, dict)), {}
    )


def nodes_of_type(workflow: dict, class_types: Iterable[str]) -> list[tuple[str, dict]]:
    wanted = set(class_types)
    return [
        (str(nid), node)
        for nid, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") in wanted
    ]


def node_title(node: dict) -> str:
    meta = node.get("_meta") or {}
    return str(meta.get("title") or "")


def link_ref(value: Any) -> tuple[str, int] | None:
    if isinstance(value, list) and len(value) >= 2:
        return str(value[0]), int(value[1])
    return None


def follow(workflow: dict, node_id: str, input_name: str) -> tuple[str, int] | None:
    node = workflow.get(str(node_id))
    if not node:
        return None
    return link_ref((node.get("inputs") or {}).get(input_name))


def walk_skip_loras(workflow: dict, node_id: str, slot: int) -> tuple[str, int]:
    """Walk upstream through LoraLoader nodes to the original model/clip source."""
    current_id, current_slot = str(node_id), slot
    seen: set[str] = set()
    while current_id not in seen:
        seen.add(current_id)
        node = workflow.get(current_id)
        if not node:
            break
        ctype = node.get("class_type")
        if ctype not in _LORA_NODES:
            break
        inputs = node.get("inputs") or {}
        if current_slot == 0 and "model" in inputs:
            ref = link_ref(inputs["model"])
        elif "clip" in inputs:
            ref = link_ref(inputs["clip"])
        else:
            ref = link_ref(inputs.get("model"))
        if not ref:
            break
        current_id, current_slot = ref
    return current_id, current_slot


def last_lora_or_source(workflow: dict, source_id: str, source_slot: int) -> tuple[str, int]:
    """Find the terminal LoraLoader that currently feeds from this source, else the source."""
    terminal = (str(source_id), source_slot)
    changed = True
    while changed:
        changed = False
        for nid, node in workflow.items():
            if not isinstance(node, dict) or node.get("class_type") not in _LORA_NODES:
                continue
            inputs = node.get("inputs") or {}
            key = "model" if source_slot == 0 and "model" in inputs else "clip" if "clip" in inputs else "model"
            ref = link_ref(inputs.get(key))
            if ref == terminal:
                out_slot = 0 if key == "model" else 1
                if node.get("class_type") == "LoraLoaderModelOnly":
                    out_slot = 0
                terminal = (str(nid), out_slot)
                changed = True
                break
    return terminal


def detect_bindings(workflow: dict) -> dict[str, tuple[str, str]]:
    """Map logical setting names to (node_id, input_key)."""
    bindings: dict[str, tuple[str, str]] = {}

    def set_if_absent(key: str, node_id: str, input_key: str) -> None:
        if key not in bindings:
            bindings[key] = (str(node_id), input_key)

    # Title-based CLIPTextEncode detection first.
    encodes = nodes_of_type(workflow, ["CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeFlux"])
    for nid, node in encodes:
        title = node_title(node).lower()
        if "negative" in title:
            set_if_absent("negative", nid, "text")
        elif "positive" in title or title in {"prompt", "positive prompt"}:
            set_if_absent("positive", nid, "text")

    samplers = nodes_of_type(workflow, _SAMPLER_NODES)
    if samplers:
        sid, snode = samplers[0]
        inputs = snode.get("inputs") or {}
        for logical, field in (
            ("seed", "seed"),
            ("steps", "steps"),
            ("cfg", "cfg"),
            ("sampler", "sampler_name"),
            ("scheduler", "scheduler"),
            ("denoise", "denoise"),
        ):
            if field in inputs:
                set_if_absent(logical, sid, field)
        pos = link_ref(inputs.get("positive"))
        if pos:
            set_if_absent("positive", pos[0], "text")
        neg = link_ref(inputs.get("negative"))
        if neg:
            set_if_absent("negative", neg[0], "text")
        model = link_ref(inputs.get("model"))
        if model:
            src, _ = walk_skip_loras(workflow, *model)
            src_node = workflow.get(src) or {}
            if src_node.get("class_type") in _MODEL_LOADERS:
                ckpt_key = "ckpt_name" if "ckpt_name" in (src_node.get("inputs") or {}) else "unet_name"
                if ckpt_key in (src_node.get("inputs") or {}):
                    set_if_absent("checkpoint", src, ckpt_key)
        latent = link_ref(inputs.get("latent_image"))
        if latent:
            lnode = workflow.get(latent[0]) or {}
            linputs = lnode.get("inputs") or {}
            for logical in ("width", "height", "batch_size"):
                if logical in linputs:
                    set_if_absent(logical, latent[0], logical)

    # Flux / custom sampler graph.
    for nid, node in nodes_of_type(workflow, ["RandomNoise"]):
        if "noise_seed" in (node.get("inputs") or {}):
            set_if_absent("seed", nid, "noise_seed")
    for nid, node in nodes_of_type(workflow, ["BasicScheduler"]):
        inputs = node.get("inputs") or {}
        if "steps" in inputs:
            set_if_absent("steps", nid, "steps")
        if "scheduler" in inputs:
            set_if_absent("scheduler", nid, "scheduler")
        if "denoise" in inputs:
            set_if_absent("denoise", nid, "denoise")
    for nid, node in nodes_of_type(workflow, ["KSamplerSelect"]):
        if "sampler_name" in (node.get("inputs") or {}):
            set_if_absent("sampler", nid, "sampler_name")
    for nid, node in nodes_of_type(workflow, ["FluxGuidance"]):
        if "guidance" in (node.get("inputs") or {}):
            set_if_absent("cfg", nid, "guidance")

    for nid, node in nodes_of_type(workflow, _LATENT_NODES):
        inputs = node.get("inputs") or {}
        for logical in ("width", "height", "batch_size"):
            if logical in inputs:
                set_if_absent(logical, nid, logical)

    for nid, node in nodes_of_type(workflow, _MODEL_LOADERS):
        inputs = node.get("inputs") or {}
        if "ckpt_name" in inputs:
            set_if_absent("checkpoint", nid, "ckpt_name")
        elif "unet_name" in inputs:
            set_if_absent("checkpoint", nid, "unet_name")

    for nid, node in nodes_of_type(workflow, ["SaveImage", "SaveImageWebsocket", "SaveVideo"]):
        if "filename_prefix" in (node.get("inputs") or {}):
            set_if_absent("filename_prefix", nid, "filename_prefix")
            break

    for nid, node in nodes_of_type(workflow, ["CLIPLoader", "CLIPLoaderGGUF"]):
        if "clip_name" in (node.get("inputs") or {}):
            set_if_absent("clip", nid, "clip_name")
            break
    for nid, node in nodes_of_type(workflow, ["DualCLIPLoader", "DualCLIPLoaderGGUF"]):
        inputs = node.get("inputs") or {}
        if "clip_name1" in inputs:
            set_if_absent("clip", nid, "clip_name1")
        if "clip_name2" in inputs:
            set_if_absent("clip2", nid, "clip_name2")
        break
    for nid, node in nodes_of_type(workflow, ["VAELoader"]):
        if "vae_name" not in (node.get("inputs") or {}):
            continue
        name = str((node.get("inputs") or {}).get("vae_name") or "").lower()
        title = node_title(node).lower()
        if "audio" in name or "audio" in title:
            set_if_absent("audio_vae", nid, "vae_name")
        else:
            set_if_absent("vae", nid, "vae_name")
    for nid, node in nodes_of_type(workflow, list(_VIDEO_CONDITIONING)):
        inputs = node.get("inputs") or {}
        if "prompt" in inputs:
            set_if_absent("positive", nid, "prompt")
        if "width" in inputs:
            set_if_absent("width", nid, "width")
        if "height" in inputs:
            set_if_absent("height", nid, "height")
        if "length" in inputs:
            set_if_absent("length", nid, "length")
        break
    for nid, node in nodes_of_type(workflow, ["LoadImage"]):
        if "image" in (node.get("inputs") or {}):
            set_if_absent("image", nid, "image")
            break
    for nid, node in nodes_of_type(workflow, ["CreateVideo"]):
        if "fps" in (node.get("inputs") or {}):
            set_if_absent("fps", nid, "fps")
            break

    # Remaining CLIPTextEncode: first unused = positive, second = negative.
    unused = [
        (nid, node)
        for nid, node in encodes
        if nid not in {bindings.get("positive", ("",))[0], bindings.get("negative", ("",))[0]}
    ]
    if "positive" not in bindings and unused:
        set_if_absent("positive", unused[0][0], "text")
        unused = unused[1:]
    if "negative" not in bindings and unused:
        set_if_absent("negative", unused[0][0], "text")

    return bindings


def parse_map_entry(value: str) -> tuple[str, str]:
    """Accept '6.inputs.text', '6.text', or '6/text'."""
    text = value.strip().replace("/", ".")
    parts = [p for p in text.split(".") if p and p != "inputs"]
    if len(parts) < 2:
        raise WorkflowError(f"Invalid workflow map entry: {value!r} (expected node_id.input_name)")
    return parts[0], parts[-1]


def merge_bindings(workflow: dict, explicit: dict | None = None) -> dict[str, tuple[str, str]]:
    bindings = detect_bindings(workflow)
    for key, value in (explicit or {}).items():
        if not value:
            continue
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            bindings[key] = (str(value[0]), str(value[1]))
        else:
            bindings[key] = parse_map_entry(str(value))
    return bindings


def normalize_sampler(name: str) -> str:
    key = str(name).strip().lower()
    return SAMPLER_ALIASES.get(key, str(name).strip())


def apply_settings(
    workflow: dict,
    settings: dict[str, Any],
    *,
    bindings: dict[str, tuple[str, str]] | None = None,
    explicit_map: dict | None = None,
) -> dict:
    """Return a patched copy of the workflow."""
    wf = deepcopy(workflow)
    bindings = bindings or merge_bindings(wf, explicit_map)
    for key, value in settings.items():
        if value is None or key not in bindings:
            continue
        node_id, input_key = bindings[key]
        node = wf.get(str(node_id))
        if not node:
            raise WorkflowError(f"Binding {key} points at missing node {node_id}")
        inputs = node.setdefault("inputs", {})
        if key == "sampler":
            value = normalize_sampler(str(value))
        inputs[input_key] = value
        # CLIPTextEncodeSDXL uses text_g / text_l instead of (or in addition to) text.
        if key in {"positive", "negative"} and input_key == "text":
            if "text_g" in inputs:
                inputs["text_g"] = value
            if "text_l" in inputs:
                inputs["text_l"] = value
    _broadcast_geometry(wf, settings)
    return wf


def _broadcast_geometry(wf: dict, settings: dict[str, Any]) -> None:
    """Patch width/height/length/fps onto every node that has those inputs."""
    width = settings.get("width")
    height = settings.get("height")
    length = settings.get("length")
    fps = settings.get("fps")
    image = settings.get("image")
    positive = settings.get("positive")
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        ctype = node.get("class_type")
        if width is not None and "width" in inputs:
            inputs["width"] = int(width)
        if height is not None and "height" in inputs:
            inputs["height"] = int(height)
        if length is not None and "length" in inputs and ctype in _VIDEO_CONDITIONING:
            inputs["length"] = int(length)
        if fps is not None and "fps" in inputs and ctype == "CreateVideo":
            inputs["fps"] = float(fps)
        if image is not None and "image" in inputs and ctype == "LoadImage":
            inputs["image"] = image
        if positive is not None and "prompt" in inputs and ctype in _VIDEO_CONDITIONING:
            inputs["prompt"] = positive


def _next_node_id(workflow: dict, prefix: str = "lora") -> str:
    i = 0
    while f"{prefix}_{i}" in workflow:
        i += 1
    return f"{prefix}_{i}"


def find_model_clip_sources(workflow: dict) -> tuple[tuple[str, int] | None, tuple[str, int] | None]:
    """Locate the original MODEL and CLIP loader outputs (before any LoRA chain)."""
    model_src = None
    clip_src = None
    samplers = nodes_of_type(workflow, _SAMPLER_NODES)
    if samplers:
        ref = link_ref((samplers[0][1].get("inputs") or {}).get("model"))
        if ref:
            model_src = walk_skip_loras(workflow, *ref)
    if model_src is None:
        for nid, node in nodes_of_type(workflow, ["BasicGuider", "SamplerCustomAdvanced"]):
            ref = link_ref((node.get("inputs") or {}).get("model"))
            if ref:
                model_src = walk_skip_loras(workflow, *ref)
                break
    if model_src is None:
        loaders = nodes_of_type(workflow, _MODEL_LOADERS)
        if loaders:
            model_src = (loaders[0][0], 0)

    encodes = nodes_of_type(workflow, ["CLIPTextEncode", "CLIPTextEncodeSDXL", "CLIPTextEncodeFlux"])
    if encodes:
        ref = link_ref((encodes[0][1].get("inputs") or {}).get("clip"))
        if ref:
            clip_src = walk_skip_loras(workflow, *ref)
    if clip_src is None:
        loaders = nodes_of_type(workflow, _CLIP_LOADERS)
        if loaders:
            node = loaders[0][1]
            slot = 1 if node.get("class_type") in {"CheckpointLoaderSimple", "CheckpointLoader"} else 0
            clip_src = (loaders[0][0], slot)
    return model_src, clip_src


def rewire_consumers(
    workflow: dict,
    old: tuple[str, int],
    new: tuple[str, int],
    *,
    skip_node_ids: set[str],
) -> None:
    old_id, old_slot = old
    for nid, node in workflow.items():
        if nid in skip_node_ids or not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            ref = link_ref(value)
            if ref == (old_id, old_slot):
                inputs[key] = [new[0], new[1]]


def inject_loras(
    workflow: dict,
    loras: list[dict],
    *,
    mode: str = "full",
) -> dict:
    """Chain LoraLoader nodes onto the workflow. Each item is {file, weight, clip_weight?}."""
    if not loras:
        return deepcopy(workflow)
    wf = deepcopy(workflow)
    model_src, clip_src = find_model_clip_sources(wf)
    if model_src is None:
        raise WorkflowError("Cannot inject LoRAs: no MODEL loader found in the workflow")
    use_full = mode != "model_only" and clip_src is not None
    if mode != "model_only" and clip_src is None:
        use_full = False

    current_model = last_lora_or_source(wf, *model_src)
    current_clip = last_lora_or_source(wf, *clip_src) if clip_src is not None else None
    created: list[str] = []

    for spec in loras:
        filename = spec.get("file") or spec.get("name")
        if not filename:
            continue
        weight = float(spec.get("weight", 1.0))
        clip_weight = float(spec.get("clip_weight", weight))
        nid = _next_node_id(wf)
        if use_full and current_clip is not None:
            wf[nid] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": [current_model[0], current_model[1]],
                    "clip": [current_clip[0], current_clip[1]],
                    "lora_name": filename,
                    "strength_model": weight,
                    "strength_clip": clip_weight,
                },
            }
            current_model = (nid, 0)
            current_clip = (nid, 1)
        else:
            wf[nid] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": [current_model[0], current_model[1]],
                    "lora_name": filename,
                    "strength_model": weight,
                },
            }
            current_model = (nid, 0)
        created.append(nid)

    skip = set(created)
    original_model_terminal = last_lora_or_source(
        {k: v for k, v in wf.items() if k not in skip}, *model_src
    )
    rewire_consumers(wf, original_model_terminal, current_model, skip_node_ids=skip)
    if use_full and clip_src is not None and current_clip is not None:
        original_clip_terminal = last_lora_or_source(
            {k: v for k, v in wf.items() if k not in skip}, *clip_src
        )
        rewire_consumers(wf, original_clip_terminal, current_clip, skip_node_ids=skip)
    return wf


def hires_target_size(width: int, height: int, scale: float) -> tuple[int, int]:
    w = int(round(int(width) * float(scale) / 8.0) * 8)
    h = int(round(int(height) * float(scale) / 8.0) * 8)
    return max(64, w), max(64, h)


def _find_decode_and_save(workflow: dict) -> tuple[str | None, str | None]:
    """Return (vae_decode_id, save_image_id) for the main output chain."""
    saves = nodes_of_type(workflow, ["SaveImage", "SaveImageWebsocket"])
    if not saves:
        return None, None
    save_id, save_node = saves[0]
    img = link_ref((save_node.get("inputs") or {}).get("images"))
    decode_id = img[0] if img else None
    return decode_id, save_id


def inject_hires(
    workflow: dict,
    *,
    scale: float = 1.5,
    steps: int = 0,
    denoise: float = 0.45,
    upscaler: str = "latent",
    width: int = 1024,
    height: int = 1280,
) -> dict:
    """Append an A1111-style hires pass after the first KSampler."""
    if any(str(k).startswith("hires_") for k in workflow):
        return workflow
    wf = deepcopy(workflow)
    samplers = nodes_of_type(wf, _SAMPLER_NODES)
    if not samplers:
        raise WorkflowError("Cannot apply hires.fix: no KSampler in the workflow")
    sid, snode = samplers[0]
    s_in = snode.get("inputs") or {}
    decode_id, save_id = _find_decode_and_save(wf)
    if not decode_id or decode_id not in wf:
        raise WorkflowError("Cannot apply hires.fix: no VAEDecode feeding SaveImage")
    decode = wf[decode_id]
    vae_ref = link_ref((decode.get("inputs") or {}).get("vae")) or ("4", 2)
    hires_steps = int(steps) if int(steps) > 0 else int(s_in.get("steps") or 20)
    hires_denoise = float(denoise)
    sampler = s_in.get("sampler_name") or "euler_ancestral"
    scheduler = s_in.get("scheduler") or "normal"
    cfg_v = s_in.get("cfg") if s_in.get("cfg") is not None else 5.0
    seed = s_in.get("seed") if s_in.get("seed") is not None else 0
    model_ref = link_ref(s_in.get("model"))
    pos_ref = link_ref(s_in.get("positive"))
    neg_ref = link_ref(s_in.get("negative"))
    if not model_ref or not pos_ref or not neg_ref:
        raise WorkflowError("Cannot apply hires.fix: KSampler missing model/conditioning")
    tw, th = hires_target_size(width, height, scale)
    method = (upscaler or "latent").strip()
    latent_mode = method.lower() in {"", "latent", "latent (nearest-exact)", "none"}

    if latent_mode:
        wf["hires_up"] = {
            "class_type": "LatentUpscaleBy",
            "inputs": {
                "samples": [sid, 0],
                "upscale_method": "nearest-exact",
                "scale_by": float(scale),
            },
        }
        latent_ref = ["hires_up", 0]
    else:
        wf["hires_loader"] = {
            "class_type": "UpscaleModelLoader",
            "inputs": {"model_name": method},
        }
        wf["hires_upimg"] = {
            "class_type": "ImageUpscaleWithModel",
            "inputs": {
                "upscale_model": ["hires_loader", 0],
                "image": [decode_id, 0],
            },
        }
        wf["hires_scale"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["hires_upimg", 0],
                "upscale_method": "lanczos",
                "width": tw,
                "height": th,
                "crop": "disabled",
            },
        }
        wf["hires_enc"] = {
            "class_type": "VAEEncode",
            "inputs": {
                "pixels": ["hires_scale", 0],
                "vae": [vae_ref[0], vae_ref[1]],
            },
        }
        latent_ref = ["hires_enc", 0]
        decode_from = None

    wf["hires_sampler"] = {
        "class_type": "KSampler",
        "inputs": {
            "seed": seed,
            "steps": hires_steps,
            "cfg": cfg_v,
            "sampler_name": sampler,
            "scheduler": scheduler,
            "denoise": hires_denoise,
            "model": [model_ref[0], model_ref[1]],
            "positive": [pos_ref[0], pos_ref[1]],
            "negative": [neg_ref[0], neg_ref[1]],
            "latent_image": latent_ref,
        },
    }
    if latent_mode:
        decode.setdefault("inputs", {})["samples"] = ["hires_sampler", 0]
    else:
        wf["hires_dec"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["hires_sampler", 0],
                "vae": [vae_ref[0], vae_ref[1]],
            },
        }
        if save_id and save_id in wf:
            wf[save_id].setdefault("inputs", {})["images"] = ["hires_dec", 0]
    return wf


def inject_adetailer(
    workflow: dict,
    *,
    model: str,
    denoise: float = 0.4,
    steps: int = 20,
    cfg: float = 5.0,
    guide_size: int = 512,
    seed: int = 0,
    sampler: str = "euler_ancestral",
    scheduler: str = "normal",
) -> dict:
    """Run Impact Pack FaceDetailer on the image that currently feeds SaveImage."""
    if any(str(k).startswith("ad_") for k in workflow):
        return workflow
    if not model:
        raise WorkflowError("ADetailer is enabled but no detector model was selected")
    wf = deepcopy(workflow)
    decode_id, save_id = _find_decode_and_save(wf)
    if not decode_id or decode_id not in wf or not save_id:
        raise WorkflowError("Cannot apply ADetailer: no VAEDecode/SaveImage chain")
    decode = wf[decode_id]
    d_in = decode.get("inputs") or {}
    vae_ref = link_ref(d_in.get("vae"))
    samples = link_ref(d_in.get("samples"))
    if not vae_ref or not samples:
        raise WorkflowError("Cannot apply ADetailer: VAEDecode missing vae/samples")
    sampler_node = wf.get(samples[0]) or {}
    s_in = sampler_node.get("inputs") or {}
    model_ref = link_ref(s_in.get("model"))
    pos_ref = link_ref(s_in.get("positive"))
    neg_ref = link_ref(s_in.get("negative"))
    if not model_ref or not pos_ref or not neg_ref:
        raise WorkflowError("Cannot apply ADetailer: sampler missing model/conditioning")
    clip_ref = None
    pos_node = wf.get(pos_ref[0]) or {}
    clip_ref = link_ref((pos_node.get("inputs") or {}).get("clip"))
    if clip_ref is None:
        raise WorkflowError("Cannot apply ADetailer: no CLIP on the positive encode node")

    wf["ad_detector"] = {
        "class_type": "UltralyticsDetectorProvider",
        "inputs": {"model_name": model},
    }
    wf["ad_face"] = {
        "class_type": "FaceDetailer",
        "inputs": {
            "image": [decode_id, 0],
            "model": [model_ref[0], model_ref[1]],
            "clip": [clip_ref[0], clip_ref[1]],
            "vae": [vae_ref[0], vae_ref[1]],
            "guide_size": float(guide_size),
            "guide_size_for": True,
            "max_size": 1024.0,
            "seed": int(seed),
            "steps": int(steps),
            "cfg": float(cfg),
            "sampler_name": sampler,
            "scheduler": scheduler,
            "positive": [pos_ref[0], pos_ref[1]],
            "negative": [neg_ref[0], neg_ref[1]],
            "denoise": float(denoise),
            "feather": 5,
            "noise_mask": True,
            "force_inpaint": True,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 3.0,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.93,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 10,
            "bbox_detector": ["ad_detector", 0],
            "wildcard": "",
            "cycle": 1,
        },
    }
    wf[save_id].setdefault("inputs", {})["images"] = ["ad_face", 0]
    return wf


def apply_vae_loader(workflow: dict, vae_name: str | None) -> dict:
    """Use a standalone VAELoader so VAE is cached independently of the checkpoint.

    CheckpointLoaderSimple always unpacks a VAE from the ckpt. When LoRAs change
    every batch job, ComfyUI offloads that VAE with the unet and reloads it for
    VAEDecode. A dedicated loader with a stable node id keeps the VAE resident.
    """
    name = str(vae_name or "").strip()
    if not name:
        return workflow
    wf = deepcopy(workflow)
    loader_id = None
    for nid, node in nodes_of_type(wf, ["VAELoader"]):
        title = node_title(node).lower()
        current = str((node.get("inputs") or {}).get("vae_name") or "").lower()
        if "audio" in title or "audio" in current:
            continue
        loader_id = nid
        break
    if loader_id is None:
        loader_id = "vae_loader"
        n = 0
        while loader_id in wf:
            n += 1
            loader_id = f"vae_loader_{n}"
        wf[loader_id] = {
            "class_type": "VAELoader",
            "_meta": {"title": "VAE"},
            "inputs": {"vae_name": name},
        }
    else:
        wf[loader_id].setdefault("inputs", {})["vae_name"] = name
    for nid, node in wf.items():
        if nid == loader_id or not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        if "vae" in inputs and link_ref(inputs.get("vae")):
            inputs["vae"] = [loader_id, 0]
    return wf


def dump_workflow(workflow: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(workflow, fh, indent=2)
