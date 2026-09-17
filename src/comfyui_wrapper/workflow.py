"""Load API-format ComfyUI workflows, detect bindings, patch settings, inject LoRAs."""

from __future__ import annotations

import json
import math
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
    "init_image",
    "audio_vae",
    "ref_image_size",
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
        if "ref_image_size" in inputs:
            set_if_absent("ref_image_size", nid, "ref_image_size")
        break
    for nid, node in nodes_of_type(workflow, ["LoadImage"]):
        if str(nid).startswith(("h3_ref_", "h3_last_", "anima_control_image", "img2img_")):
            continue
        if "image" in (node.get("inputs") or {}):
            set_if_absent("image", nid, "image")
            break
    # img2img init image has its own key so it never clobbers control/video images.
    if "img2img_loader" in workflow:
        loader = workflow.get("img2img_loader") or {}
        if "image" in (loader.get("inputs") or {}):
            set_if_absent("init_image", "img2img_loader", "image")
    else:
        for nid, node in nodes_of_type(workflow, ["LoadImage"]):
            if str(nid).startswith("img2img_") and "image" in (node.get("inputs") or {}):
                set_if_absent("init_image", nid, "image")
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
    ref_image_size = settings.get("ref_image_size")
    for nid, node in wf.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        ctype = node.get("class_type")
        # Ref/last-frame helpers own their own image inputs; the generic
        # first-frame key must not clobber them. Anima controls and the
        # img2img init image have their own keys for the same reason.
        is_helper = str(nid).startswith(("h3_ref_", "h3_last_"))
        is_owned_image = str(nid).startswith(("anima_control_image", "img2img_"))
        if width is not None and "width" in inputs and not is_helper:
            inputs["width"] = int(width)
        if height is not None and "height" in inputs and not is_helper:
            inputs["height"] = int(height)
        if length is not None and "length" in inputs and ctype in _VIDEO_CONDITIONING:
            inputs["length"] = int(length)
        if fps is not None and "fps" in inputs and ctype == "CreateVideo":
            inputs["fps"] = float(fps)
        if image is not None and "image" in inputs and ctype == "LoadImage" and not is_helper and not is_owned_image:
            inputs["image"] = image
        if positive is not None and "prompt" in inputs and ctype in _VIDEO_CONDITIONING:
            inputs["prompt"] = positive
        if ref_image_size is not None and "ref_image_size" in inputs and ctype in _VIDEO_CONDITIONING:
            inputs["ref_image_size"] = ref_image_size


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


ANIMA_PREPROCESSORS = ("none", "canny")


def inject_anima_lllite(workflow: dict, controls: list[dict] | None) -> dict:
    """Chain built-in Anima LLLite model patches onto every sampler model.

    Each control accepts an optional ``preprocess`` step (``none`` = use the
    image as-is, i.e. already preprocessed; ``canny`` = derive edges in-graph
    with ComfyUI's built-in Canny node, A1111-preprocessor style, so a raw
    photo can feed the lineart patch). Canny thresholds use ``canny_low`` /
    ``canny_high`` (defaults 0.4 / 0.8).
    """
    if not controls:
        return deepcopy(workflow)
    wf = deepcopy(workflow)
    samplers = nodes_of_type(wf, _SAMPLER_NODES)
    if not samplers:
        raise WorkflowError("Cannot inject Anima LLLite: no KSampler found")
    current_model = link_ref((samplers[0][1].get("inputs") or {}).get("model"))
    if current_model is None:
        raise WorkflowError("Cannot inject Anima LLLite: sampler has no MODEL input")
    original_model = current_model
    created: list[str] = []

    for spec in controls:
        if not isinstance(spec, dict):
            raise WorkflowError("Each Anima LLLite control must be an object")
        image = str(spec.get("image") or "").strip()
        model_patch = str(spec.get("model_patch") or spec.get("file") or "").strip()
        if not image or not model_patch:
            raise WorkflowError("Anima LLLite controls require image and model_patch")
        strength = float(spec.get("strength", 1.0))
        start_percent = float(spec.get("start_percent", 0.0))
        end_percent = float(spec.get("end_percent", 1.0))
        if not all(math.isfinite(value) for value in (strength, start_percent, end_percent)):
            raise WorkflowError("Anima LLLite strength and schedule must be finite")
        if not -10.0 <= strength <= 10.0:
            raise WorkflowError("Anima LLLite strength must be between -10 and 10")
        if not 0.0 <= start_percent <= end_percent <= 1.0:
            raise WorkflowError("Anima LLLite schedule must satisfy 0 <= start <= end <= 1")
        preprocess = str(spec.get("preprocess") or "none").strip().lower()
        if preprocess not in ANIMA_PREPROCESSORS:
            raise WorkflowError(
                f"Unknown Anima preprocess {preprocess!r} (expected one of {list(ANIMA_PREPROCESSORS)})"
            )

        image_id = _next_node_id(wf, "anima_control_image")
        wf[image_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": image},
        }
        created.append(image_id)
        control_image_ref: list = [image_id, 0]
        if preprocess == "canny":
            try:
                canny_low = float(spec.get("canny_low", 0.4))
                canny_high = float(spec.get("canny_high", 0.8))
            except (TypeError, ValueError):
                raise WorkflowError("Anima Canny thresholds must be numbers")
            if not all(math.isfinite(v) for v in (canny_low, canny_high)):
                raise WorkflowError("Anima Canny thresholds must be finite")
            if not 0.01 <= canny_low < canny_high <= 0.99:
                raise WorkflowError("Anima Canny thresholds must satisfy 0.01 <= low < high <= 0.99")
            canny_id = _next_node_id(wf, "anima_canny")
            wf[canny_id] = {
                "class_type": "Canny",
                "_meta": {"title": "Anima Canny preprocess"},
                "inputs": {
                    "image": [image_id, 0],
                    "low_threshold": float(canny_low),
                    "high_threshold": float(canny_high),
                },
            }
            created.append(canny_id)
            control_image_ref = [canny_id, 0]
        patch_id = _next_node_id(wf, "anima_model_patch")
        wf[patch_id] = {
            "class_type": "ModelPatchLoader",
            "inputs": {"name": model_patch},
        }
        apply_id = _next_node_id(wf, "anima_lllite")
        wf[apply_id] = {
            "class_type": "AnimaLLLiteApply",
            "inputs": {
                "model": [current_model[0], current_model[1]],
                "model_patch": [patch_id, 0],
                "image": control_image_ref,
                "strength": strength,
                "start_percent": start_percent,
                "end_percent": end_percent,
            },
        }
        created.extend((patch_id, apply_id))
        current_model = (apply_id, 0)

    rewire_consumers(
        wf,
        original_model,
        current_model,
        skip_node_ids=set(created),
    )
    return wf


def inject_img2img(
    workflow: dict,
    *,
    image: str | None,
    denoise: float | None = None,
    width: int | None = None,
    height: int | None = None,
    batch_size: int | None = None,
    upscale_method: str = "lanczos",
    crop: str = "disabled",
) -> dict:
    """Convert a txt2img graph to img2img via LoadImage → ImageScale → VAEEncode.

    Works for SDXL, Anima, Krea2, and Flux GGUF graphs: the VAE is taken from
    the VAEDecode feeding SaveImage (or the standalone VAELoader), and the
    first KSampler's ``latent_image`` is rewired to the encoded init image.
    Idempotent: fixed ``img2img_*`` node ids are reused on repeat calls.
    """
    name = str(image or "").strip()
    if not name:
        return deepcopy(workflow)
    if denoise is not None:
        try:
            denoise_v = float(denoise)
        except (TypeError, ValueError):
            raise WorkflowError(f"img2img denoise must be a number (got {denoise!r})")
        if not math.isfinite(denoise_v) or not 0.0 < denoise_v <= 1.0:
            raise WorkflowError("img2img denoise must satisfy 0 < denoise <= 1")
    else:
        denoise_v = None
    wf = deepcopy(workflow)
    samplers = nodes_of_type(wf, _SAMPLER_NODES)
    if not samplers:
        raise WorkflowError("Cannot apply img2img: no KSampler found")
    sid, snode = samplers[0]
    s_in = snode.get("inputs") or {}
    latent_ref = link_ref(s_in.get("latent_image"))
    latent_node = wf.get(latent_ref[0]) if latent_ref else None
    latent_inputs = (latent_node.get("inputs") or {}) if isinstance(latent_node, dict) else {}

    target_w = int(width) if width is not None else int(latent_inputs.get("width") or 1024)
    target_h = int(height) if height is not None else int(latent_inputs.get("height") or 0)
    if target_h <= 0:
        # Fall back to whatever ImageScale/EmptyLatent already carries.
        for _nid, _node in nodes_of_type(wf, list(_LATENT_NODES)):
            _in = _node.get("inputs") or {}
            if target_h <= 0 and _in.get("height"):
                target_h = int(_in["height"])
            if target_w <= 0 and _in.get("width"):
                target_w = int(_in["width"])
    if target_w <= 0 or target_h <= 0:
        raise WorkflowError("Cannot apply img2img: width/height are unknown")
    target_batch = int(batch_size) if batch_size is not None else int(latent_inputs.get("batch_size") or 1)
    target_batch = max(1, target_batch)

    vae_ref: tuple[str, int] | None = None
    decode_id, _save_id = _find_decode_and_save(wf)
    if decode_id and decode_id in wf:
        vae_ref = link_ref(((wf[decode_id].get("inputs") or {}).get("vae")))
    if vae_ref is None:
        for nid, node in nodes_of_type(wf, ["VAELoader"]):
            title = node_title(node).lower()
            current = str((node.get("inputs") or {}).get("vae_name") or "").lower()
            if "audio" in title or "audio" in current:
                continue
            vae_ref = (nid, 0)
            break
    if vae_ref is None:
        for nid, node in nodes_of_type(wf, ["CheckpointLoaderSimple", "CheckpointLoader"]):
            vae_ref = (nid, 2)
            break
    if vae_ref is None:
        raise WorkflowError("Cannot apply img2img: no VAE source found")

    wf["img2img_loader"] = {
        "class_type": "LoadImage",
        "_meta": {"title": "img2img init image"},
        "inputs": {"image": name},
    }
    wf["img2img_scale"] = {
        "class_type": "ImageScale",
        "_meta": {"title": "img2img resize"},
        "inputs": {
            "image": ["img2img_loader", 0],
            "upscale_method": str(upscale_method or "lanczos"),
            "width": int(target_w),
            "height": int(target_h),
            "crop": str(crop or "disabled"),
        },
    }
    wf["img2img_encode"] = {
        "class_type": "VAEEncode",
        "_meta": {"title": "img2img VAEEncode"},
        "inputs": {
            "pixels": ["img2img_scale", 0],
            "vae": [vae_ref[0], vae_ref[1]],
        },
    }
    latent_out: list = ["img2img_encode", 0]
    if target_batch > 1:
        wf["img2img_repeat"] = {
            "class_type": "RepeatLatentBatch",
            "_meta": {"title": "img2img batch"},
            "inputs": {"samples": ["img2img_encode", 0], "amount": int(target_batch)},
        }
        latent_out = ["img2img_repeat", 0]
    elif "img2img_repeat" in wf:
        del wf["img2img_repeat"]

    s_in["latent_image"] = latent_out
    if denoise_v is not None:
        s_in["denoise"] = float(denoise_v)
    # Keep the (now unused) EmptyLatent geometry in sync so width/height
    # bindings and future txt2img runs stay consistent.
    if isinstance(latent_node, dict) and latent_node.get("class_type") in _LATENT_NODES:
        latent_inputs["width"] = int(target_w)
        latent_inputs["height"] = int(target_h)
        latent_inputs["batch_size"] = int(target_batch)
    return wf


def clear_img2img(workflow: dict) -> dict:
    """Remove img2img nodes and restore the EmptyLatent feed if possible."""
    wf = deepcopy(workflow)
    if not any(str(k).startswith("img2img_") for k in wf):
        return wf
    samplers = nodes_of_type(wf, _SAMPLER_NODES)
    latents = nodes_of_type(wf, list(_LATENT_NODES))
    if samplers and latents:
        sid = samplers[0][0]
        s_in = (wf[sid].get("inputs") or {})
        current = link_ref(s_in.get("latent_image"))
        if current and str(current[0]).startswith("img2img_"):
            s_in["latent_image"] = [latents[0][0], 0]
    for key in [k for k in list(wf) if str(k).startswith("img2img_")]:
        del wf[key]
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
