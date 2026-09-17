"""MiniMax H3 I2V recipes: native 12-step vs PDD Acc, with no-fault wiring.

Acc is MiniMaxH3PDDAccApply (not Load LoRA). Distills do not stack: turbo /
Spectrum / sage / SLA / Sol are disabled or bypassed when Acc is on. Wrong
sampler, steps, shift, or Acc file are swapped to the legal recipe instead of
being queued.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from comfyui_wrapper.workflow import (
    link_ref,
    nodes_of_type,
    rewire_consumers,
)

SHIFT_VIDEO = 12.0
SHIFT_AUDIO = 3.0
ACC_NODE = "MiniMaxH3PDDAccApply"
SHIFT_NODE = "MiniMaxH3SigmaShift"
I2V_NODE = "MiniMaxH3ImageToVideo"
R2V_NODE = "MiniMaxH3ReferenceToVideo"
GUIDE_NODE = "MiniMaxH3AddGuide"
ACC_ID = "h3_acc"
SHIFT_ID = "h3_shift"
AV_SPLIT_ID = "h3_av_split"
LATENT_UP_ID = "h3_latent_up"
LAST_IMAGE_ID = "h3_last_image"
LAST_SCALE_ID = "h3_last_scale"
REF_IMAGE_PREFIX = "h3_ref_img_"
REF_VIDEO_PREFIX = "h3_ref_vid_"
REF_VC_PREFIX = "h3_ref_vc_"
REF_AUDIO_PREFIX = "h3_ref_aud_"
MAX_REF_IMAGES = 9
MAX_REF_VIDEOS = 3
MAX_REF_AUDIOS = 3
DEFAULT_LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"

DEFAULT_UNET = "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
FALLBACK_UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
DEFAULT_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
DEFAULT_VAE = "minimax_h3_video_vae_fp16.safetensors"
DEFAULT_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
DEFAULT_ACC = {
    "fl2va": "MiniMax-H3-FL2VA-Acc-8Step.safetensors",
    "ref2va": "MiniMax-H3-Ref2VA-Acc-8Step.safetensors",
}

_DISTILL_MARKERS = (
    "turbo",
    "lightx2v",
    "lightx",
    "pdd_acc",
    "pdd-acc",
    "acc-8step",
    "acc_8step",
    "acc-4step",
    "acc_4step",
    "8step",
    "4step",
    "distill",
)
_ACC_FILE_MARKERS = ("pdd_acc", "pdd-acc", "-acc-", "_acc_", "acc-8", "acc_8", "acc-4", "acc_4")
_PATCH_BYPASS = {
    "PathchSageAttentionKJ",
    "PatchSageAttentionKJ",
    "MiniMaxH3MemoryEfficientSageAttentionPatch",
    "SolAttnPatch",
}
_DISABLE_WIDGETS = {
    "SpectrumApplyMiniMaxH3": {"enabled": False},
    "H3SLAAttention": {"sparsity_ratio": 0.0},
    "PathchSageAttentionKJ": {"sage_attention": "disabled"},
    "PatchSageAttentionKJ": {"sage_attention": "disabled"},
}


@dataclass(frozen=True)
class H3Recipe:
    key: str
    label: str
    steps: int
    sampler: str
    scheduler: str
    acc: bool
    nfe: int | None


RECIPES: dict[str, H3Recipe] = {
    "native": H3Recipe(
        key="native",
        label="Native (12-step)",
        steps=12,
        sampler="res_multistep",
        scheduler="simple",
        acc=False,
        nfe=None,
    ),
    "acc": H3Recipe(
        key="acc",
        label="Acc 8-step",
        steps=8,
        sampler="euler",
        scheduler="simple",
        acc=True,
        nfe=8,
    ),
    "acc4": H3Recipe(
        key="acc4",
        label="Acc 4-step",
        steps=4,
        sampler="euler",
        scheduler="simple",
        acc=True,
        nfe=4,
    ),
}

_RECIPE_ALIASES = {
    "": "native",
    "native": "native",
    "base": "native",
    "12": "native",
    "12step": "native",
    "resmultistep": "native",
    "acc": "acc",
    "acc8": "acc",
    "pdd": "acc",
    "pddacc": "acc",
    "8": "acc",
    "8step": "acc",
    "acc4": "acc4",
    "4": "acc4",
    "4step": "acc4",
}


def acc_nfe(name: str | None) -> int:
    """NFE is the Acc filename, not a dropdown rewrite. 8Step weights stay at 8."""
    text = (name or "").replace("\\", "/").lower().replace("_", "-")
    compact = text.replace("-", "")
    if "4step" in compact:
        return 4
    return 8


def has_acc_4step_weight(files: Iterable[str] | None) -> bool:
    """Alibaba currently ships only *-Acc-8Step.safetensors."""
    return any(acc_nfe(name) == 4 for name in (files or []) if name)


def normalize_recipe(raw: Any) -> str:
    if isinstance(raw, H3Recipe):
        return raw.key if raw.key in RECIPES else "native"
    if raw is True:
        return "acc"
    if raw is False or raw is None:
        return "native"
    key = str(raw).strip().lower().replace(" ", "").replace("_", "").replace("-", "")
    if key in _RECIPE_ALIASES:
        return _RECIPE_ALIASES[key]
    if "acc4" in key or key.endswith("4step"):
        return "acc4"
    if "acc" in key or "pdd" in key:
        return "acc"
    return "native"


def recipe_settings(raw: Any = None) -> H3Recipe:
    if isinstance(raw, H3Recipe):
        return RECIPES.get(raw.key, raw)
    return RECIPES[normalize_recipe(raw)]


def resolve_recipe(
    raw: Any = None,
    *,
    acc_file: str | None = None,
    acc_files: Iterable[str] | None = None,
) -> H3Recipe:
    """Acc 4-step stays hidden until a 4-step Acc weight is actually present."""
    rec = recipe_settings(raw)
    if rec.key != "acc4":
        return rec
    names = [n for n in (acc_files or []) if n]
    if acc_file:
        names = [acc_file, *names]
    if has_acc_4step_weight(names):
        return rec
    return RECIPES["acc"]


def recipe_choices(acc_files: Iterable[str] | None = None) -> list[tuple[str, str]]:
    choices = [(RECIPES["native"].label, RECIPES["native"].key), (RECIPES["acc"].label, RECIPES["acc"].key)]
    if has_acc_4step_weight(acc_files):
        choices.append((RECIPES["acc4"].label, RECIPES["acc4"].key))
    return choices


def video_recipe_ui(
    recipe_key: Any,
    unet: str | None,
    acc_files: Iterable[str] | None = None,
    preferred_acc: str | None = None,
) -> tuple[H3Recipe, int, str]:
    """Steps + Acc filename for the video tab. Native clears the Acc field."""
    files = [n for n in (acc_files or []) if n]
    rec = resolve_recipe(recipe_key, acc_file=preferred_acc, acc_files=files)
    if rec.acc:
        acc = match_acc_file(unet, files, preferred_acc, nfe=rec.nfe)
        return rec, acc_nfe(acc), acc
    return rec, rec.steps, ""


def h3_trunk(name: str | None) -> str:
    text = (name or "").replace("\\", "/").lower()
    if "ref2va" in text or "ref2v" in text:
        return "ref2va"
    return "fl2va"


def pick_model_name(choices: Iterable[str], *wanted: str) -> str | None:
    values = [str(x) for x in choices if x]
    if not values:
        return None
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
    return values[0]


def match_h3_unet(choices: Iterable[str], recipe: H3Recipe | str | None = None, preferred: str | None = None) -> str | None:
    rec = recipe if isinstance(recipe, H3Recipe) else recipe_settings(recipe)
    wanted: list[str] = []
    if preferred:
        wanted.append(preferred)
    if rec.acc:
        wanted.extend(
            [
                "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
                "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "minimax_h3_fl2va",
            ]
        )
    else:
        wanted.extend(
            [
                "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
                "minimax_h3_fl2va",
            ]
        )
    return pick_model_name(choices, *wanted)


def match_acc_file(
    unet: str | None,
    choices: Iterable[str] | None = None,
    preferred: str | None = None,
    nfe: int | None = None,
) -> str:
    """Pick the Acc file for this UNET trunk. A crossed FL2VA/Ref2VA pair is swapped.

    NFE follows the filename. There is no 4-step Acc weight yet, so a request
    for nfe=4 still returns the 8-step file.
    """
    trunk = h3_trunk(unet)
    default = DEFAULT_ACC[trunk]
    values = [str(x) for x in (choices or []) if x]
    tagged = [v for v in values if h3_trunk(v) == trunk]
    want_nfe = 4 if nfe == 4 else 8
    pool = [v for v in tagged if acc_nfe(v) == want_nfe]
    if not pool:
        pool = [v for v in tagged if acc_nfe(v) == 8] or tagged
        want_nfe = 8 if pool else want_nfe
    if preferred and h3_trunk(preferred) == trunk and acc_nfe(preferred) == want_nfe:
        if not values:
            return preferred
        hit = pick_model_name(pool or [preferred], preferred, default)
        if hit and h3_trunk(hit) == trunk:
            return hit
        return preferred
    if pool:
        hit = pick_model_name(pool, preferred or "", default, trunk, "acc")
        if hit and h3_trunk(hit) == trunk:
            return hit
    return default


def is_h3_unet(name: str | None) -> bool:
    text = (name or "").lower()
    return "minimax_h3" in text and ("fl2va" in text or "ref2va" in text or "ref2v" in text)


def is_h3_clip(name: str | None) -> bool:
    text = (name or "").lower()
    return "minimax" in text or "qwen3vl_32b" in text


def is_h3_video_vae(name: str | None) -> bool:
    text = (name or "").lower()
    return "minimax_h3" in text and "video" in text and "audio" not in text


def is_h3_audio_vae(name: str | None) -> bool:
    text = (name or "").lower()
    return "minimax_h3" in text and "audio" in text


def coerce_video_stack(
    *,
    unet: str | None,
    clip: str | None,
    vae: str | None,
    audio_vae: str | None,
    acc_file: str | None,
    recipe: H3Recipe | str | None,
    defaults: Any = None,
    unets: Iterable[str] | None = None,
    clips: Iterable[str] | None = None,
    vaes: Iterable[str] | None = None,
    acc_files: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Swap illegal H3 filenames onto the matching stack. Never pass a wrong trunk through."""
    rec = resolve_recipe(recipe, acc_file=acc_file, acc_files=acc_files)
    def_unet = getattr(defaults, "unet", None) or DEFAULT_UNET
    def_clip = getattr(defaults, "clip", None) or DEFAULT_CLIP
    def_vae = getattr(defaults, "vae", None) or DEFAULT_VAE
    def_audio = getattr(defaults, "audio_vae", None) or DEFAULT_AUDIO_VAE
    if is_h3_unet(unet):
        unet_n = unet
    elif unets:
        unet_n = match_h3_unet(unets, rec, def_unet) or def_unet
    else:
        unet_n = def_unet
    clip_n = clip if is_h3_clip(clip) else (pick_model_name(clips or [], def_clip, "qwen3vl_32b_minimax") or def_clip)
    vae_n = vae if is_h3_video_vae(vae) else (pick_model_name(vaes or [], def_vae, "minimax_h3_video_vae") or def_vae)
    audio_n = audio_vae if is_h3_audio_vae(audio_vae) else (
        pick_model_name(vaes or [], def_audio, "minimax_h3_audio_vae") or def_audio
    )
    acc_n = match_acc_file(unet_n, acc_files, acc_file, nfe=rec.nfe)
    if rec.acc:
        nfe = acc_nfe(acc_n)
        rec = RECIPES["acc"] if nfe != 4 else rec
        steps = nfe
        sampler = rec.sampler
        scheduler = rec.scheduler
    else:
        steps = rec.steps
        sampler = rec.sampler
        scheduler = rec.scheduler
    return {
        "recipe": rec,
        "unet": unet_n,
        "clip": clip_n,
        "vae": vae_n,
        "audio_vae": audio_n,
        "acc_file": acc_n,
        "steps": steps,
        "sampler": sampler,
        "scheduler": scheduler,
    }


def _is_distill_lora(name: str | None) -> bool:
    text = (name or "").replace("\\", "/").lower()
    if any(m in text for m in _ACC_FILE_MARKERS):
        return True
    return any(m in Path(text).name for m in _DISTILL_MARKERS)


def _bypass_model_node(workflow: dict, node_id: str) -> None:
    node = workflow.get(node_id)
    if not node:
        return
    src = link_ref((node.get("inputs") or {}).get("model"))
    if src is None:
        workflow.pop(node_id, None)
        return
    rewire_consumers(workflow, (node_id, 0), src, skip_node_ids={node_id})
    workflow.pop(node_id, None)


def _strip_distill_loras(workflow: dict) -> None:
    for nid, node in list(nodes_of_type(workflow, ["LoraLoader", "LoraLoaderModelOnly"])):
        name = str((node.get("inputs") or {}).get("lora_name") or "")
        if _is_distill_lora(name):
            _bypass_model_node(workflow, nid)


def _disable_conflicts(workflow: dict) -> None:
    for nid, node in list(workflow.items()):
        if not isinstance(node, dict):
            continue
        ctype = node.get("class_type")
        widgets = _DISABLE_WIDGETS.get(ctype)
        if widgets:
            inputs = node.setdefault("inputs", {})
            inputs.update(widgets)
        if ctype in _PATCH_BYPASS:
            _bypass_model_node(workflow, str(nid))


def _first_id(workflow: dict, class_types: Iterable[str]) -> str | None:
    found = nodes_of_type(workflow, class_types)
    return found[0][0] if found else None


def _ensure_sigma_shift(workflow: dict) -> str:
    existing = _first_id(workflow, [SHIFT_NODE])
    if existing:
        inputs = workflow[existing].setdefault("inputs", {})
        inputs["shift_video"] = SHIFT_VIDEO
        inputs["shift_audio"] = SHIFT_AUDIO
        return existing
    src = None
    guider_id = _first_id(workflow, ["BasicGuider", "CFGGuider"])
    if guider_id:
        src = link_ref((workflow[guider_id].get("inputs") or {}).get("model"))
    if src is None:
        loaders = nodes_of_type(
            workflow,
            ["UNETLoader", "UnetLoaderGGUF", "CheckpointLoaderSimple", "CheckpointLoader"],
        )
        if loaders:
            src = (loaders[0][0], 0)
    if src is None:
        raise ValueError("H3 graph has no MODEL source to attach MiniMaxH3SigmaShift")
    nid = SHIFT_ID if SHIFT_ID not in workflow else f"{SHIFT_ID}_1"
    workflow[nid] = {
        "class_type": SHIFT_NODE,
        "_meta": {"title": "Sigma Shift"},
        "inputs": {
            "model": [src[0], src[1]],
            "shift_video": SHIFT_VIDEO,
            "shift_audio": SHIFT_AUDIO,
        },
    }
    rewire_consumers(workflow, src, (nid, 0), skip_node_ids={nid})
    return nid


def _swap_cfg_guider(workflow: dict) -> None:
    for nid, node in nodes_of_type(workflow, ["CFGGuider"]):
        inputs = node.get("inputs") or {}
        cond = inputs.get("conditioning") or inputs.get("positive")
        model = inputs.get("model")
        node["class_type"] = "BasicGuider"
        node["inputs"] = {"model": model, "conditioning": cond}


def _set_sampler(workflow: dict, sampler: str, steps: int, scheduler: str) -> None:
    for _nid, node in nodes_of_type(workflow, ["KSamplerSelect"]):
        node.setdefault("inputs", {})["sampler_name"] = sampler
    for _nid, node in nodes_of_type(workflow, ["BasicScheduler"]):
        inputs = node.setdefault("inputs", {})
        inputs["steps"] = int(steps)
        inputs["scheduler"] = scheduler
        inputs["denoise"] = 1.0
    for _nid, node in nodes_of_type(workflow, ["KSampler", "KSamplerAdvanced"]):
        inputs = node.setdefault("inputs", {})
        if "sampler_name" in inputs:
            inputs["sampler_name"] = sampler
        if "steps" in inputs:
            inputs["steps"] = int(steps)
        if "scheduler" in inputs:
            inputs["scheduler"] = scheduler
        if "cfg" in inputs:
            inputs["cfg"] = 1.0


def _ensure_acc_node(workflow: dict, shift_id: str, *, acc_file: str, nfe: int, enabled: bool) -> str:
    existing = _first_id(workflow, [ACC_NODE])
    nid = existing or (ACC_ID if ACC_ID not in workflow else f"{ACC_ID}_1")
    scheduler_id = _first_id(workflow, ["BasicScheduler"])
    inputs: dict[str, Any] = {
        "model": [shift_id, 0],
        "pdd_file": acc_file,
        "nfe": str(int(nfe)),
        "lora_strength": 1.0,
        "head_strength": 1.0,
        "on_off_grid": "error",
        "enabled": bool(enabled),
        "partition": "",
        "partition_check": "error",
    }
    if scheduler_id:
        inputs["bypass_sigmas"] = [scheduler_id, 0]
    if nid in workflow and workflow[nid].get("class_type") == ACC_NODE:
        workflow[nid].setdefault("inputs", {}).update(inputs)
        workflow[nid]["_meta"] = {"title": "PDD Acc"}
    else:
        workflow[nid] = {
            "class_type": ACC_NODE,
            "_meta": {"title": "PDD Acc"},
            "inputs": inputs,
        }
    skip = {nid}
    if scheduler_id:
        skip.add(scheduler_id)
    rewire_consumers(workflow, (shift_id, 0), (nid, 0), skip_node_ids=skip)
    for _sid, sampler in nodes_of_type(workflow, ["SamplerCustomAdvanced"]):
        sampler.setdefault("inputs", {})["sigmas"] = [nid, 1]
    return nid


def apply_h3_graph(
    workflow: dict,
    *,
    recipe: H3Recipe | str | None = None,
    acc_file: str | None = None,
    unet: str | None = None,
    steps: int | None = None,
) -> dict:
    """Return a copy of the H3 graph with the recipe forced onto legal wiring."""
    wf = deepcopy(workflow)
    rec = resolve_recipe(recipe, acc_file=acc_file)
    if rec.acc:
        _strip_distill_loras(wf)
        _disable_conflicts(wf)
        _swap_cfg_guider(wf)
    shift_id = _ensure_sigma_shift(wf)
    if rec.acc:
        file_name = match_acc_file(unet, None, acc_file, nfe=rec.nfe)
        nfe = acc_nfe(file_name)
        _ensure_acc_node(wf, shift_id, acc_file=file_name, nfe=nfe, enabled=True)
        _set_sampler(wf, rec.sampler, nfe, rec.scheduler)
    else:
        existing = _first_id(wf, [ACC_NODE])
        scheduler_id = _first_id(wf, ["BasicScheduler"])
        if existing:
            inputs = wf[existing].setdefault("inputs", {})
            inputs["enabled"] = False
            if scheduler_id:
                inputs["bypass_sigmas"] = [scheduler_id, 0]
                for _sid, sampler in nodes_of_type(wf, ["SamplerCustomAdvanced"]):
                    sampler.setdefault("inputs", {})["sigmas"] = [scheduler_id, 0]
        native_steps = int(steps) if steps is not None else rec.steps
        _set_sampler(wf, rec.sampler, native_steps, rec.scheduler)
    return wf


def apply_h3_latent_upscale(
    workflow: dict,
    *,
    model_name: str | None = None,
    scale: float = 2.0,
    align: int = 32,
    enable_temporal_chunking: bool = True,
    force_unload: bool = True,
    device: str = "cuda",
    precision: str = "fp16",
) -> dict:
    """Split the sampled AV latent, 2x the video half, rewire decode. No second sample."""
    wf = deepcopy(workflow)
    samplers = nodes_of_type(wf, ["SamplerCustomAdvanced"])
    if not samplers:
        raise ValueError("H3 graph has no SamplerCustomAdvanced to attach MinimaxH3LatentUpscaler3D")
    sid = samplers[0][0]
    split_id = AV_SPLIT_ID if AV_SPLIT_ID not in wf else f"{AV_SPLIT_ID}_1"
    up_id = LATENT_UP_ID if LATENT_UP_ID not in wf else f"{LATENT_UP_ID}_1"
    wf[split_id] = {
        "class_type": "LTXVSeparateAVLatent",
        "_meta": {"title": "Split AV latent"},
        "inputs": {"av_latent": [sid, 0]},
    }
    wf[up_id] = {
        "class_type": "MinimaxH3LatentUpscaler3D",
        "_meta": {"title": "H3 3D latent upscaler"},
        "inputs": {
            "latent": [split_id, 0],
            "model_name": model_name or DEFAULT_LATENT_UPSCALER,
            "mode": "scale by multiplier",
            "mode.scale": float(scale),
            "align": int(align),
            "enable_temporal_chunking": bool(enable_temporal_chunking),
            "force_unload": bool(force_unload),
            "device": device,
            "precision": precision,
        },
    }
    for _nid, node in nodes_of_type(wf, ["VAEDecode", "VAEDecodeTiled"]):
        inputs = node.setdefault("inputs", {})
        src = link_ref(inputs.get("samples"))
        if src and src[0] == sid:
            inputs["samples"] = [up_id, 0]
    for _nid, node in nodes_of_type(wf, ["VAEDecodeAudio"]):
        inputs = node.setdefault("inputs", {})
        src = link_ref(inputs.get("samples"))
        if src and src[0] == sid:
            inputs["samples"] = [split_id, 1]
    return wf


def rtx_vsr_graph(
    input_key: str,
    prefix: str,
    *,
    scale: float = 2.0,
    quality: str = "ULTRA",
) -> dict:
    """Post-process graph: LoadVideo → RTXVideoSuperResolution → SaveVideo."""
    return {
        "1": {
            "class_type": "LoadVideo",
            "_meta": {"title": "Load clip"},
            "inputs": {"file": input_key},
        },
        "2": {
            "class_type": "GetVideoComponents",
            "inputs": {"video": ["1", 0]},
        },
        "3": {
            "class_type": "RTXVideoSuperResolution",
            "_meta": {"title": "RTX VSR"},
            "inputs": {
                "images": ["2", 0],
                "resize_type": "scale by multiplier",
                "resize_type.scale": float(scale),
                "quality": quality or "ULTRA",
            },
        },
        "4": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["3", 0],
                "fps": ["2", 2],
                "audio": ["2", 1],
                "bit_depth": 8,
            },
        },
        "5": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["4", 0],
                "filename_prefix": prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


def has_r2v_node(workflow: dict) -> bool:
    return _first_id(workflow, [R2V_NODE]) is not None


def has_i2v_node(workflow: dict) -> bool:
    return _first_id(workflow, [I2V_NODE]) is not None


def normalize_ref_image_size(raw: Any) -> str:
    return "max" if str(raw or "").strip().lower() == "max" else "match"


def _prune_helper_prefix(workflow: dict, prefix: str, keep: set[str]) -> None:
    for nid in [k for k in workflow if str(k).startswith(prefix)]:
        if nid not in keep:
            workflow.pop(nid, None)
    # Drop dangling links that pointed at pruned helpers.
    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs") or {}
        for key, value in list(inputs.items()):
            ref = link_ref(value)
            if ref and str(ref[0]).startswith(prefix) and ref[0] not in keep:
                inputs.pop(key, None)


def set_last_frame(workflow: dict, image_key: str | None) -> dict:
    """Wire (or clear) the FLF2V last_frame on the I2V node. No new model."""
    wf = deepcopy(workflow)
    i2v = _first_id(wf, [I2V_NODE])
    if i2v is None:
        raise ValueError("Graph has no MiniMaxH3ImageToVideo node for last_frame")
    inputs = wf[i2v].setdefault("inputs", {})
    if not image_key:
        inputs.pop("last_frame", None)
        _prune_helper_prefix(wf, "h3_last_", set())
        return wf
    # Mirror the first-frame ImageScale geometry so both keyframes share the canvas.
    width = inputs.get("width")
    height = inputs.get("height")
    for _nid, node in nodes_of_type(wf, ["ImageScale"]):
        nin = node.get("inputs") or {}
        width = width if width is not None else nin.get("width")
        height = height if height is not None else nin.get("height")
        break
    wf[LAST_IMAGE_ID] = {
        "class_type": "LoadImage",
        "_meta": {"title": "H3 last frame"},
        "inputs": {"image": image_key},
    }
    wf[LAST_SCALE_ID] = {
        "class_type": "ImageScale",
        "_meta": {"title": "H3 last frame scale"},
        "inputs": {
            "image": [LAST_IMAGE_ID, 0],
            "upscale_method": "lanczos",
            "width": int(width) if width is not None else 832,
            "height": int(height) if height is not None else 480,
            "crop": "center",
        },
    }
    inputs["last_frame"] = [LAST_SCALE_ID, 0]
    return wf


def set_r2v_refs(
    workflow: dict,
    *,
    ref_images: list[str] | None = None,
    ref_videos: list[str] | None = None,
    ref_audios: list[str] | None = None,
    ref_image_size: str = "match",
    include_video_audio: bool = True,
) -> dict:
    """Wire R2V-lite refs onto the Ref2VA node. Reuses the FL2VA UNET.

    Dotted keys follow the ComfyUI API convention
    (``ref_images.ref_image_0`` ...). LoadVideo rides through
    GetVideoComponents so frames feed ``ref_videos`` and the soundtrack
    auto-pairs to ``ref_video_audios``.
    """
    wf = deepcopy(workflow)
    r2v = _first_id(wf, [R2V_NODE])
    if r2v is None:
        raise ValueError("Graph has no MiniMaxH3ReferenceToVideo node for refs")
    images = [str(x) for x in (ref_images or []) if str(x or "").strip()]
    videos = [str(x) for x in (ref_videos or []) if str(x or "").strip()]
    audios = [str(x) for x in (ref_audios or []) if str(x or "").strip()]
    if len(images) > MAX_REF_IMAGES:
        raise ValueError(f"At most {MAX_REF_IMAGES} ref_images (got {len(images)})")
    if len(videos) > MAX_REF_VIDEOS:
        raise ValueError(f"At most {MAX_REF_VIDEOS} ref_videos (got {len(videos)})")
    if len(audios) > MAX_REF_AUDIOS:
        raise ValueError(f"At most {MAX_REF_AUDIOS} ref_audios (got {len(audios)})")
    inputs = wf[r2v].setdefault("inputs", {})
    for key in list(inputs):
        if key.startswith(("ref_images.", "ref_videos.", "ref_video_audios.", "ref_audios.")):
            inputs.pop(key, None)
    inputs["ref_image_size"] = normalize_ref_image_size(ref_image_size)
    keep: set[str] = set()
    for idx, key in enumerate(images):
        nid = f"{REF_IMAGE_PREFIX}{idx}"
        wf[nid] = {
            "class_type": "LoadImage",
            "_meta": {"title": f"H3 ref image {idx + 1}"},
            "inputs": {"image": key},
        }
        inputs[f"ref_images.ref_image_{idx}"] = [nid, 0]
        keep.add(nid)
    for idx, key in enumerate(videos):
        vid = f"{REF_VIDEO_PREFIX}{idx}"
        vc = f"{REF_VC_PREFIX}{idx}"
        wf[vid] = {
            "class_type": "LoadVideo",
            "_meta": {"title": f"H3 ref video {idx + 1}"},
            "inputs": {"file": key},
        }
        wf[vc] = {
            "class_type": "GetVideoComponents",
            "_meta": {"title": f"H3 ref video components {idx + 1}"},
            "inputs": {"video": [vid, 0]},
        }
        inputs[f"ref_videos.ref_video_{idx}"] = [vc, 0]
        if include_video_audio:
            inputs[f"ref_video_audios.ref_video_audio_{idx}"] = [vc, 1]
        keep.update({vid, vc})
    for idx, key in enumerate(audios):
        nid = f"{REF_AUDIO_PREFIX}{idx}"
        wf[nid] = {
            "class_type": "LoadAudio",
            "_meta": {"title": f"H3 ref audio {idx + 1}"},
            "inputs": {"audio": key},
        }
        inputs[f"ref_audios.ref_audio_{idx}"] = [nid, 0]
        keep.add(nid)
    _prune_helper_prefix(wf, REF_IMAGE_PREFIX, keep)
    _prune_helper_prefix(wf, REF_VIDEO_PREFIX, keep)
    _prune_helper_prefix(wf, REF_VC_PREFIX, keep)
    _prune_helper_prefix(wf, REF_AUDIO_PREFIX, keep)
    return wf
