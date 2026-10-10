"""MiniMax H3 frame, reference and swap workflows with matched presets.

Acc is MiniMaxH3PDDAccApply (not Load LoRA). Distills do not stack: turbo /
Spectrum / sage / SLA / Sol are disabled or bypassed when Acc is on. Wrong
sampler, steps, shift, or Acc file are swapped to the legal recipe instead of
being queued.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
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
HEAD_SWAP_LORA = "minimax_h3_head_swap_v1.0_r32.safetensors"
HEAD_SWAP_STRENGTH = 1.0
HEAD_SWAP_UNET = "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
HEAD_LORA_ID = "h3_head_lora"
FACE_IMAGE_ID = "h3_face"
BODY_VIDEO_ID = "h3_body"
BODY_VC_ID = "h3_body_vc"
BODY_TRIM_ID = "h3_body_trim"
HEAD_GUIDE_ID = "h3_guide"
_HEAD_SWAP_MODES = frozenset({"headswap", "head_swap", "faceswap", "face_swap"})
SAM3_CHECKPOINT = "sam3.1_multiplex_fp16.safetensors"
CHAR_UNET = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
CHAR_LORA = "minimax_h3_turbo_v4_step600_pruned_comfyui.safetensors"
CHAR_LORA_ID = "h3_ref_lora"
CHAR_LORA_STRENGTH = 1.0
CHAR_SHIFT_VIDEO = 6.0
CHAR_STEPS = 8
CHAR_SAMPLER = "res_multistep"
CHAR_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
CHAR_IMAGE_ID = "h3_ref_char"
CHAR_SAM_ID = "h3_sam3"
CHAR_TEXT_ID = "h3_sam3_text"
CHAR_TRACK_ID = "h3_sam3_track"
CHAR_MASK_ID = "h3_sam3_mask"
CHAR_INVERT_ID = "h3_invert"
CHAR_TREAT_ID = "h3_treat"
_CHAR_SWAP_MODES = frozenset({"charswap", "char_swap", "character_swap", "characterswap"})
_INVERT_MARK = (
    "The altered colors in the selected source area identify where to regenerate; "
    "render natural colors there, not photographic-negative colors."
)
_FACE_DETAILS_MARK = "Facial details of <Subject 1>:\n"
DEFAULT_LATENT_UPSCALER = "minimax_h3_latent_upscaler_3d_fp16.safetensors"

DEFAULT_UNET = "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"


def h3_frame_count(frames: int) -> int:
    """Round up to H3's 17n+5 temporal grid."""
    frames = max(5, int(frames))
    return frames + (5 - frames) % 17
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
    shift_video: float = SHIFT_VIDEO


RECIPES: dict[str, H3Recipe] = {
    "native": H3Recipe(
        key="native",
        label="Quality (base model)",
        steps=20,
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
    "ref_native": H3Recipe("ref_native", "Quality (base Ref2VA)", 20, "res_multistep", "normal", False, None),
    "ref_turbo4": H3Recipe("ref_turbo4", "Fast (reference Turbo, 4 steps)", 4, "euler", "simple", False, None),
    "ref_turbo8": H3Recipe("ref_turbo8", "Fast (reference Turbo, 8 steps)", 8, "euler", "simple", False, None, 6.0),
}

REF_TURBO4 = "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"
REF_TURBO8 = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
VIDEO_MODES = [
    ("Animate a still / start & end frames", "i2v"),
    ("Reference appearance → new video", "r2v"),
    ("Reference appearance + start / end frames", "r2v_frames"),
    ("Character swap in a source video", "charswap"),
    ("Head swap in a source video", "headswap"),
    ("Text → video", "t2v"),
]


def normalize_video_mode(raw: Any) -> str:
    key = str(raw or "i2v").strip().lower().replace("-", "_")
    aliases = {"flf": "i2v", "flf2v": "i2v", "ref": "r2v", "reference": "r2v",
               "r2v_lite": "r2v", "r2vlite": "r2v", "ref_frames": "r2v_frames"}
    if key in _CHAR_SWAP_MODES:
        return "charswap"
    if key in _HEAD_SWAP_MODES:
        return "headswap"
    key = aliases.get(key, key)
    if key not in {value for _label, value in VIDEO_MODES}:
        raise ValueError(f"Unknown video workflow {raw!r}")
    return key


def video_prompt_default(defaults: Any, mode: str) -> str:
    mode = normalize_video_mode(mode)
    field = ("swap_prompt" if mode in {"charswap", "headswap"} else "reference_prompt"
             if mode in {"r2v", "r2v_frames"} else "text_prompt" if mode == "t2v" else "prompt")
    return str(getattr(defaults, field, "") or "")


def video_model_choices(names: Iterable[str], mode: str) -> list[str]:
    """Only offer the checkpoint family trained for the selected workflow."""
    trunk = "ref2va" if normalize_video_mode(mode) in {"r2v", "r2v_frames", "charswap"} else "fl2va"
    return [name for name in names if is_h3_unet(name) and h3_trunk(name) == trunk
            and not any(tag in name.lower() for tag in
                        ("fasth3", "headswap", "hybrid", "turbo", "fused", "dmd", "hyperflow", "lightx", "_acc_", "-acc-"))]


def reference_recipe_choices(loras: Iterable[str], defaults: Any = None) -> list[tuple[str, str]]:
    names = {Path(name.replace("\\", "/")).name.lower() for name in loras}
    result = [(RECIPES["ref_native"].label, "ref_native")]
    for key, field, fallback in (("ref_turbo4", "ref_turbo4_lora", REF_TURBO4),
                                 ("ref_turbo8", "ref_turbo8_lora", REF_TURBO8)):
        name = getattr(defaults, field, fallback) or fallback
        if Path(name.replace("\\", "/")).name.lower() in names:
            result.append((RECIPES[key].label, key))
    return result


def video_workflow_options(mode: str, defaults: Any, *, unets: Iterable[str] = (),
                           acc_files: Iterable[str] = (), loras: Iterable[str] = (),
                           recipe: str | None = None) -> dict[str, Any]:
    mode = normalize_video_mode(mode)
    reference = mode in {"r2v", "r2v_frames"}
    swap = mode in {"charswap", "headswap"}
    names = video_model_choices(unets, mode)
    preferred = (defaults.character_swap_unet if mode == "charswap" else defaults.head_swap_unet
                 if mode == "headswap" else defaults.r2v_unet if reference else defaults.unet)
    model = pick_model_name(names, preferred) or preferred
    if model not in names:
        names = [model, *names]
    if reference:
        choices = reference_recipe_choices(loras, defaults)
        wanted = recipe or defaults.ref_recipe
    elif mode == "charswap":
        choices, wanted = [("Character swap preset", "native")], "native"
    elif mode == "headswap":
        choices, wanted = [("Head swap preset", "acc")], "acc"
    else:
        matched = [n for n in acc_files if h3_trunk(n) == "fl2va"]
        choices = [(RECIPES["native"].label, "native")]
        if matched:
            choices += [(RECIPES["acc"].label, "acc")]
            if has_acc_4step_weight(matched):
                choices += [(RECIPES["acc4"].label, "acc4")]
        wanted = recipe or defaults.recipe
    selected = wanted if wanted in {key for _label, key in choices} else choices[0][1]
    rec = RECIPES[selected]
    steps = CHAR_STEPS if mode == "charswap" else defaults.ref_steps if selected == "ref_native" else rec.steps
    hints = {
        "i2v": "Animate the supplied scene. Start and end pictures anchor the composition and pose. Supply at least one frame.",
        "r2v": "Use a character or design in a new scene. Appearance pictures do not set the opening frame. Describe the scene and action below.",
        "r2v_frames": "Use appearance references and separately anchor the opening or ending scene. Supply references and at least one frame guide.",
        "charswap": "Replace the selected subject. Appearance pictures define the replacement; the source video supplies the pose, action and timing.",
        "headswap": "Replace the head using one face picture. The source video supplies the body, action and timing.",
        "t2v": "Describe a scene and action to generate a video without image inputs.",
    }
    return {"mode": mode, "reference": reference, "swap": swap, "models": names, "model": model,
            "recipes": choices, "recipe": selected, "steps": steps,
            "steps_visible": not swap and selected in {"native", "ref_native"},
            "acc": match_acc_file(model, acc_files, defaults.acc_file, nfe=rec.nfe) if rec.acc else "",
            "hint": hints[mode]}

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
    "refnative": "ref_native",
    "refturbo4": "ref_turbo4",
    "refturbo8": "ref_turbo8",
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
    text = (name or "").lower().replace("-", "_")
    return "qwen3vl_32b" in text


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


LATENT_SUFFIXES = {".latent", ".safetensors"}
CLIP_SUFFIXES = {".mp4", ".webm", ".mov", ".mkv"}
SAVE_LATENT_ID = "h3_save_latent"  # legacy single-node id (joint AV — crashes stock SaveLatent, pruned)
SAVE_SPLIT_ID = "h3_save_split"
SAVE_VIDEO_ID = "h3_save_latent_video"
SAVE_AUDIO_ID = "h3_save_latent_audio"


def sibling_latent_pair(path: str | Path) -> tuple[Path | None, Path | None]:
    """Given one dropped .latent, find its (video, audio) pair in the same folder.

    Stock ``SaveLatent`` cannot save the joint H3 AV latent (NestedTensor has
    no ``.contiguous()``), so :func:`ensure_save_latent` saves the split
    halves as ``<prefix>_video_*.latent`` + ``<prefix>_audio_*.latent``.
    Returns ``(video_path, audio_path)`` with Nones for the missing half.
    """
    src = Path(str(path or ""))
    parent = src.parent
    stem = src.stem
    suffix = src.suffix.lower()
    alt_suffixes = [suffix] + [s for s in (".latent", ".safetensors") if s != suffix]

    def _existing(stem_cand: str) -> Path | None:
        for suf in alt_suffixes:
            cand = parent / f"{stem_cand}{suf}"
            try:
                if cand.is_file():
                    return cand
            except OSError:
                continue
        return None

    if "_video" in stem:
        video = _existing(stem) or (src if src.suffix.lower() in LATENT_SUFFIXES else None)
        audio = _existing(stem.replace("_video", "_audio"))
        return video, audio
    if "_audio" in stem:
        audio = _existing(stem) or (src if src.suffix.lower() in LATENT_SUFFIXES else None)
        video = _existing(stem.replace("_audio", "_video"))
        return video, audio
    # Unmarked single file: assume it is the video half (joint AV latents
    # cannot be produced by stock SaveLatent, so there is nothing else it
    # could be that stock LoadLatent can read back usefully).
    single = _existing(stem) or (src if src.suffix.lower() in LATENT_SUFFIXES else None)
    return single, None


def resolve_dropped_latent_pair(paths: Iterable[str | Path]) -> tuple[Path | None, Path | None]:
    """Pick (video, audio) latents out of a multi-file drop (e.g. Gradio temp dir).

    ``sibling_latent_pair`` only looks next to one file, but the WebUI
    ``Upscale`` tab uploads to an isolated temp dir — dropping just
    ``*_video.latent`` leaves the ``*_audio.latent`` sibling behind, so the
    upscale silently renders without audio. Passing both halves explicitly
    fixes it; this helper finds them in the uploaded set.
    """
    cands: list[Path] = []
    for raw in paths or []:
        if raw is None:
            continue
        try:
            if isinstance(raw, (str, Path)):
                text = str(raw).strip()
            elif isinstance(raw, dict):
                text = ""
                for key in ("path", "name", "filename", "orig_name", "url"):
                    cand = raw.get(key)
                    if isinstance(cand, str) and cand.strip():
                        text = cand.strip()
                        break
                if not text:
                    continue
            else:
                cand = getattr(raw, "name", None)
                if isinstance(cand, (str, Path)) and str(cand).strip():
                    full = str(raw).strip()
                    name = str(cand).strip()
                    try:
                        if Path(full).is_file():
                            text = full
                        elif Path(name).is_file():
                            text = name
                        else:
                            text = full or name
                    except OSError:
                        text = name
                else:
                    text = str(raw).strip()
        except Exception:
            continue
        if not text:
            continue
        try:
            cand = Path(text)
        except Exception:
            continue
        try:
            if cand.is_file():
                cands.append(cand)
        except OSError:
            continue
    latents = [p for p in cands if p.suffix.lower() in LATENT_SUFFIXES]
    if not latents:
        # No latents at all — let the caller fall back to single-file handling
        # (e.g. an .mp4 for RTX VSR).
        if not cands:
            return None, None
        return sibling_latent_pair(cands[0])
    videos = [p for p in latents if "_video" in p.stem]
    audios = [p for p in latents if "_audio" in p.stem and "_video" not in p.stem]
    if videos and audios:
        video = videos[0]
        want = video.stem.replace("_video", "_audio")
        for cand in audios:
            if cand.stem == want:
                return video, cand
        return video, audios[0]
    if videos and not audios:
        video = videos[0]
        _video, sibling_audio = sibling_latent_pair(video)
        if sibling_audio is not None:
            return video, sibling_audio
        # Same temp dir but different naming? Fall back to any other latent.
        others = [p for p in latents if p != video]
        if others:
            return video, others[0]
        return video, None
    if audios and not videos:
        audio = audios[0]
        sibling_video, _audio = sibling_latent_pair(audio)
        if sibling_video is not None:
            return sibling_video, audio
        others = [p for p in latents if p != audio]
        if others:
            return others[0], audio
        return None, audio
    # Unmarked latents: first is video; a second file (if dropped) is audio.
    video = latents[0]
    if len(latents) >= 2:
        return video, latents[1]
    _video, sibling_audio = sibling_latent_pair(video)
    return video, sibling_audio


def detect_upscale_kind(path: str | Path | None) -> str:
    """Return 'latent' for saved AV latents, 'clip' for rendered video."""
    suffix = Path(str(path or "")).suffix.lower()
    if suffix in LATENT_SUFFIXES:
        return "latent"
    if suffix in CLIP_SUFFIXES:
        return "clip"
    raise ValueError(
        f"Cannot detect upscale kind for {path!r}: "
        "drop a .latent (3D 2x) or a rendered clip .mp4/.webm/.mov/.mkv (RTX VSR)"
    )


def ensure_save_latent(workflow: dict, prefix: str = "h3_latent") -> dict:
    """Save the split video/audio halves so the Upscale tab can decode-only 2x later.

    Stock ``SaveLatent`` calls ``samples["samples"].contiguous()``, which
    crashes on the joint H3 AV latent (``NestedTensor`` has no
    ``.contiguous``). So split first via ``LTXVSeparateAVLatent`` and save
    each plain-tensor half: ``<prefix>_video_*.latent`` + ``<prefix>_audio_*.latent``.

    Idempotent: reuses ``h3_save_split`` / ``h3_save_latent_video`` /
    ``h3_save_latent_audio`` when present, and prunes the legacy single
    ``h3_save_latent`` joint node (it always crashes).
    """
    wf = deepcopy(workflow)
    wf.pop(SAVE_LATENT_ID, None)
    samplers = nodes_of_type(wf, ["SamplerCustomAdvanced"])
    if not samplers:
        raise ValueError("H3 graph has no SamplerCustomAdvanced to attach SaveLatent")
    sid = samplers[0][0]
    base = (prefix or "h3_latent").strip() or "h3_latent"
    wf[SAVE_SPLIT_ID] = {
        "class_type": "LTXVSeparateAVLatent",
        "_meta": {"title": "Split AV latent for save"},
        "inputs": {"av_latent": [sid, 0]},
    }
    wf[SAVE_VIDEO_ID] = {
        "class_type": "SaveLatent",
        "_meta": {"title": "H3 save video latent"},
        "inputs": {
            "samples": [SAVE_SPLIT_ID, 0],
            "filename_prefix": f"{base}_video",
        },
    }
    wf[SAVE_AUDIO_ID] = {
        "class_type": "SaveLatent",
        "_meta": {"title": "H3 save audio latent"},
        "inputs": {
            "samples": [SAVE_SPLIT_ID, 1],
            "filename_prefix": f"{base}_audio",
        },
    }
    return wf


def latent_upscale_graph(
    video_latent_key: str,
    prefix: str,
    *,
    audio_latent_key: str | None = None,
    vae: str | None = None,
    audio_vae: str | None = None,
    model_name: str | None = None,
    scale: float = 2.0,
    fps: float = 24.0,
    align: int = 32,
    enable_temporal_chunking: bool = True,
    force_unload: bool = True,
    device: str = "cuda",
    precision: str = "fp16",
) -> dict:
    """Decode-only graph from split latents: LoadLatent(s) → 3D 2x video → VAEDecode(+Audio) → SaveVideo.

    No sampler runs and no AV split is needed: :func:`ensure_save_latent`
    already saved the plain-tensor halves, so each ``LoadLatent`` output
    feeds its decoder directly. ``audio_latent_key`` is optional — without
    it the upscaled clip is silent.
    """
    if not str(video_latent_key or "").strip():
        raise ValueError("latent_upscale_graph needs a video latent input key")
    video_key = str(video_latent_key)
    audio_key = str(audio_latent_key or "").strip() or None
    upscaler_inputs: dict[str, Any] = {
        "latent": ["1", 0],
        "model_name": model_name or DEFAULT_LATENT_UPSCALER,
        "mode": "scale by multiplier",
        "mode.scale": float(scale),
        "align": int(align),
        "enable_temporal_chunking": bool(enable_temporal_chunking),
        "force_unload": bool(force_unload),
        "device": device,
        "precision": precision,
    }
    if audio_key:
        return {
            "1": {
                "class_type": "LoadLatent",
                "_meta": {"title": "Load video latent"},
                "inputs": {"latent": video_key},
            },
            "2": {
                "class_type": "LoadLatent",
                "_meta": {"title": "Load audio latent"},
                "inputs": {"latent": audio_key},
            },
            "3": {
                "class_type": "MinimaxH3LatentUpscaler3D",
                "_meta": {"title": "H3 3D latent upscaler"},
                "inputs": upscaler_inputs,
            },
            "4": {
                "class_type": "VAELoader",
                "_meta": {"title": "H3 Video VAE"},
                "inputs": {"vae_name": vae or DEFAULT_VAE},
            },
            "5": {
                "class_type": "VAELoader",
                "_meta": {"title": "H3 Audio VAE"},
                "inputs": {"vae_name": audio_vae or DEFAULT_AUDIO_VAE},
            },
            "6": {
                "class_type": "VAEDecode",
                "_meta": {"title": "VAE Decode video"},
                "inputs": {"samples": ["3", 0], "vae": ["4", 0]},
            },
            "7": {
                "class_type": "VAEDecodeAudio",
                "_meta": {"title": "VAE Decode Audio"},
                "inputs": {"samples": ["2", 0], "vae": ["5", 0]},
            },
            "8": {
                "class_type": "CreateVideo",
                "_meta": {"title": "Create upscaled video"},
                "inputs": {
                    "images": ["6", 0],
                    "fps": float(fps),
                    "audio": ["7", 0],
                    "bit_depth": 8,
                },
            },
            "9": {
                "class_type": "SaveVideo",
                "_meta": {"title": "Save upscaled video"},
                "inputs": {
                    "video": ["8", 0],
                    "filename_prefix": prefix,
                    "format": "mp4",
                    "codec": "h264",
                },
            },
        }
    return {
        "1": {
            "class_type": "LoadLatent",
            "_meta": {"title": "Load video latent"},
            "inputs": {"latent": video_key},
        },
        "3": {
            "class_type": "MinimaxH3LatentUpscaler3D",
            "_meta": {"title": "H3 3D latent upscaler"},
            "inputs": upscaler_inputs,
        },
        "4": {
            "class_type": "VAELoader",
            "_meta": {"title": "H3 Video VAE"},
            "inputs": {"vae_name": vae or DEFAULT_VAE},
        },
        "6": {
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode video"},
            "inputs": {"samples": ["3", 0], "vae": ["4", 0]},
        },
        "8": {
            "class_type": "CreateVideo",
            "_meta": {"title": "Create upscaled video"},
            "inputs": {
                "images": ["6", 0],
                "fps": float(fps),
                "bit_depth": 8,
            },
        },
        "9": {
            "class_type": "SaveVideo",
            "_meta": {"title": "Save upscaled video"},
            "inputs": {
                "video": ["8", 0],
                "filename_prefix": prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


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
    """Wire semantic references onto the Ref2VA node.

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


def set_reference_frame_guides(workflow: dict, *, first_frame: str | None = None,
                               last_frame: str | None = None) -> dict:
    """Keep timeline anchors separate from numbered appearance references."""
    wf = deepcopy(workflow)
    r2v = _first_id(wf, [R2V_NODE])
    if r2v is None:
        raise ValueError("Frame-guided references need MiniMaxH3ReferenceToVideo")
    inputs = wf[r2v]["inputs"]
    for nid in list(wf):
        if nid.startswith("h3_frame_"):
            rewire_consumers(wf, (nid, 0), (r2v, 0), skip_node_ids={nid})
            wf.pop(nid)
    positive = [r2v, 0]
    helpers = set()
    for role, key, index in (("start", first_frame, 0), ("end", last_frame, -1)):
        if not key:
            continue
        image_id, guide_id = f"h3_frame_{role}_image", f"h3_frame_{role}_guide"
        helpers.update({image_id, guide_id})
        wf[image_id] = {"class_type": "LoadImage", "inputs": {"image": key}}
        wf[guide_id] = {
            "class_type": GUIDE_NODE,
            "_meta": {"title": f"Anchor the {role} frame"},
            "inputs": {"positive": positive, "latent": [r2v, 1], "vae": inputs["vae"],
                       "image": [image_id, 0], "frame_idx": index},
        }
        positive = [guide_id, 0]
    rewire_consumers(wf, (r2v, 0), tuple(positive), skip_node_ids=helpers)
    return wf


def reference_video_prompt(prompt: str | None, *, image_count: int,
                           reference_description: str | None = None,
                           guided: bool = False) -> str:
    user = str(prompt or "").strip()
    if "subject_definitions:" in user and "summary:" in user:
        return user
    task = "keyframe completion + reference generation" if guided else "reference generation"
    instructions = f"[{task}] Generate the requested scene using the connected references."
    if image_count:
        instructions += (
            " Use the subject in <Picture 1> as the primary appearance reference, unless the scene instructions assign a different role. "
            "Appearance references define identity and design without forcing an opening pose. "
            "Multiple views on a character sheet depict one character; do not display the sheet or duplicate its character."
        )
    if guided:
        instructions += " Follow the connected timeline guides for the opening and ending composition and pose."
    if reference_description:
        instructions += "\nAppearance details: " + str(reference_description).strip()
    return instructions + "\n\n" + user


def apply_reference_recipe(workflow: dict, *, recipe: H3Recipe,
                           lora_name: str | None = None, steps: int | None = None) -> dict:
    wf = deepcopy(workflow)
    _strip_distill_loras(wf)
    wf = apply_h3_graph(wf, recipe="native",
                        steps=recipe.steps if recipe.key != "ref_native" else (steps or recipe.steps))
    _set_sampler(wf, recipe.sampler,
                 recipe.steps if recipe.key != "ref_native" else (steps or recipe.steps), recipe.scheduler)
    for _nid, node in nodes_of_type(wf, [SHIFT_NODE]):
        node["inputs"]["shift_video"] = recipe.shift_video
    if recipe.key == "ref_native":
        return wf
    if not lora_name:
        raise ValueError("The selected reference Turbo preset needs its matching LoRA")
    loader = _first_id(wf, ["UNETLoader"])
    if loader is None:
        raise ValueError("Reference workflow is missing its model loader")
    nid = "h3_reference_turbo"
    wf[nid] = {"class_type": "LoraLoaderModelOnly",
               "inputs": {"model": [loader, 0], "lora_name": lora_name, "strength_model": 1.0}}
    rewire_consumers(wf, (loader, 0), (nid, 0), skip_node_ids={nid})
    return wf


def clear_first_frame(workflow: dict) -> dict:
    """Disconnect frame initialization for end-only or text generation."""
    wf = deepcopy(workflow)
    candidates = []
    for _nid, node in nodes_of_type(wf, [I2V_NODE]):
        src = link_ref(node.setdefault("inputs", {}).pop("first_frame", None))
        if src:
            candidates.append(src[0])
    while candidates:
        nid = candidates.pop()
        node = wf.get(nid)
        if not node or node.get("class_type") not in {"ImageScale", "LoadImage"}:
            continue
        if any(link_ref(value) and link_ref(value)[0] == nid
               for other in wf.values() for value in (other.get("inputs") or {}).values()):
            continue
        candidates.extend(src[0] for value in (node.get("inputs") or {}).values()
                          if (src := link_ref(value)))
        wf.pop(nid)
    return wf


# Alissonerdx head-swap template. The face still is <Picture 1> / <Subject 1>.
# The source clip is the aligned MiniMaxH3AddGuide at frame 0, referred to as <Video 1>.
HEAD_SWAP_PROMPT = """head_swap:

subject_definitions:

<Subject 1> is the reference head identity from <Picture 1>, including the complete head, facial identity, facial structure, head shape, hairstyle, hairline, skin appearance, proportions, and all distinctive visual features.

Facial details of <Subject 1>:

<Subject 2> is the original person in <Video 1>, whose body, neck, clothing, pose, physical proportions below the head, actions, motion, timing, and interaction with the environment must remain unchanged.

<Video 1> is the source video for the target video edit, providing the original body performance, motion, camera movement, framing, timing, lighting, environment, scene composition, and temporal structure.

summary:

[video editing + reference generation]

The target video is an edited version of <Video 1>. Replace only the entire head of <Subject 2> with the reference head identity defined by <Subject 1> from <Picture 1>. Preserve the source video's body, clothing, performance, motion, camera behavior, environment, lighting, timing, and composition.

retention_analysis:

<Subject 1> (appears throughout the target video): attribute_transfer - the complete head identity from <Picture 1>, including facial identity, facial structure, hairstyle, head shape, proportions, skin appearance, and distinctive features, is transferred consistently onto the body of <Subject 2>.

<Subject 2> (appears throughout the target video): partially_preserved - the original body, neck position, clothing, pose, motion, gestures, actions, and physical performance are preserved, while only the original head is replaced.

<Video 1> (entire target timeline): partially_preserved - the original video timing, body motion, camera movement, framing, lighting, background, environment, scene composition, and temporal structure are preserved; only the subject's head appearance is modified.

detailed_description:

Preserve <Video 1> as the complete temporal and compositional foundation of the target video. <Subject 2> retains the exact original body, clothing, pose, body proportions, movement, gestures, actions, speed, timing, and interaction with the environment.

Replace only the entire original head of <Subject 2> with <Subject 1>, the reference head identity from <Picture 1>. Preserve the complete recognizable identity of <Subject 1>, including facial structure, skull and head shape, hairstyle, hairline, skin appearance, facial proportions, eyes, eyebrows, nose, mouth, jawline, ears, and distinctive identifying features.

The transferred head follows the original head motion and performance of <Subject 2> throughout <Video 1>, including head rotation, tilt, pitch, yaw, natural translation, facial orientation, gaze direction, and perspective changes. Maintain anatomically correct head scale and a seamless connection between the transferred head and the original neck.

Preserve the lighting conditions of <Video 1> on the transferred head. The head naturally responds to changes in illumination, shadows, highlights, color temperature, exposure, depth of field, motion blur, occlusion, camera angle, and perspective while continuously maintaining the identity established by <Picture 1>.

Maintain strong temporal consistency across every frame. The identity, facial geometry, hairstyle, head size, skin appearance, and distinguishing features of <Subject 1> remain stable during motion and viewpoint changes. Prevent identity drift, facial morphing, hairstyle changes, head-size fluctuations, geometry instability, texture crawling, temporal flickering, frame-to-frame inconsistency, or sudden changes in facial appearance.

Do not alter the body, clothing, hands, pose, gestures, action, background, objects, environment, camera trajectory, framing, lens behavior, scene timing, or composition inherited from <Video 1>. The final result should appear as though <Subject 1>'s head naturally belonged to <Subject 2>'s body during the original recorded performance.

overall_soundscape:

Preserve the original audiovisual timing and visible action structure of <Video 1>. No visual modifications are made outside the head replacement.

non_diegetic_music:
N/A
"""


def is_head_swap_mode(mode: str | None) -> bool:
    key = str(mode or "").strip().lower().replace("-", "_")
    return key in _HEAD_SWAP_MODES


def head_swap_prompt(extra: str | None = None) -> str:
    """Official head_swap: template. A short note fills the facial-details slot.

    A prompt that already starts with the trigger, or pastes the template, is
    sent through unchanged.
    """
    text = str(extra or "").strip()
    lowered = text.lower()
    if lowered.startswith("head_swap:") or "\nhead_swap:" in lowered:
        return text
    base = HEAD_SWAP_PROMPT
    at = base.find(_FACE_DETAILS_MARK)
    if at < 0 or not text:
        return base
    at += len(_FACE_DETAILS_MARK)
    return base[:at] + text.rstrip() + "\n" + base[at:]


def apply_head_swap(
    workflow: dict,
    *,
    face_image: str,
    source_video: str,
    length: int,
    unet_name: str | None = None,
    lora_name: str | None = None,
    lora_strength: float = HEAD_SWAP_STRENGTH,
) -> dict:
    """Load the regular w4a8 UNET and a separate head-swap LoRA.

    The rank-32 face edit is applied at ``lora_strength`` before sigma shift.
    ``apply_h3_graph`` then inserts PDD Acc on the sigma-shift output when
    that recipe is selected. The face still is ``<Picture 1>`` and the
    source clip is the aligned guide at frame 0 (``<Video 1>``).
    """
    face = str(face_image or "").strip()
    source = str(source_video or "").strip()
    if not face:
        raise ValueError("Head swap needs a face still for <Picture 1>")
    if not source:
        raise ValueError("Head swap needs a source clip for <Video 1>")
    wf = deepcopy(workflow)
    expected = {
        FACE_IMAGE_ID: "LoadImage",
        BODY_VIDEO_ID: "LoadVideo",
        BODY_VC_ID: "GetVideoComponents",
        BODY_TRIM_ID: "ImageFromBatch",
        HEAD_GUIDE_ID: GUIDE_NODE,
    }
    for nid, kind in expected.items():
        node = wf.get(nid)
        if not isinstance(node, dict) or node.get("class_type") != kind:
            raise ValueError(f"Head swap graph is missing {kind} node {nid}")
    r2v = _first_id(wf, [R2V_NODE])
    if r2v is None:
        raise ValueError("Head swap graph has no MiniMaxH3ReferenceToVideo node")
    loader = _first_id(wf, ["UNETLoader"])
    if loader is None:
        raise ValueError("Head swap graph has no UNETLoader")
    unet = str(unet_name or HEAD_SWAP_UNET).strip() or HEAD_SWAP_UNET
    lora = str(lora_name or HEAD_SWAP_LORA).strip() or HEAD_SWAP_LORA
    try:
        strength = float(lora_strength)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Head swap LoRA strength must be a number, got {lora_strength!r}") from exc
    if not isfinite(strength):
        raise ValueError("Head swap LoRA strength must be a finite number")
    wf[loader].setdefault("inputs", {})["unet_name"] = unet
    wf[HEAD_LORA_ID] = {
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "Head swap LoRA"},
        "inputs": {
            "model": [loader, 0],
            "lora_name": lora,
            "strength_model": strength,
        },
    }
    rewire_consumers(wf, (loader, 0), (HEAD_LORA_ID, 0), skip_node_ids={HEAD_LORA_ID})
    wf[FACE_IMAGE_ID].setdefault("inputs", {})["image"] = face
    wf[BODY_VIDEO_ID].setdefault("inputs", {})["file"] = source
    wf[BODY_TRIM_ID].setdefault("inputs", {})["length"] = max(1, int(length))
    guide_inputs = wf[HEAD_GUIDE_ID].setdefault("inputs", {})
    guide_inputs["frame_idx"] = 0
    guide_inputs["image"] = [BODY_TRIM_ID, 0]
    wf[r2v].setdefault("inputs", {})["ref_images.ref_image_0"] = [FACE_IMAGE_ID, 0]
    return wf


def is_character_swap_mode(mode: str | None) -> bool:
    key = str(mode or "").strip().lower().replace("-", "_")
    return key in _CHAR_SWAP_MODES


def character_swap_prompt(
    extra: str | None = None,
    target: str | None = None,
    reference_description: str | None = None,
    reference_image_count: int = 1,
) -> str:
    """Studio edit instructions, with character appearance separate from the scene."""
    user = str(extra or "").strip()
    subject = str(target or "").strip()
    pictures = " and ".join(f"<Picture {i}>" for i in range(1, max(1, reference_image_count) + 1))
    description = str(reference_description or "").strip()
    heading = (
        "[video editing] Edit <Video 1>. "
        f"Replace {subject or 'the selected subject'} with the character whose appearance comes from {pictures}, "
        "starting with the first output frame. Transfer the reference character's identity, face, hair, outfit, "
        "colors and design into the selected region. Preserve the source performance, timing, camera and "
        "unrelated scene details. The original subject's appearance must be replaced, including its inverted colors. "
        + _INVERT_MARK
    )
    lowered = user.lower()
    if lowered.startswith("[video editing]") or ("subject_definitions:" in lowered and "summary:" in lowered):
        return heading + "\n\n" + user + ("\n\nReference character:\n" + description if description else "")
    body = user or (
        "Replace the selected subject with the character shown in <Picture 1>. "
        "Keep the source performance."
    )
    return (
        heading + "\n\nsubject_definitions:\n"
        f"<Subject 1> is the replacement character whose appearance comes from {pictures}. "
        "Transfer its face, hair, outfit, colors and design; ignore the reference poses, backgrounds and framing. "
        "Multiple views on a character sheet show one character. " + description + "\n"
        f"<Subject 2> is the source subject to replace in <Video 1>, identified as: {subject or 'the selected subject'}.\n"
        "<Video 1> supplies the original performance, positions, poses, expressions, timing, camera, lighting and environment.\n\n"
        "summary:\n"
        "[video editing + reference generation] Replace <Subject 2> in <Video 1> with <Subject 1> from the first output frame.\n\n"
        "retention_analysis:\n"
        f"<Subject 1>: fully_preserved - identity and visible appearance follow {pictures}.\n"
        "<Video 1>: partially_preserved - retain the performance and scene; replace the selected subject's appearance.\n"
        "Preserve the source timing, actions, expressions, camera movement and all unrelated scene details. "
        + _INVERT_MARK + "\n\n"
        "detailed_description:\n"
        "[Shot 1] Follow the source subject's position, scale, pose and performance from the first frame. "
        "Do not begin in a T-pose, display a character sheet, or animate a transition from the reference pose.\n"
        + body + "\n\n"
        "overall_soundscape:\nUse the source video's sound.\n\n"
        "non_diegetic_music:\nDo not add music."
    )


def apply_character_swap(
    workflow: dict,
    *,
    character_image: str,
    source_video: str,
    length: int,
    sam_target: str,
    sam3_checkpoint: str | None = None,
    unet_name: str | None = None,
    lora_name: str | None = None,
    lora_strength: float = CHAR_LORA_STRENGTH,
    vae_name: str | None = None,
    shift_video: float = CHAR_SHIFT_VIDEO,
    steps: int = CHAR_STEPS,
    reference_images: list[str] | None = None,
) -> dict:
    """Point character swap at the Studio Ref2VA/V4 preset and appearance stills.

    Native SAM3 tracks the phrase. ``ImageInvert`` plus ``ImageCompositeMasked``
    flips colors only inside that mask. The still is ``<Picture 1>`` and the
    treated frames are ``<Video 1>``. Source audio is not fed in. This mode
    does not use the FL2VA UNET or PDD Acc: the V4 turbo LoRA sits on Ref2VA,
    sigma shift is 6, and the sampler is 8-step ``res_multistep``.
    """
    still = str(character_image or "").strip()
    source = str(source_video or "").strip()
    target = str(sam_target or "").strip()
    ckpt = SAM3_CHECKPOINT if sam3_checkpoint is None else str(sam3_checkpoint).strip()
    if not still:
        raise ValueError("Character swap needs a character still for <Picture 1>")
    if not source:
        raise ValueError("Character swap needs a source clip for <Video 1>")
    if not target:
        raise ValueError("Character swap needs a SAM3 target (who to replace in the source clip)")
    if not ckpt:
        raise ValueError(
            "Character swap needs a SAM3 checkpoint in models/checkpoints "
            f"(video.sam3_checkpoint). The usual file is {SAM3_CHECKPOINT}."
        )
    wf = deepcopy(workflow)
    expected = {
        CHAR_IMAGE_ID: "LoadImage",
        BODY_VIDEO_ID: "LoadVideo",
        BODY_VC_ID: "GetVideoComponents",
        BODY_TRIM_ID: "ImageFromBatch",
        CHAR_SAM_ID: "CheckpointLoaderSimple",
        CHAR_TEXT_ID: "CLIPTextEncode",
        CHAR_TRACK_ID: "SAM3_VideoTrack",
        CHAR_MASK_ID: "SAM3_TrackToMask",
        CHAR_INVERT_ID: "ImageInvert",
        CHAR_TREAT_ID: "ImageCompositeMasked",
    }
    for nid, kind in expected.items():
        node = wf.get(nid)
        if not isinstance(node, dict) or node.get("class_type") != kind:
            raise ValueError(f"Character swap graph is missing {kind} node {nid}")
    r2v = _first_id(wf, [R2V_NODE])
    if r2v is None:
        raise ValueError("Character swap graph has no MiniMaxH3ReferenceToVideo node")
    wf[CHAR_IMAGE_ID].setdefault("inputs", {})["image"] = still
    wf[BODY_VIDEO_ID].setdefault("inputs", {})["file"] = source
    wf[BODY_TRIM_ID].setdefault("inputs", {})["length"] = max(1, int(length))
    wf[CHAR_SAM_ID].setdefault("inputs", {})["ckpt_name"] = ckpt
    text_inputs = wf[CHAR_TEXT_ID].setdefault("inputs", {})
    text_inputs["text"] = target
    text_inputs["clip"] = [CHAR_SAM_ID, 1]
    track_inputs = wf[CHAR_TRACK_ID].setdefault("inputs", {})
    track_inputs["images"] = [BODY_TRIM_ID, 0]
    track_inputs["model"] = [CHAR_SAM_ID, 0]
    track_inputs["conditioning"] = [CHAR_TEXT_ID, 0]
    mask_inputs = wf[CHAR_MASK_ID].setdefault("inputs", {})
    mask_inputs["track_data"] = [CHAR_TRACK_ID, 0]
    mask_inputs["object_indices"] = ""
    wf[CHAR_INVERT_ID].setdefault("inputs", {})["image"] = [BODY_TRIM_ID, 0]
    treat = wf[CHAR_TREAT_ID].setdefault("inputs", {})
    treat["destination"] = [BODY_TRIM_ID, 0]
    treat["source"] = [CHAR_INVERT_ID, 0]
    treat["x"] = 0
    treat["y"] = 0
    treat["resize_source"] = False
    treat["mask"] = [CHAR_MASK_ID, 0]
    r2v_inputs = wf[r2v].setdefault("inputs", {})
    for key in list(r2v_inputs):
        if key.split(".", 1)[0] in {"ref_images", "ref_videos", "ref_video_audios", "ref_audios"}:
            r2v_inputs.pop(key)
    _prune_helper_prefix(wf, REF_IMAGE_PREFIX, set())
    r2v_inputs["ref_images.ref_image_0"] = [CHAR_IMAGE_ID, 0]
    extras = list(reference_images or [])
    if len(extras) >= MAX_REF_IMAGES:
        raise ValueError(f"Character swap accepts at most {MAX_REF_IMAGES - 1} additional appearance images")
    for idx, key in enumerate(extras, 1):
        nid = f"{REF_IMAGE_PREFIX}{idx}"
        wf[nid] = {"class_type": "LoadImage", "inputs": {"image": key}}
        r2v_inputs[f"ref_images.ref_image_{idx}"] = [nid, 0]
    r2v_inputs["ref_videos.ref_video_0"] = [CHAR_TREAT_ID, 0]
    r2v_inputs["length"] = max(1, int(length))
    try:
        strength = float(lora_strength)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Character swap LoRA strength must be a number, got {lora_strength!r}") from exc
    if not isfinite(strength):
        raise ValueError("Character swap LoRA strength must be a finite number")
    unet_id = _first_id(wf, ["UNETLoader"])
    if unet_id is None:
        raise ValueError("Character swap graph has no UNETLoader")
    shift_id = _first_id(wf, [SHIFT_NODE])
    if shift_id is None:
        raise ValueError("Character swap graph has no MiniMaxH3SigmaShift")
    unet = str(unet_name or CHAR_UNET).strip() or CHAR_UNET
    lora = str(lora_name or CHAR_LORA).strip() or CHAR_LORA
    vae = str(vae_name or CHAR_VAE).strip() or CHAR_VAE
    wf[unet_id].setdefault("inputs", {})["unet_name"] = unet
    vae_ref = link_ref(r2v_inputs.get("vae"))
    if not vae_ref or wf.get(vae_ref[0], {}).get("class_type") != "VAELoader":
        raise ValueError("Character swap needs a video VAELoader connected to reference conditioning")
    wf[vae_ref[0]].setdefault("inputs", {})["vae_name"] = vae
    wf[CHAR_LORA_ID] = {
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": "Ref2V turbo"},
        "inputs": {
            "model": [unet_id, 0],
            "lora_name": lora,
            "strength_model": strength,
        },
    }
    shift_inputs = wf[shift_id].setdefault("inputs", {})
    shift_inputs["model"] = [CHAR_LORA_ID, 0]
    shift_inputs["shift_video"] = CHAR_SHIFT_VIDEO
    shift_inputs["shift_audio"] = SHIFT_AUDIO
    acc_id = _first_id(wf, [ACC_NODE])
    if acc_id:
        rewire_consumers(wf, (acc_id, 0), (shift_id, 0), skip_node_ids={acc_id})
        wf.pop(acc_id, None)
    guider_id = _first_id(wf, ["BasicGuider", "CFGGuider"])
    if guider_id is None:
        raise ValueError("Character swap graph has no guider for reference conditioning")
    wf[guider_id]["class_type"] = "BasicGuider"
    wf[guider_id]["inputs"] = {"model": [shift_id, 0], "conditioning": [r2v, 0]}
    scheduler_id = _first_id(wf, ["BasicScheduler"])
    if scheduler_id is None:
        raise ValueError("Character swap graph has no BasicScheduler")
    sched = wf[scheduler_id].setdefault("inputs", {})
    sched["model"] = [shift_id, 0]
    sched["steps"] = CHAR_STEPS
    sched["scheduler"] = "simple"
    sched["denoise"] = 1.0
    samplers = nodes_of_type(wf, ["SamplerCustomAdvanced"])
    if not samplers:
        raise ValueError("Character swap graph has no SamplerCustomAdvanced")
    for nid, sampler in samplers:
        sampler_inputs = sampler.setdefault("inputs", {})
        sampler_inputs["guider"] = [guider_id, 0]
        sampler_inputs["sigmas"] = [scheduler_id, 0]
        sampler_inputs["latent_image"] = [r2v, 1]
        # Studio decodes the model's denoised prediction, not the integrator output.
        rewire_consumers(wf, (nid, 0), (nid, 1), skip_node_ids={nid})
    for _nid, select in nodes_of_type(wf, ["KSamplerSelect"]):
        select.setdefault("inputs", {})["sampler_name"] = CHAR_SAMPLER
    for _nid, decode in nodes_of_type(wf, ["VAEDecode", "VAEDecodeTiled"]):
        decode["class_type"] = "VAEDecodeTiled"
        decode_inputs = decode.setdefault("inputs", {})
        decode_inputs["samples"] = [samplers[0][0], 1]
        decode_inputs["vae"] = [vae_ref[0], vae_ref[1]]
        decode_inputs["tile_size"] = 512
        decode_inputs["overlap"] = 64
        decode_inputs["temporal_size"] = 32
        decode_inputs["temporal_overlap"] = 8
    for _nid, decode in nodes_of_type(wf, ["VAEDecodeAudio"]):
        decode.setdefault("inputs", {})["samples"] = [samplers[0][0], 1]
    return wf


def finish_swap_video(workflow: dict, *, frames: int) -> dict:
    """Remove H3 padding and retain the selected source clip's soundtrack."""
    wf = deepcopy(workflow)
    if CHAR_IMAGE_ID in wf:
        # Save/upscale helpers are inserted after apply_character_swap; they
        # must use the same denoised prediction as the Character Swap decoder.
        for nid, _sampler in nodes_of_type(wf, ["SamplerCustomAdvanced"]):
            rewire_consumers(wf, (nid, 0), (nid, 1), skip_node_ids={nid})
    video_id = _first_id(wf, ["CreateVideo"])
    if video_id is None or BODY_VC_ID not in wf:
        raise ValueError("Swap graph needs CreateVideo and source video components")
    inputs = wf[video_id].setdefault("inputs", {})
    output_image = inputs["images"]
    if CHAR_IMAGE_ID in wf:
        decode_id = _first_id(wf, ["VAEDecode", "VAEDecodeTiled"])
        if decode_id is None:
            raise ValueError("Character swap graph has no video decoder")
        output_image = [decode_id, 0]
    wf["h3_output_trim"] = {
        "class_type": "ImageFromBatch",
        "_meta": {"title": "Return the selected source duration"},
        "inputs": {"image": output_image, "batch_index": 0, "length": int(frames)},
    }
    inputs["images"] = ["h3_output_trim", 0]
    inputs["audio"] = [BODY_VC_ID, 1]
    inputs["fps"] = 24.0
    return wf
