"""Fixed, independent recipes for the selected October 2026 workflows.

These graphs deliberately bypass the generic workflow/settings injector. Each
panel owns its model family, conditioning, adapter and sampler connections.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

Graph = dict[str, dict[str, Any]]

QWEN_MODEL = "qwen_image_2.1-Q6_K.gguf"
QWEN_CLIP = "qwen3vl_8b_w4a8.safetensors"
QWEN_VAE = "qwen_image_2.1_vae_bf16.safetensors"
QWEN_CONSISTENCY = "qwen-image-2.1-consistency.safetensors"
QWEN_OUTFIT = "qwen-image-2.1-outfit-swap.safetensors"
QWEN_OUTPAINT = "qwen-image-2.1-outpaint-v2.safetensors"
H3_FL2VA = "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
H3_REF2VA = "minimax_h3_ref2va_pruned_int8_convrot.safetensors"
H3_CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
H3_VAE = "minimax_h3_video_vae_fp16.safetensors"
H3_AUDIO_VAE = "minimax_h3_audio_vae_fp32.safetensors"
H3_SWAP = "h3_character_swap_pro4500_1000.safetensors"
H3_SWAP_TURBO = "minimax_h3_ref2v_turbo_8step_v1.0_768p_comfyui_bf16.safetensors"
H3_UPSCALE = "minimax_h3_latent_upscaler_3d_fp16.safetensors"
H3_UPSCALE_LORA = "h3upscale_v2_x2.safetensors"
WAN_HIGH = "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
WAN_LOW = "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
WAN_CLIP = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
WAN_VAE = "wan_2.1_vae.safetensors"
WAN_ANISORA = "Wan2_2_I2V_AniSora_3_2_HIGH_rank_64_fp16.safetensors"
WAN_LIGHT = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

OUTPAINT_PROMPT = (
    "Outpaint the image: replace the solid gray areas with a seamless continuation "
    "of the scene, keeping the existing picture unchanged."
)
UPSCALE_PROMPT = (
    "h3upscale, increase resolution and restore fine details while preserving "
    "the original content and visual style."
)


def node(kind: str, title: str = "", **inputs: Any) -> dict:
    return {"class_type": kind, "inputs": inputs,
            "_meta": {"title": title or kind}}


def prompt_required(prompt: str) -> str:
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("Describe the edit, scene or action before generating.")
    return text


def grid_size(width: int, height: int, *, max_pixels: int) -> tuple[int, int]:
    width, height = int(width), int(height)
    if min(width, height) < 32 or width % 32 or height % 32:
        raise ValueError("Output dimensions must be positive multiples of 32.")
    if width * height > max_pixels:
        raise ValueError(f"This workflow's size limit is {max_pixels / 1e6:g} MP.")
    return width, height


def _qwen(image_keys: Sequence[str], prompt: str, seed: int, *,
          adapter: str = "", strength: float = 1.0, phrases: bool = False,
          steps: int = 25) -> Graph:
    if not 1 <= len(image_keys) <= 3 or not all(image_keys):
        raise ValueError("Supply the source image and at most two reference images.")
    wf: Graph = {
        "model": node("UnetLoaderGGUF", "Qwen Image 2.1 Q6", unet_name=QWEN_MODEL),
        "clip": node("CLIPLoader", clip_name=QWEN_CLIP, type="qwen_image", device="default"),
        "vae": node("VAELoader", vae_name=QWEN_VAE),
        "encode": node("QwenImage21ConditioningPhraseEncoder" if phrases else "TextEncodeQwenImage21",
                       "Encode edit and reference canvas", clip=["clip", 0], vae=["vae", 0],
                       prompt=prompt_required(prompt), negative_prompt="", resolution=0),
        "cache": node("QwenImage21Cache", "Lossless reference cache in RAM",
                      model=["adapter" if adapter else "model", 0], device="cpu", dtype="default"),
        "sample": node("KSampler", model=["cache", 0], positive=["encode", 0],
                       negative=["encode", 1], latent_image=["encode", 2], seed=int(seed),
                       steps=int(steps), cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0),
        "decode": node("VAEDecode", samples=["sample", 0], vae=["vae", 0]),
        "rgb": node("SplitImageWithAlpha", image=["decode", 0]),
        "save": node("SaveImage", images=["rgb", 0], filename_prefix="containers/qwen_edit"),
    }
    for index, key in enumerate(image_keys, 1):
        wf[f"image_{index}"] = node("LoadImage", f"Reference {index}", image=key)
        wf["encode"]["inputs"][f"images.image_{index}"] = [f"image_{index}", 0]
    if adapter:
        wf["adapter"] = node("LoraLoaderModelOnly", model=["model", 0],
                             lora_name=adapter, strength_model=float(strength))
    return wf


def build_qwen_controlled(image_keys: Sequence[str], prompt: str, seed: int = 0, *,
                          strengths: Sequence[float] = (), mask_key: str = "",
                          mask_index: int = 1, mask_strength: float = 1.0,
                          isolate_mask: bool = False) -> Graph:
    wf = _qwen(image_keys, prompt, seed, phrases=True)
    if strengths and len(strengths) != len(image_keys):
        raise ValueError("Each supplied reference needs its own strength.")
    model = ["model", 0]
    for index, value in enumerate(strengths or [1.0] * len(image_keys), 1):
        if not math.isfinite(value) or not 0 <= value <= 4:
            raise ValueError("Reference strength must be between 0 and 4.")
        key = f"strength_{index}"
        wf[key] = node("QwenImage21ReferenceStrength", model=model,
                       reference_index=index, strength=float(value))
        model = [key, 0]
    if mask_key:
        if not 1 <= int(mask_index) <= len(image_keys):
            raise ValueError("Choose a supplied image for the reference mask.")
        wf["mask_image"] = node("LoadImage", image=mask_key)
        wf["mask"] = node("ImageToMask", image=["mask_image", 0], channel="red")
        wf["mask_strength"] = node(
            "QwenImage21ReferenceStrengthMasked", model=model, mask=["mask", 0],
            reference_index=int(mask_index), strength=float(mask_strength),
            mask_resize_method="nearest-exact", mask_threshold=1.0,
            mask_behavior="zero_unmasked_tokens" if isolate_mask else "focus_only")
        model = ["mask_strength", 0]
    wf["cache"]["inputs"]["model"] = model
    wf["save"]["inputs"]["filename_prefix"] = "containers/qwen_reference_control"
    return wf


def build_qwen_consistent(image_key: str, prompt: str, seed: int = 0, *, strength: float = 1.0) -> Graph:
    wf = _qwen([image_key], prompt, seed, adapter=QWEN_CONSISTENCY, strength=strength)
    wf["save"]["inputs"]["filename_prefix"] = "containers/qwen_consistent_edit"
    return wf


def build_qwen_outfit(person_key: str, outfit_key: str, prompt: str, seed: int = 0, *,
                      strength: float = 1.0) -> Graph:
    wf = _qwen([person_key, outfit_key], prompt, seed, adapter=QWEN_OUTFIT, strength=strength)
    wf["save"]["inputs"]["filename_prefix"] = "containers/qwen_outfit_swap"
    return wf


def outpaint_margins(source_size: tuple[int, int], margins: Sequence[int]) -> tuple[int, int, int, int]:
    if len(margins) != 4:
        raise ValueError("Give left, top, right and bottom extension sizes.")
    left, top, right, bottom = (int(n) for n in margins)
    if min(left, top, right, bottom) < 0 or max(left, top, right, bottom) > 1024:
        raise ValueError("Extensions must be between 0 and 1024 pixels.")
    if not any((left, top, right, bottom)):
        raise ValueError("Extend at least one side of the picture.")
    width, height = source_size
    right += -(width + left + right) % 32
    bottom += -(height + top + bottom) % 32
    grid_size(width + left + right, height + top + bottom, max_pixels=2 * 1024**2)
    return left, top, right, bottom


def build_qwen_outpaint(image_key: str, source_size: tuple[int, int], *,
                       margins: Sequence[int] = (128, 128, 128, 128),
                       scene: str = "", seed: int = 0, strength: float = 1.0) -> Graph:
    left, top, right, bottom = outpaint_margins(source_size, margins)
    prompt = OUTPAINT_PROMPT + (" Scene: " + scene.strip() if scene.strip() else "")
    wf = _qwen([image_key], prompt, seed, adapter=QWEN_OUTPAINT, strength=strength)
    wf["pad"] = node("ImagePadForOutpaint", "Gray extension canvas",
                     image=["image_1", 0], left=left, top=top, right=right, bottom=bottom, feathering=0)
    wf["encode"]["inputs"]["images.image_1"] = ["pad", 0]
    # No latent noise mask: the publisher specifically warns about rectangle seams.
    wf["keep_mask"] = node("InvertMask", mask=["pad", 1])
    wf["restore"] = node("ImageCompositeMasked", "Restore exact source pixels",
                         destination=["rgb", 0], source=["pad", 0], x=0, y=0,
                         resize_source=False, mask=["keep_mask", 0])
    wf["save"]["inputs"].update(images=["restore", 0], filename_prefix="containers/qwen_outpaint")
    return wf


def _h3(base: str) -> Graph:
    return {
        "model": node("UNETLoader", unet_name=base, weight_dtype="default"),
        "clip": node("CLIPLoader", clip_name=H3_CLIP, type="minimax", device="default"),
        "vae": node("VAELoader", vae_name=H3_VAE),
        "audio_vae": node("VAELoader", vae_name=H3_AUDIO_VAE),
    }


def _memory(model: list) -> dict:
    return node("H3MemoryOptimization", "Bounded activations, native precision",
                model=model, fused_qkv="auto", mlp_memory="auto", chunk_rows=4096,
                preserve_precision=True, precision_mode="Preserve native", qkv_streaming_mode="Auto",
                embedding_memory_mode="Auto", kitchen_v_memory_mode="Standard")


def _h3_sample(wf: Graph, model: list, seed: int, *, steps: int = 20, prefix: str,
               shift_video: float = 12.0, sampler_name: str = "res_multistep") -> None:
    if not 1 <= int(steps) <= 40:
        raise ValueError("Choose 1–40 sampling steps.")
    wf.update({
        "memory": _memory(model),
        "shift": node("MiniMaxH3SigmaShift", model=["memory", 0], shift_video=shift_video, shift_audio=3.0),
        "guider": node("BasicGuider", model=["shift", 0], conditioning=["encode", 0]),
        "noise": node("RandomNoise", noise_seed=int(seed)),
        "sampler": node("KSamplerSelect", sampler_name=sampler_name),
        "schedule": node("BasicScheduler", model=["shift", 0], scheduler="simple", steps=int(steps), denoise=1.0),
        "sample": node("SamplerCustomAdvanced", noise=["noise", 0], guider=["guider", 0],
                       sampler=["sampler", 0], sigmas=["schedule", 0], latent_image=["encode", 1]),
        "decode": node("VAEDecodeTiled", samples=["sample", 1], vae=["vae", 0],
                       tile_size=512, overlap=64, temporal_size=32, temporal_overlap=8),
        "audio": node("VAEDecodeAudio", samples=["sample", 1], vae=["audio_vae", 0]),
        "video": node("CreateVideo", images=["decode", 0], audio=["audio", 0], fps=24.0, bit_depth=8),
        "save": node("SaveVideo", video=["video", 0], filename_prefix=prefix, format="mp4", codec="h264"),
    })


def _h3_dimensions(width: int, height: int, frames: int) -> tuple[int, int, int]:
    from comfyui_wrapper.h3 import h3_frame_count
    width, height = grid_size(width, height, max_pixels=900_000)
    if not 5 <= int(frames) <= 365:
        raise ValueError("This H3 panel supports clips up to about 15 seconds.")
    return width, height, h3_frame_count(int(frames))


def build_h3_memory_i2v(image_key: str, prompt: str, seed: int = 0, *,
                        width: int = 832, height: int = 480, frames: int = 124,
                        last_image_key: str = "", steps: int = 20) -> Graph:
    width, height, frames = _h3_dimensions(width, height, frames)
    if not image_key:
        raise ValueError("Supply the first frame.")
    wf = _h3(H3_FL2VA)
    wf["first"] = node("LoadImage", image=image_key)
    wf["first_fit"] = node("ImageScale", image=["first", 0], upscale_method="lanczos",
                           width=width, height=height, crop="center")
    wf["encode"] = node("MiniMaxH3ImageToVideo", clip=["clip", 0], vae=["vae", 0],
                        prompt=prompt_required(prompt), width=width, height=height,
                        length=frames, first_frame=["first_fit", 0])
    if last_image_key:
        wf["last"] = node("LoadImage", image=last_image_key)
        wf["last_fit"] = node("ImageScale", image=["last", 0], upscale_method="lanczos",
                              width=width, height=height, crop="center")
        wf["encode"]["inputs"]["last_frame"] = ["last_fit", 0]
    _h3_sample(wf, ["model", 0], seed, steps=steps, prefix="containers/h3_memory_i2v")
    return wf


def build_h3_lora_swap(image_key: str, video_key: str, target: str, seed: int = 0, *,
                       width: int = 832, height: int = 480, frames: int = 124,
                       delivered_frames: int | None = None, strength: float = 1.0,
                       instruction: str = "", turbo: bool = False,
                       steps: int | None = None) -> Graph:
    width, height, frames = _h3_dimensions(width, height, frames)
    if not image_key or not video_key:
        raise ValueError("Supply both the replacement character image and the source clip.")
    target = prompt_required(target)
    prompt = (
        f"Replace only {target} in <Video 1> with the character in <Picture 1>. "
        "Use the reference character's appearance and clothing. Keep the camera, scene "
        "and other people unchanged. Follow the source movement."
    ) + (" " + instruction.strip() if instruction.strip() else "")
    wf = _h3(H3_REF2VA)
    wf.update({
        "adapter": node("LoraLoaderModelOnly", "Dedicated character swap LoRA",
                        model=["model", 0], lora_name=H3_SWAP, strength_model=float(strength)),
        "character": node("LoadImage", image=image_key),
        "source": node("LoadVideo", file=video_key),
        "source_parts": node("GetVideoComponents", video=["source", 0]),
        "encode": node("MiniMaxH3ReferenceToVideo", clip=["clip", 0], vae=["vae", 0],
                       audio_vae=["audio_vae", 0], prompt=prompt, width=width, height=height,
                       length=frames, ref_image_size="match", **{
                           "ref_images.ref_image_0": ["character", 0],
                           "ref_videos.ref_video_0": ["source_parts", 0],
                       }),
    })
    model = ["adapter", 0]
    if turbo:
        wf["turbo"] = node("LoraLoaderModelOnly", "Ref2VA Turbo 8-step",
                           model=model, lora_name=H3_SWAP_TURBO, strength_model=1.0)
        model = ["turbo", 0]
    _h3_sample(wf, model, seed, steps=(8 if turbo else 20) if steps is None else steps,
               shift_video=6.0 if turbo else 12.0,
               sampler_name="euler" if turbo else "res_multistep",
               prefix="containers/h3_lora_character_swap")
    length = frames if delivered_frames is None else int(delivered_frames)
    if not 1 <= length <= frames:
        raise ValueError("Delivered frame count must fit the prepared source clip.")
    wf["trim"] = node("ImageFromBatch", image=["decode", 0], batch_index=0, length=length)
    wf["video"]["inputs"].update(images=["trim", 0], audio=["source_parts", 1])
    del wf["audio"]
    return wf


def build_h3_omnichar(char_key: str, prompt: str, reference_count: int, seed: int = 0, *,
                       width: int = 832, height: int = 480, frames: int = 124,
                       use_voice: bool = False, use_adapter: bool = False,
                       steps: int = 20) -> Graph:
    width, height, frames = _h3_dimensions(width, height, frames)
    if not char_key or not 1 <= int(reference_count) <= 4:
        raise ValueError("Choose a readable character containing 1–4 selected references.")
    wf = _h3(H3_REF2VA)
    wf["character"] = node("OmnicharLoadCharacter", char=char_key)
    wf["character_decode"] = node("OmnicharDecodeCharacter", char=["character", 0],
                                  style="token", prompt="\n" + prompt_required(prompt), arch="originals",
                                  max_references=4, size_from="first", fit="pad",
                                  voice="always" if use_voice else "never", audio_position=1)
    wf["encode"] = node("MiniMaxH3ReferenceToVideo", clip=["clip", 0], vae=["vae", 0],
                        audio_vae=["audio_vae", 0], prompt=["character_decode", 4],
                        width=width, height=height, length=frames, ref_image_size="match")
    for index in range(reference_count):
        key = f"character_ref_{index}"
        wf[key] = node("OmnicharCharacterReference", refs=["character_decode", 2], index=index)
        wf["encode"]["inputs"][f"ref_images.ref_image_{index}"] = [key, 0]
    if use_voice:
        wf["encode"]["inputs"]["ref_audios.ref_audio_0"] = ["character_decode", 5]
    model = ["model", 0]
    if use_adapter:
        wf["adapter"] = node("OmnicharApplyCharacterLoRA", model=model, char=["character", 0],
                             strength=-1.0, arch="minimax-h3", min_key_coverage=0.9)
        model = ["adapter", 0]
    _h3_sample(wf, model, seed, steps=steps, prefix="containers/h3_omnichar")
    return wf


def build_h3_guided_upscale(video_key: str, *, megapixels: float = 2.0,
                            delivered_frames: int | None = None) -> Graph:
    if not video_key:
        raise ValueError("Supply a finished clip.")
    if not math.isfinite(megapixels) or not 0.5 <= megapixels <= 2.0:
        raise ValueError("Choose an output size between 0.5 and 2 MP.")
    # The upsampler has its own forward/tiling implementation. Feed a plain model;
    # do not combine its re-draw with Turbo or the memory pack's forward patches.
    wf = _h3(H3_REF2VA)
    wf.update({
        "source": node("LoadVideo", file=video_key),
        "source_parts": node("GetVideoComponents", video=["source", 0]),
        "upscale": node("H3VideoUpsampler", model=["model", 0], clip=["clip", 0],
                        vae=["vae", 0], audio_vae=["audio_vae", 0],
                        images=["source_parts", 0], audio=["source_parts", 1],
                        prompt=UPSCALE_PROMPT, upscale_model=H3_UPSCALE, megapixels=float(megapixels),
                        denoise=0.1, lora=H3_UPSCALE_LORA, lora_strength=1.0,
                        tile_tokens=25000, sampler_name="euler"),
        "video": node("CreateVideo", images=["upscale", 0], audio=["source_parts", 1], fps=24.0, bit_depth=8),
        "save": node("SaveVideo", video=["video", 0], filename_prefix="containers/h3_guided_upscale",
                     format="mp4", codec="h264"),
    })
    if delivered_frames is not None:
        wf["trim"] = node("ImageFromBatch", image=["upscale", 0], batch_index=0, length=int(delivered_frames))
        wf["video"]["inputs"]["images"] = ["trim", 0]
    return wf


def wan_frame_count(seconds: float) -> int:
    if not math.isfinite(seconds) or not 0.5 <= seconds <= 4.0:
        raise ValueError("Each keyframe transition must last 0.5–4 seconds.")
    # Duration describes first-frame to last-frame distance. Wan requires 4n+1.
    return max(17, math.ceil(float(seconds) * 24 / 4) * 4 + 1)


def build_wan_keyframes(image_keys: Sequence[str], prompts: Sequence[str],
                         durations: Sequence[float], seed: int = 0, *,
                         width: int = 832, height: int = 480) -> Graph:
    width, height = grid_size(width, height, max_pixels=600_000)
    if not 2 <= len(image_keys) <= 5 or not all(image_keys):
        raise ValueError("Supply 2–5 keyframes in consecutive order.")
    pairs = len(image_keys) - 1
    if len(prompts) != pairs or len(durations) != pairs:
        raise ValueError("Each adjacent keyframe pair needs its own motion prompt and duration.")
    wf: Graph = {
        "high": node("UNETLoader", unet_name=WAN_HIGH, weight_dtype="default"),
        "low": node("UNETLoader", unet_name=WAN_LOW, weight_dtype="default"),
        "high_lora": node("LoraLoaderModelOnly", model=["high", 0], lora_name=WAN_ANISORA, strength_model=1.0),
        "low_lora": node("LoraLoaderModelOnly", model=["low", 0], lora_name=WAN_LIGHT, strength_model=1.0),
        "high_shift": node("ModelSamplingSD3", model=["high_lora", 0], shift=5.0),
        "low_shift": node("ModelSamplingSD3", model=["low_lora", 0], shift=5.0),
        "clip": node("CLIPLoader", clip_name=WAN_CLIP, type="wan", device="default"),
        "vae": node("VAELoader", vae_name=WAN_VAE),
        "negative": node("CLIPTextEncode", clip=["clip", 0], text="blurry, distorted, subtitles, watermark, static shot"),
    }
    for index, key in enumerate(image_keys):
        wf[f"keyframe_{index}"] = node("LoadImage", f"Keyframe {index + 1}", image=key)
    batches = []
    for index, (prompt, duration) in enumerate(zip(prompts, durations)):
        length = wan_frame_count(float(duration))
        p = f"pair_{index}"
        wf[p + "_text"] = node("CLIPTextEncode", clip=["clip", 0], text=prompt_required(prompt))
        wf[p + "_encode"] = node("WanFirstLastFrameToVideo", positive=[p + "_text", 0],
                                   negative=["negative", 0], vae=["vae", 0], width=width, height=height,
                                   length=length, batch_size=1, start_image=[f"keyframe_{index}", 0],
                                   end_image=[f"keyframe_{index + 1}", 0])
        common = dict(steps=4, cfg=1.0, sampler_name="euler", scheduler="simple",
                      positive=[p + "_encode", 0], negative=[p + "_encode", 1])
        wf[p + "_high"] = node("KSamplerAdvanced", model=["high_shift", 0],
                                 latent_image=[p + "_encode", 2], noise_seed=(int(seed) + index) % 2**64,
                                 add_noise="enable", start_at_step=0, end_at_step=2,
                                 return_with_leftover_noise="enable", **common)
        wf[p + "_low"] = node("KSamplerAdvanced", model=["low_shift", 0],
                                latent_image=[p + "_high", 0], noise_seed=0, add_noise="disable",
                                start_at_step=2, end_at_step=10000, return_with_leftover_noise="disable", **common)
        wf[p + "_decode"] = node("VAEDecodeTiled", samples=[p + "_low", 0], vae=["vae", 0],
                                   tile_size=512, overlap=64, temporal_size=32, temporal_overlap=8)
        # Shared keyframes occur only once at joins, so timing remains predictable.
        skip = 0 if index == 0 else 1
        wf[p + "_trim"] = node("ImageFromBatch", image=[p + "_decode", 0], batch_index=skip, length=length - skip)
        batches.append([p + "_trim", 0])
    if len(batches) == 1:
        images = batches[0]
    else:
        wf["join"] = node("BatchImagesNode", **{f"images.image{i}": ref for i, ref in enumerate(batches)})
        images = ["join", 0]
    wf["video"] = node("CreateVideo", images=images, fps=24.0, bit_depth=8)
    wf["save"] = node("SaveVideo", video=["video", 0], filename_prefix="containers/wan_keyframes",
                      format="mp4", codec="h264")
    return wf
