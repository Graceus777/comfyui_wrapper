import pytest

from comfyui_wrapper.config import from_dict
from comfyui_wrapper.h3 import (
    CHAR_LORA,
    CHAR_UNET,
    CHAR_VAE,
    SAM3_CHECKPOINT,
    _INVERT_MARK,
    apply_character_swap,
    apply_h3_graph,
    apply_h3_latent_upscale,
    character_swap_prompt,
    ensure_save_latent,
    finish_swap_video,
    is_character_swap_mode,
)
from comfyui_wrapper.workflow import apply_settings, detect_bindings, load_workflow


def _graph(repo_root):
    return load_workflow(repo_root / "workflows" / "minimax_h3_character_swap.api.json")


def test_character_swap_prompt_names_the_inverted_region():
    assert is_character_swap_mode("charswap")
    assert is_character_swap_mode("character-swap")
    assert not is_character_swap_mode("headswap")
    filled = character_swap_prompt("keep the red coat", "the person")
    assert filled.startswith("[video editing] Edit <Video 1>.")
    assert "[video editing + reference generation]" in filled
    assert _INVERT_MARK in filled
    assert "ignore the reference poses" in filled
    assert "identified as: the person." in filled
    assert "<Picture 1>" in filled and "<Video 1>" in filled
    assert "keep the red coat" in filled
    assert (
        "Replace the selected subject with the character shown in <Picture 1>. "
        "Keep the source performance."
    ) in character_swap_prompt("", "the man")
    custom = "[video editing] leave this alone\n\n" + _INVERT_MARK
    custom_filled = character_swap_prompt(custom, "the person")
    assert custom in custom_filled
    assert "Replace the person with the character whose appearance comes from <Picture 1>" in custom_filled
    assert "fully_preserved - identity and visible appearance follow <Picture 1>" in filled
    described = character_swap_prompt("Keep the source performance", "the man", "white feathers and dark eyes")
    assert "<Subject 1> is the replacement character whose appearance comes from <Picture 1>" in described
    assert "white feathers and dark eyes" in described
    assert "from the first output frame" in described


def test_character_swap_bindings_keep_h3_and_sam3_apart(repo_root):
    graph = _graph(repo_root)
    bindings = detect_bindings(graph)
    assert bindings["checkpoint"] == ("1", "unet_name")
    assert bindings["clip"] == ("2", "clip_name")
    assert bindings["positive"] == ("6", "prompt")
    assert "image" not in bindings
    prompt = character_swap_prompt("silver hair", "the woman")
    wf = apply_settings(
        graph,
        {
            "positive": prompt,
            "image": "wrapper/h3/should-not-clobber.png",
            "checkpoint": "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
            "width": 640,
            "height": 800,
            "length": 90,
            "filename_prefix": "h3_char_swap",
        },
    )
    assert wf["1"]["inputs"]["unet_name"] == "minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
    assert wf["h3_sam3"]["inputs"]["ckpt_name"] == SAM3_CHECKPOINT
    assert wf["h3_ref_char"]["inputs"]["image"] == "wrapper/h3/character.png"
    assert wf["h3_sam3_text"]["inputs"]["text"] == "the person"
    assert wf["6"]["inputs"]["prompt"] == prompt
    assert wf["6"]["inputs"]["width"] == 640
    assert wf["6"]["inputs"]["length"] == 90


def test_character_swap_graph_inverts_then_references(repo_root):
    wf = apply_settings(
        _graph(repo_root),
        {"positive": character_swap_prompt("keep the coat", "the man"), "length": 73},
    )
    wf = apply_h3_graph(
        wf,
        recipe="acc",
        acc_file="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        unet="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    )
    wf = apply_character_swap(
        wf,
        character_image="wrapper/h3/hero.png",
        source_video="wrapper/h3/scene.mp4",
        length=73,
        sam_target="the man",
        sam3_checkpoint="custom_sam3.safetensors",
    )
    assert wf["h3_ref_char"]["inputs"]["image"] == "wrapper/h3/hero.png"
    assert wf["h3_body"]["inputs"]["file"] == "wrapper/h3/scene.mp4"
    assert wf["h3_body_trim"]["inputs"]["length"] == 73
    assert wf["h3_sam3"]["inputs"]["ckpt_name"] == "custom_sam3.safetensors"
    assert wf["h3_sam3"]["class_type"] == "CheckpointLoaderSimple"
    assert wf["h3_sam3_text"]["inputs"]["text"] == "the man"
    assert wf["h3_sam3_text"]["inputs"]["clip"] == ["h3_sam3", 1]
    assert wf["h3_sam3_track"]["inputs"]["images"] == ["h3_body_trim", 0]
    assert wf["h3_sam3_track"]["inputs"]["model"] == ["h3_sam3", 0]
    assert wf["h3_sam3_track"]["inputs"]["conditioning"] == ["h3_sam3_text", 0]
    assert wf["h3_invert"]["inputs"]["image"] == ["h3_body_trim", 0]
    assert wf["h3_treat"]["inputs"]["destination"] == ["h3_body_trim", 0]
    assert wf["h3_treat"]["inputs"]["source"] == ["h3_invert", 0]
    assert wf["h3_treat"]["inputs"]["mask"] == ["h3_sam3_mask", 0]
    assert wf["6"]["inputs"]["ref_images.ref_image_0"] == ["h3_ref_char", 0]
    assert wf["6"]["inputs"]["ref_videos.ref_video_0"] == ["h3_treat", 0]
    assert not any(key.startswith("ref_video_audios.") for key in wf["6"]["inputs"])
    assert "the man" in wf["6"]["inputs"]["prompt"]
    assert _INVERT_MARK in wf["6"]["inputs"]["prompt"]
    assert "h3_acc" not in wf
    assert wf["h3_ref_lora"]["inputs"]["model"] == ["1", 0]
    assert wf["h3_ref_lora"]["inputs"]["lora_name"] == CHAR_LORA
    assert wf["h3_ref_lora"]["inputs"]["strength_model"] == 1.0
    assert wf["16"]["inputs"]["model"] == ["h3_ref_lora", 0]
    assert wf["16"]["inputs"]["shift_video"] == 6.0
    assert wf["7"]["inputs"]["model"] == ["16", 0]
    assert wf["7"]["inputs"]["conditioning"] == ["6", 0]
    assert wf["10"]["inputs"]["steps"] == 8
    assert wf["10"]["inputs"]["model"] == ["16", 0]
    assert wf["9"]["inputs"]["sampler_name"] == "res_multistep"
    assert wf["11"]["inputs"]["sigmas"] == ["10", 0]
    assert wf["11"]["inputs"]["latent_image"] == ["6", 1]
    assert wf["1"]["inputs"]["unet_name"] == CHAR_UNET
    assert wf["3"]["inputs"]["vae_name"] == CHAR_VAE
    assert wf["12"]["class_type"] == "VAEDecodeTiled"
    assert wf["12"]["inputs"]["samples"] == ["11", 1]
    assert wf["17"]["inputs"]["samples"] == ["11", 1]
    assert "ignore the reference poses" in wf["6"]["inputs"]["prompt"]
    finished = finish_swap_video(wf, frames=72)
    assert finished["h3_output_trim"]["inputs"]["image"] == ["12", 0]
    assert finished["h3_output_trim"]["inputs"]["length"] == 72
    assert finished["13"]["inputs"]["images"] == ["h3_output_trim", 0]
    assert finished["13"]["inputs"]["audio"] == ["h3_body_vc", 1]
    assert finished["13"]["inputs"]["fps"] == 24


def test_character_swap_repairs_conditioning_and_output_bypasses(repo_root):
    graph = _graph(repo_root)
    graph["7"]["class_type"] = "CFGGuider"
    graph["7"]["inputs"] = {
        "model": ["1", 0], "positive": ["h3_sam3_text", 0],
        "negative": ["h3_sam3_text", 0], "cfg": 7.0,
    }
    graph["11"]["inputs"]["guider"] = ["old_guider", 0]
    graph["11"]["inputs"]["latent_image"] = ["old_latent", 0]
    graph["6"]["inputs"]["ref_videos.ref_video_1"] = ["h3_body_trim", 0]
    graph["6"]["inputs"]["ref_audios.ref_audio_0"] = ["h3_body_vc", 1]
    graph["12"]["inputs"]["samples"] = ["old_latent", 0]
    graph["13"]["inputs"]["images"] = ["h3_treat", 0]
    wf = apply_character_swap(
        graph, character_image="hero.png", source_video="scene.mp4", length=124, sam_target="the person",
    )
    assert wf["7"]["class_type"] == "BasicGuider"
    assert wf["7"]["inputs"] == {"model": ["16", 0], "conditioning": ["6", 0]}
    assert wf["11"]["inputs"]["guider"] == ["7", 0]
    assert wf["11"]["inputs"]["latent_image"] == ["6", 1]
    assert wf["12"]["inputs"]["samples"] == ["11", 1]
    assert "ref_videos.ref_video_1" not in wf["6"]["inputs"]
    assert "ref_audios.ref_audio_0" not in wf["6"]["inputs"]
    assert finish_swap_video(wf, frames=120)["h3_output_trim"]["inputs"]["image"] == ["12", 0]


def test_character_swap_saved_and_upscaled_latents_use_denoised_output(repo_root):
    wf = apply_character_swap(
        _graph(repo_root), character_image="hero.png", source_video="scene.mp4", length=124, sam_target="the person",
    )
    wf = apply_h3_latent_upscale(wf)
    wf = ensure_save_latent(wf)
    wf = finish_swap_video(wf, frames=120)
    assert wf["h3_av_split"]["inputs"]["av_latent"] == ["11", 1]
    assert wf["h3_save_split"]["inputs"]["av_latent"] == ["11", 1]


def test_character_swap_requires_visual_vae_conditioning(repo_root):
    graph = _graph(repo_root)
    graph["6"]["inputs"].pop("vae")
    with pytest.raises(ValueError, match="video VAELoader connected to reference conditioning"):
        apply_character_swap(
            graph, character_image="hero.png", source_video="scene.mp4", length=124, sam_target="the person",
        )


def test_character_swap_keeps_ref2va_after_native_shift(repo_root):
    wf = apply_h3_graph(_graph(repo_root), recipe="native", steps=12)
    wf = apply_character_swap(
        wf,
        character_image="hero.png",
        source_video="scene.mp4",
        length=124,
        sam_target="the person",
    )
    assert wf["7"]["inputs"]["model"] == ["16", 0]
    assert wf["16"]["inputs"]["shift_video"] == 6.0
    assert wf["16"]["inputs"]["model"] == ["h3_ref_lora", 0]
    assert wf["10"]["inputs"]["steps"] == 8
    assert "h3_acc" not in wf
    assert wf["1"]["inputs"]["unet_name"] == CHAR_UNET
    assert wf["h3_sam3"]["inputs"]["ckpt_name"] == SAM3_CHECKPOINT


def test_character_swap_requires_still_clip_target_and_checkpoint(repo_root):
    graph = _graph(repo_root)
    with pytest.raises(ValueError, match="character still"):
        apply_character_swap(
            graph, character_image="", source_video="scene.mp4", length=124, sam_target="the person"
        )
    with pytest.raises(ValueError, match="source clip"):
        apply_character_swap(
            graph, character_image="hero.png", source_video="", length=124, sam_target="the person"
        )
    with pytest.raises(ValueError, match="SAM3 target"):
        apply_character_swap(
            graph, character_image="hero.png", source_video="scene.mp4", length=124, sam_target="  "
        )
    with pytest.raises(ValueError, match="SAM3 checkpoint"):
        apply_character_swap(
            graph,
            character_image="hero.png",
            source_video="scene.mp4",
            length=124,
            sam_target="the person",
            sam3_checkpoint="",
        )


def test_character_swap_config_defaults(tmp_path):
    cfg = from_dict({}, root=tmp_path)
    assert cfg.video.sam3_checkpoint == SAM3_CHECKPOINT
    assert cfg.video.sam3_target == ""
    assert cfg.video.character_swap_unet == CHAR_UNET
    assert cfg.video.character_swap_lora == CHAR_LORA
    assert cfg.video.character_swap_vae == CHAR_VAE
    assert cfg.video.character_swap_shift_video == 6.0
    assert cfg.video.character_swap_size_megapixels == 0.8
    assert cfg.video.swap_source_end == 3.0
    assert cfg.character_swap_workflow_path().name == "minimax_h3_character_swap.api.json"
    custom = from_dict(
        {"video": {"sam3_checkpoint": "other_sam3.safetensors", "sam3_target": "the man"}},
        root=tmp_path,
    )
    assert custom.video.sam3_checkpoint == "other_sam3.safetensors"
    assert custom.video.sam3_target == "the man"
