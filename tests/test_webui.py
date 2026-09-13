import pytest

from comfyui_wrapper.cli import build_parser
from comfyui_wrapper.config import from_dict
from comfyui_wrapper.webui import (
    VAE_FROM_CHECKPOINT,
    filter_choices,
    format_infotext,
    format_progress,
    insert_embedding_tag,
    insert_lora_tag,
    match_choice,
    parse_size_preset,
    parse_slot_text,
    prefer_first,
    sampler_choices,
    selected_vae,
    split_csv,
    vae_dropdown_choices,
)


def test_parser_ui_subcommand():
    parser = build_parser()
    args = parser.parse_args(["ui", "--port", "7861", "--listen", "--no-browser"])
    assert args.cmd == "ui"
    assert args.port == 7861
    assert args.listen is True
    assert args.no_browser is True


def test_format_infotext_a1111_shape():
    text = format_infotext(
        positive="1girl, looking at viewer",
        negative="low quality",
        steps=28,
        sampler="euler_ancestral",
        scheduler="normal",
        cfg=5.0,
        seed=42,
        width=1024,
        height=1280,
        checkpoint="wai.safetensors",
        loras=[{"name": "AsheIL-21", "weight": 0.8}],
    )
    assert text.startswith("1girl, looking at viewer")
    assert "Negative prompt: low quality" in text
    assert "Steps: 28" in text
    assert "Sampler: Euler a" in text
    assert "Seed: 42" in text
    assert "Size: 1024x1280" in text
    assert "Model: wai.safetensors" in text
    assert "LoRAs: AsheIL-21:0.8" in text


def test_format_progress_sampling():
    assert format_progress({"type": "progress", "data": {"value": 5, "max": 28}}) == "Sampling 5/28"
    assert format_progress({"type": "executing", "data": {"node": None}}) == "Finishing..."
    assert format_progress({"type": "poll", "data": {}}) == "Waiting for ComfyUI..."


def test_insert_lora_tag_adds_and_replaces():
    assert insert_lora_tag("", "AsheIL-21", 0.8) == "<lora:AsheIL-21:0.8>"
    assert insert_lora_tag("1girl", "AsheIL-21", 1) == "1girl, <lora:AsheIL-21:1>"
    updated = insert_lora_tag("1girl, <lora:AsheIL-21:0.5>", "AsheIL-21.safetensors", 0.9)
    assert "<lora:AsheIL-21.safetensors:0.9>" in updated
    assert "0.5" not in updated


def test_insert_embedding_tag():
    assert insert_embedding_tag("", "lazypos.safetensors") == "embedding:lazypos"
    assert insert_embedding_tag("1girl", "FastNegativeV2.pt", 1) == "1girl, embedding:FastNegativeV2"
    assert insert_embedding_tag("1girl", "lazyneg", 0.8) == "1girl, (embedding:lazyneg:0.8)"
    replaced = insert_embedding_tag("1girl, embedding:lazypos", "lazypos.safetensors", 1.2)
    assert "(embedding:lazypos:1.2)" in replaced
    assert replaced.count("embedding:lazypos") == 1


def test_selected_vae_from_checkpoint():
    assert selected_vae(None) is None
    assert selected_vae(VAE_FROM_CHECKPOINT) is None
    assert selected_vae("sdxl_vae.safetensors") == "sdxl_vae.safetensors"
    choices = vae_dropdown_choices(["ae.safetensors", "sdxl_vae.safetensors"])
    assert choices[0] == VAE_FROM_CHECKPOINT
    assert "sdxl_vae.safetensors" in choices


def test_parse_size_preset():
    assert parse_size_preset("1024 x 1280 (portrait)") == (1024, 1280)
    assert parse_size_preset("832x1216") == (832, 1216)
    assert parse_size_preset("nope") is None


def test_parse_slot_text():
    slots = parse_slot_text("pose=standing,kneeling\nview=from side,from behind")
    assert slots == {"pose": ["standing", "kneeling"], "view": ["from side", "from behind"]}
    assert parse_slot_text("") == {}
    assert parse_slot_text("pose=standing,kneeling; view=side")["view"] == ["side"]


def test_split_csv_and_filter():
    assert split_csv("a, b, ,c") == ["a", "b", "c"]
    assert split_csv("  ") is None
    assert filter_choices("ash", ["AsheIL-21.safetensors", "other.safetensors"]) == ["AsheIL-21.safetensors"]
    assert prefer_first(["b", "a"], "a") == ["a", "b"]
    assert prefer_first(["b"], "z") == ["z", "b"]


def test_match_choice_picks_h3_stack():
    unets = [
        "waiIllustriousSDXL_v150.safetensors",
        "krea2_turbo_fp8_scaled.safetensors",
        "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "flux1-dev-Q8_0.gguf",
    ]
    vaes = [
        "ae.safetensors",
        "qwen_image_vae.safetensors",
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    ]
    clips = [
        "qwen3vl_4b_fp8_scaled.safetensors",
        "clip_l.safetensors",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ]
    assert match_choice(unets, "minimax_h3_fl2va_pruned_int8_convrot.safetensors").startswith("minimax_h3_fl2va")
    assert match_choice(unets, "missing.safetensors", "minimax_h3_fl2va").startswith("minimax_h3")
    assert "video_vae" in match_choice(vaes, "minimax_h3_video_vae_fp16.safetensors")
    assert "audio_vae" in match_choice(vaes, "minimax_h3_audio_vae_fp32.safetensors")
    assert "32b_minimax" in match_choice(clips, "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")


def test_sampler_choices_use_a1111_labels():
    choices = sampler_choices(["euler_ancestral", "dpmpp_2m", "custom_sampler"])
    assert ("Euler a", "euler_ancestral") in choices
    assert ("DPM++ 2M", "dpmpp_2m") in choices
    assert ("custom_sampler", "custom_sampler") in choices


def test_build_ui_without_comfy(tmp_path):
    pytest.importorskip("gradio")
    from comfyui_wrapper.webui import build_ui

    cfg = from_dict({"generation": {"positive": "a cat", "steps": 12}}, root=tmp_path)
    demo = build_ui(cfg)
    assert demo is not None
