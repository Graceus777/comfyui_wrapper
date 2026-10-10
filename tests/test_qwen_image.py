from pathlib import Path

import pytest

from comfyui_wrapper.cli import build_parser
from comfyui_wrapper.config import from_dict
from comfyui_wrapper.qwen_image import (
    MAX_REFERENCES,
    align_qwen_resolution,
    align_qwen_size,
    collect_reference_paths,
    generate_qwen,
    normalize_qwen_mode,
    prepare_qwen_graph,
    resolve_qwen_asset,
)
from comfyui_wrapper.workflow import detect_bindings, load_workflow

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "qwen_image_2_1.api.json"


def _base():
    return load_workflow(WORKFLOW)


def _prepare(**overrides):
    settings = {
        "prompt": "a red teapot",
        "negative": "blurry",
        "seed": 7,
        "steps": 25,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "width": 1024,
        "height": 1024,
        "batch_size": 1,
        "unet": "qwen_image_2.1-Q6_K.gguf",
        "clip": "qwen3vl_8b_w4a8.safetensors",
        "vae": "qwen_image_2.1_vae_bf16.safetensors",
        "resolution": 1024,
        "filename_prefix": "qwen_image_2_1",
    }
    settings.update(overrides)
    return prepare_qwen_graph(_base(), **settings)


def test_bundled_qwen_workflow_bindings():
    wf = _base()
    bindings = detect_bindings(wf)
    assert bindings["positive"] == ("4", "prompt")
    assert bindings["negative"] == ("4", "negative_prompt")
    assert bindings["checkpoint"] == ("1", "unet_name")
    assert bindings["clip"] == ("2", "clip_name")
    assert bindings["vae"] == ("3", "vae_name")
    assert bindings["width"] == ("5", "width")
    assert bindings["seed"] == ("6", "seed")
    assert wf["2"]["inputs"]["type"] == "qwen_image"
    assert wf["1"]["class_type"] == "UnetLoaderGGUF"


def test_txt2img_keeps_the_q6_smoke_graph():
    original = _base()
    wf = prepare_qwen_graph(
        original,
        prompt="a red teapot",
        negative="blurry",
        seed=7,
        unet="qwen_image_2.1-Q6_K.gguf",
        clip="qwen3vl_8b_w4a8.safetensors",
        vae="qwen_image_2.1_vae_bf16.safetensors",
    )
    assert original["4"]["inputs"]["prompt"] == ""
    assert wf["4"]["inputs"]["prompt"] == "a red teapot"
    assert wf["4"]["inputs"]["negative_prompt"] == "blurry"
    assert wf["6"]["inputs"]["model"] == ["1", 0]
    assert wf["6"]["inputs"]["latent_image"] == ["5", 0]
    assert wf["6"]["inputs"]["positive"] == ["4", 0]
    assert wf["6"]["inputs"]["negative"] == ["4", 1]
    assert wf["6"]["inputs"]["cfg"] == 1.0
    assert wf["6"]["inputs"]["sampler_name"] == "euler"
    assert wf["6"]["inputs"]["scheduler"] == "simple"
    assert wf["6"]["inputs"]["denoise"] == 1.0
    assert "qwen_cache" not in wf
    assert "images.image_1" not in wf["4"]["inputs"]
    assert "vae" not in wf["4"]["inputs"]


def test_edit_wires_references_and_encoder_latent():
    wf = _prepare(
        mode="edit",
        image_keys=["person.png", "shirt.png"],
        match_reference=True,
        cache_device="cpu",
        cache_dtype="int8",
    )
    assert wf["qwen_ref_1"]["inputs"]["image"] == "person.png"
    assert wf["qwen_ref_2"]["inputs"]["image"] == "shirt.png"
    assert wf["4"]["inputs"]["images.image_1"] == ["qwen_ref_1", 0]
    assert wf["4"]["inputs"]["images.image_2"] == ["qwen_ref_2", 0]
    assert wf["4"]["inputs"]["vae"] == ["3", 0]
    assert wf["6"]["inputs"]["latent_image"] == ["4", 2]
    assert wf["qwen_cache"]["class_type"] == "QwenImage21Cache"
    assert wf["qwen_cache"]["inputs"]["device"] == "cpu"
    assert wf["qwen_cache"]["inputs"]["dtype"] == "int8"
    assert wf["6"]["inputs"]["model"] == ["qwen_cache", 0]
    assert wf["qwen_cache"]["inputs"]["model"] == ["1", 0]


def test_edit_custom_size_keeps_empty_latent():
    wf = _prepare(
        mode="edit",
        image_keys=["person.png"],
        match_reference=False,
        width=768,
        height=1280,
    )
    assert wf["6"]["inputs"]["latent_image"] == ["5", 0]
    assert wf["5"]["inputs"]["width"] == 768
    assert wf["5"]["inputs"]["height"] == 1280
    assert wf["4"]["inputs"]["images.image_1"] == ["qwen_ref_1", 0]
    assert "qwen_cache" in wf


def test_edit_batch_repeats_encoder_latent():
    wf = _prepare(mode="edit", image_keys=["person.png"], batch_size=3)
    assert wf["qwen_repeat"]["inputs"]["samples"] == ["4", 2]
    assert wf["qwen_repeat"]["inputs"]["amount"] == 3
    assert wf["6"]["inputs"]["latent_image"] == ["qwen_repeat", 0]


def test_safetensors_unet_uses_unetloader():
    wf = _prepare(unet="qwen_image_2.1_int8_convrot.safetensors")
    assert wf["1"]["class_type"] == "UNETLoader"
    assert wf["1"]["inputs"]["unet_name"] == "qwen_image_2.1_int8_convrot.safetensors"
    assert wf["1"]["inputs"]["weight_dtype"] == "default"


def test_txt2img_cache_override_inserts_node():
    wf = _prepare(cache_device="off", cache_dtype="int4")
    assert wf["qwen_cache"]["inputs"]["device"] == "off"
    assert wf["qwen_cache"]["inputs"]["dtype"] == "int4"
    assert wf["6"]["inputs"]["model"] == ["qwen_cache", 0]


def test_edit_requires_an_image_and_caps_references():
    with pytest.raises(ValueError, match="at least one"):
        _prepare(mode="edit", image_keys=[])
    with pytest.raises(ValueError, match="at most"):
        _prepare(mode="edit", image_keys=[f"{i}.png" for i in range(MAX_REFERENCES + 1)])


def test_size_and_mode_helpers():
    assert align_qwen_size(1000) == 992
    assert align_qwen_size(8) == 16
    assert align_qwen_resolution(0) == 0
    assert align_qwen_resolution(1000) == 992
    assert align_qwen_resolution(5000) == 4096
    assert normalize_qwen_mode("txt2img", ["a.png"]) == "txt2img"
    assert normalize_qwen_mode(None, ["a.png"]) == "edit"
    assert normalize_qwen_mode("image-edit") == "edit"
    assert collect_reference_paths("a.png", ["b.png", "", None]) == ["a.png", "b.png"]
    assert resolve_qwen_asset(
        ["other.safetensors", "folder/qwen_image_2.1-Q6_K.gguf"],
        "qwen_image_2.1-Q6_K.gguf",
    ).endswith("qwen_image_2.1-Q6_K.gguf")
    assert resolve_qwen_asset(["sdxl.safetensors"], "qwen_image_2.1-Q6_K.gguf") == "qwen_image_2.1-Q6_K.gguf"


def test_qwen_config_defaults(tmp_path: Path):
    cfg = from_dict({}, root=tmp_path)
    assert cfg.qwen.unet == "qwen_image_2.1-Q6_K.gguf"
    assert cfg.qwen.clip == "qwen3vl_8b_w4a8.safetensors"
    assert cfg.qwen.vae == "qwen_image_2.1_vae_bf16.safetensors"
    assert cfg.qwen.steps == 25
    assert cfg.qwen.cfg == 1.0
    assert cfg.qwen.sampler == "euler"
    assert cfg.qwen.resolution == 1024
    assert cfg.qwen.match_reference is True
    custom = from_dict({"qwen_image": {"resolution": 0, "cache_dtype": "int8"}}, root=tmp_path)
    assert custom.qwen.resolution == 0
    assert custom.qwen.cache_dtype == "int8"
    assert custom.qwen.unet == "qwen_image_2.1-Q6_K.gguf"


def test_generate_qwen_edit_rejects_missing_image_before_queue(tmp_path: Path):
    cfg = from_dict({}, root=tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        generate_qwen("change the shirt", mode="edit", cfg=cfg)


def test_parser_qwen_edit_flags():
    parser = build_parser()
    args = parser.parse_args([
        "qwen",
        "--image", "a.png",
        "--image", "b.png",
        "--prompt", "wear the shirt",
        "--mode", "edit",
        "--resolution", "0",
        "--no-match-reference",
        "--cache-device", "cpu",
        "--cache-dtype", "int8",
        "--unet", "qwen_image_2.1-Q6_K.gguf",
    ])
    assert args.cmd == "qwen"
    assert args.image == ["a.png", "b.png"]
    assert args.mode == "edit"
    assert args.resolution == 0
    assert args.match_reference is False
    assert args.cache_device == "cpu"
    assert args.cache_dtype == "int8"
    assert args.func.__name__ == "cmd_qwen"
