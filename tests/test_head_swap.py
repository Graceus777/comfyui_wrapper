import pytest

from comfyui_wrapper.config import from_dict
from comfyui_wrapper.h3 import (
    HEAD_SWAP_LORA,
    HEAD_SWAP_PROMPT,
    HEAD_SWAP_UNET,
    apply_h3_graph,
    apply_head_swap,
    head_swap_prompt,
    is_head_swap_mode,
)
from comfyui_wrapper.video import video_frame_size
from comfyui_wrapper.workflow import apply_settings, load_workflow


def _graph(repo_root):
    return load_workflow(repo_root / "workflows" / "minimax_h3_head_swap.api.json")


def test_head_swap_prompt_inserts_details_and_keeps_trigger():
    assert is_head_swap_mode("headswap")
    assert is_head_swap_mode("face-swap")
    assert not is_head_swap_mode("i2v")
    assert head_swap_prompt("") == HEAD_SWAP_PROMPT
    assert head_swap_prompt(None) == HEAD_SWAP_PROMPT
    filled = head_swap_prompt("silver hoop earring")
    assert filled.startswith("head_swap:")
    assert "Facial details of <Subject 1>:\nsilver hoop earring\n" in filled
    assert "<Picture 1>" in filled and "<Video 1>" in filled
    assert head_swap_prompt("head_swap: keep this") == "head_swap: keep this"


def test_head_swap_graph_loads_regular_unet_with_separate_lora_and_acc(repo_root):
    prompt = head_swap_prompt("oval face, dark eyes")
    wf = apply_settings(
        _graph(repo_root),
        {
            "positive": prompt,
            "image": "wrapper/h3/face.png",
            "width": 832,
            "height": 480,
            "length": 90,
            "filename_prefix": "h3_head_swap",
        },
    )
    wf = apply_head_swap(
        wf,
        face_image="wrapper/h3/face.png",
        source_video="wrapper/h3/body.mp4",
        length=90,
    )
    wf["turbo"] = {
        "class_type": "LoraLoaderModelOnly",
        "inputs": {
            "model": ["h3_head_lora", 0],
            "lora_name": "minimax_h3_turbo_3step.safetensors",
            "strength_model": 1.0,
        },
    }
    wf["16"]["inputs"]["model"] = ["turbo", 0]
    wf = apply_h3_graph(
        wf,
        recipe="acc",
        acc_file="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        unet="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    )
    assert "turbo" not in wf
    assert wf["h3_head_lora"]["class_type"] == "LoraLoaderModelOnly"
    assert wf["h3_head_lora"]["inputs"] == {
        "model": ["1", 0],
        "lora_name": HEAD_SWAP_LORA,
        "strength_model": 1.0,
    }
    assert wf["1"]["inputs"]["unet_name"] == HEAD_SWAP_UNET
    assert wf["16"]["inputs"]["model"] == ["h3_head_lora", 0]
    assert wf["h3_acc"]["inputs"]["model"] == ["16", 0]
    assert wf["7"]["inputs"]["model"] == ["h3_acc", 0]
    assert wf["7"]["inputs"]["conditioning"] == ["h3_guide", 0]
    assert wf["h3_guide"]["inputs"]["frame_idx"] == 0
    assert wf["h3_guide"]["inputs"]["image"] == ["h3_body_trim", 0]
    assert wf["h3_guide"]["inputs"]["latent"] == ["6", 1]
    assert wf["h3_body"]["inputs"]["file"] == "wrapper/h3/body.mp4"
    assert wf["h3_face"]["inputs"]["image"] == "wrapper/h3/face.png"
    assert wf["h3_body_trim"]["inputs"]["length"] == 90
    assert wf["6"]["inputs"]["ref_images.ref_image_0"] == ["h3_face", 0]
    assert wf["6"]["inputs"]["length"] == 90
    assert wf["6"]["inputs"]["prompt"].startswith("head_swap:")
    assert "oval face, dark eyes" in wf["6"]["inputs"]["prompt"]
    assert wf["11"]["inputs"]["latent_image"] == ["6", 1]
    assert wf["11"]["inputs"]["sigmas"] == ["h3_acc", 1]
    assert wf["9"]["inputs"]["sampler_name"] == "euler"
    assert wf["10"]["inputs"]["steps"] == 8


def test_head_swap_native_keeps_configured_lora_and_strength(repo_root):
    wf = apply_head_swap(
        _graph(repo_root),
        face_image="face.png",
        source_video="body.mp4",
        length=124,
        lora_name="custom_head.safetensors",
        lora_strength=0.6,
    )
    wf = apply_h3_graph(wf, recipe="native", unet="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors")
    assert wf["h3_head_lora"]["inputs"] == {
        "model": ["1", 0],
        "lora_name": "custom_head.safetensors",
        "strength_model": 0.6,
    }
    assert wf["1"]["inputs"]["unet_name"] == HEAD_SWAP_UNET
    assert wf["16"]["inputs"]["model"] == ["h3_head_lora", 0]
    assert wf["7"]["inputs"]["model"] == ["16", 0]
    assert "h3_acc" not in wf


def test_head_swap_requires_face_and_clip(repo_root):
    graph = _graph(repo_root)
    with pytest.raises(ValueError, match="face still"):
        apply_head_swap(graph, face_image="", source_video="body.mp4", length=124)
    with pytest.raises(ValueError, match="source clip"):
        apply_head_swap(graph, face_image="face.png", source_video="", length=124)


def test_head_swap_config_defaults(tmp_path):
    cfg = from_dict({}, root=tmp_path)
    assert cfg.video.head_swap_unet == HEAD_SWAP_UNET
    assert cfg.video.head_swap_lora == HEAD_SWAP_LORA
    assert cfg.video.head_swap_strength == 1.0
    assert cfg.head_swap_workflow_path().name == "minimax_h3_head_swap.api.json"
    custom = from_dict(
        {"video": {"head_swap_lora": "custom_head.safetensors", "head_swap_strength": 0.6}},
        root=tmp_path,
    )
    assert custom.video.head_swap_lora == "custom_head.safetensors"
    assert custom.video.head_swap_strength == 0.6


def test_video_frame_size_missing_file_is_none(tmp_path):
    assert video_frame_size(tmp_path / "missing.mp4") is None
