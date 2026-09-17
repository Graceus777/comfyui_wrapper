import pytest

from comfyui_wrapper.config import from_dict
from comfyui_wrapper.h3 import (
    apply_h3_graph,
    normalize_ref_image_size,
    set_last_frame,
    set_r2v_refs,
)
from comfyui_wrapper.workflow import apply_settings, detect_bindings, load_workflow


def _i2v(repo_root):
    return load_workflow(repo_root / "workflows" / "minimax_h3_i2v_loop.api.json")


def _r2v(repo_root):
    return load_workflow(repo_root / "workflows" / "minimax_h3_r2v_loop.api.json")


def test_r2v_workflow_is_fl2va_lite(repo_root):
    wf = _r2v(repo_root)
    r2v = wf["6"]
    assert r2v["class_type"] == "MiniMaxH3ReferenceToVideo"
    assert r2v["inputs"]["clip"] == ["2", 0]
    assert r2v["inputs"]["vae"] == ["3", 0]
    assert r2v["inputs"]["audio_vae"] == ["15", 0]
    assert r2v["inputs"]["ref_image_size"] == "match"
    assert "fl2va" in wf["1"]["inputs"]["unet_name"]
    assert "ref2va" not in wf["1"]["inputs"]["unet_name"]
    assert wf["7"]["inputs"] == {"model": ["16", 0], "conditioning": ["6", 0]}
    assert wf["11"]["inputs"]["latent_image"] == ["6", 1]
    b = detect_bindings(wf)
    assert b["positive"] == ("6", "prompt")
    assert b["ref_image_size"] == ("6", "ref_image_size")
    assert b["length"] == ("6", "length")


def test_set_r2v_refs_wires_dotted_keys(repo_root):
    wf = set_r2v_refs(
        _r2v(repo_root),
        ref_images=["wrapper/h3/refs/a.png", "wrapper/h3/refs/b.png"],
        ref_videos=["wrapper/rtx/clip.mp4"],
        ref_audios=["wrapper/h3/refs/voice.wav"],
        ref_image_size="max",
    )
    cond = wf["6"]["inputs"]
    assert cond["ref_image_size"] == "max"
    assert cond["ref_images.ref_image_0"] == ["h3_ref_img_0", 0]
    assert cond["ref_images.ref_image_1"] == ["h3_ref_img_1", 0]
    assert wf["h3_ref_img_0"] == {
        "class_type": "LoadImage",
        "_meta": {"title": "H3 ref image 1"},
        "inputs": {"image": "wrapper/h3/refs/a.png"},
    }
    assert cond["ref_videos.ref_video_0"] == ["h3_ref_vc_0", 0]
    assert cond["ref_video_audios.ref_video_audio_0"] == ["h3_ref_vc_0", 1]
    assert wf["h3_ref_vid_0"]["inputs"] == {"file": "wrapper/rtx/clip.mp4"}
    assert wf["h3_ref_vc_0"]["inputs"] == {"video": ["h3_ref_vid_0", 0]}
    assert cond["ref_audios.ref_audio_0"] == ["h3_ref_aud_0", 0]
    assert wf["h3_ref_aud_0"]["inputs"] == {"audio": "wrapper/h3/refs/voice.wav"}


def test_set_r2v_refs_clears_and_enforces_caps(repo_root):
    wf = set_r2v_refs(_r2v(repo_root), ref_images=["a.png"])
    assert "ref_images.ref_image_0" in wf["6"]["inputs"]
    cleared = set_r2v_refs(wf, ref_images=[], ref_videos=[], ref_audios=[])
    assert not any(k.startswith(("ref_images.", "ref_videos.", "ref_audios.")) for k in cleared["6"]["inputs"])
    assert "h3_ref_img_0" not in cleared
    with pytest.raises(ValueError):
        set_r2v_refs(_r2v(repo_root), ref_images=[f"{i}.png" for i in range(10)])
    with pytest.raises(ValueError):
        set_r2v_refs(_r2v(repo_root), ref_videos=[f"{i}.mp4" for i in range(4)])
    with pytest.raises(ValueError):
        set_r2v_refs(_r2v(repo_root), ref_audios=[f"{i}.wav" for i in range(4)])
    with pytest.raises(ValueError):
        set_r2v_refs(_i2v(repo_root), ref_images=["a.png"])


def test_set_last_frame_flf2v(repo_root):
    wf = set_last_frame(_i2v(repo_root), "wrapper/h3/end.png")
    assert wf["6"]["inputs"]["last_frame"] == ["h3_last_scale", 0]
    assert wf["h3_last_image"]["inputs"] == {"image": "wrapper/h3/end.png"}
    assert wf["h3_last_scale"]["inputs"]["image"] == ["h3_last_image", 0]
    assert wf["h3_last_scale"]["inputs"]["width"] == 832
    cleared = set_last_frame(wf, None)
    assert "last_frame" not in cleared["6"]["inputs"]
    assert "h3_last_image" not in cleared
    with pytest.raises(ValueError):
        set_last_frame(_r2v(repo_root), "wrapper/h3/end.png")


def test_broadcast_keeps_ref_helpers(repo_root):
    wf = set_r2v_refs(_r2v(repo_root), ref_images=["wrapper/h3/refs/a.png"])
    patched = apply_settings(wf, {"width": 640, "height": 800, "image": "first.png"})
    assert patched["6"]["inputs"]["width"] == 640
    assert patched["h3_ref_img_0"]["inputs"]["image"] == "wrapper/h3/refs/a.png"
    i2v = set_last_frame(_i2v(repo_root), "wrapper/h3/end.png")
    patched_i2v = apply_settings(i2v, {"image": "first.png", "width": 640, "height": 800})
    assert patched_i2v["4"]["inputs"]["image"] == "first.png"
    assert patched_i2v["h3_last_image"]["inputs"]["image"] == "wrapper/h3/end.png"


def test_h3_graph_recipe_applies_to_r2v(repo_root):
    wf = apply_h3_graph(
        _r2v(repo_root),
        recipe="acc",
        acc_file="MiniMax-H3-FL2VA-Acc-8Step.safetensors",
        unet="minimax_h3_fl2va_pruned_w4a8_mixed.safetensors",
    )
    assert wf["h3_acc"]["class_type"] == "MiniMaxH3PDDAccApply"
    assert wf["7"]["inputs"]["model"] == ["h3_acc", 0]
    assert wf["9"]["inputs"]["sampler_name"] == "euler"
    assert wf["10"]["inputs"]["steps"] == 8


def test_normalize_ref_image_size():
    assert normalize_ref_image_size("max") == "max"
    assert normalize_ref_image_size("MAX") == "max"
    assert normalize_ref_image_size("match") == "match"
    assert normalize_ref_image_size("anything-else") == "match"
    assert normalize_ref_image_size(None) == "match"


def test_r2v_config_defaults(tmp_path):
    cfg = from_dict({}, root=tmp_path)
    assert cfg.video.mode == "i2v"
    assert cfg.video.ref_image_size == "match"
    assert cfg.video.r2v_workflow_file == "minimax_h3_r2v_loop.api.json"
    assert cfg.r2v_workflow_path().name == "minimax_h3_r2v_loop.api.json"
