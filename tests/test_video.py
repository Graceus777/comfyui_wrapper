from pathlib import Path

from comfyui_wrapper.config import from_dict
from comfyui_wrapper.video import h3_size, target_dimensions
from comfyui_wrapper.workflow import apply_settings, detect_bindings, load_workflow, merge_bindings


def test_h3_bindings(repo_root):
    wf = load_workflow(repo_root / "workflows" / "minimax_h3_i2v_loop.api.json")
    b = detect_bindings(wf)
    assert b["positive"] == ("6", "prompt")
    assert b["width"] == ("6", "width")
    assert b["length"] == ("6", "length")
    assert b["seed"] == ("8", "noise_seed")
    assert b["steps"] == ("10", "steps")
    assert b["sampler"] == ("9", "sampler_name")
    assert b["checkpoint"] == ("1", "unet_name")
    assert b["clip"] == ("2", "clip_name")
    assert b["vae"] == ("3", "vae_name")
    assert b["audio_vae"] == ("15", "vae_name")
    assert b["image"] == ("4", "image")
    assert b["filename_prefix"] == ("14", "filename_prefix")
    assert wf["16"]["class_type"] == "MiniMaxH3SigmaShift"
    assert wf["17"]["class_type"] == "VAEDecodeAudio"
    assert wf["13"]["inputs"]["audio"] == ["17", 0]
    assert wf["7"]["inputs"]["model"] == ["16", 0]


def test_h3_apply_settings_broadcasts_size(repo_root):
    wf = load_workflow(repo_root / "workflows" / "minimax_h3_i2v_loop.api.json")
    wf = apply_settings(
        wf,
        {
            "positive": "snow drifts",
            "width": 640,
            "height": 640,
            "length": 124,
            "steps": 8,
            "seed": 7,
            "image": "panel.png",
            "checkpoint": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
        },
    )
    assert wf["15"]["inputs"]["vae_name"] == "minimax_h3_audio_vae_fp32.safetensors"
    assert wf["6"]["inputs"]["prompt"] == "snow drifts"
    assert wf["6"]["inputs"]["width"] == 640
    assert wf["6"]["inputs"]["height"] == 640
    assert wf["5"]["inputs"]["width"] == 640
    assert wf["5"]["inputs"]["height"] == 640
    assert wf["6"]["inputs"]["length"] == 124
    assert wf["4"]["inputs"]["image"] == "panel.png"
    assert wf["10"]["inputs"]["steps"] == 8
    assert wf["8"]["inputs"]["noise_seed"] == 7


def test_krea2_and_flux_bindings(repo_root):
    krea = detect_bindings(load_workflow(repo_root / "workflows" / "krea2_txt2img.api.json"))
    assert krea["checkpoint"] == ("1", "unet_name")
    assert krea["clip"] == ("2", "clip_name")
    assert krea["vae"] == ("3", "vae_name")
    flux = detect_bindings(load_workflow(repo_root / "workflows" / "flux_gguf_txt2img.api.json"))
    assert flux["checkpoint"] == ("1", "unet_name")
    assert flux["clip"] == ("2", "clip_name1")
    assert flux["clip2"] == ("2", "clip_name2")
    assert flux["vae"] == ("3", "vae_name")


def test_anima_bindings(repo_root):
    anima = detect_bindings(load_workflow(repo_root / "workflows" / "anima_txt2img.api.json"))
    assert anima["checkpoint"] == ("1", "unet_name")
    assert anima["clip"] == ("2", "clip_name")
    assert anima["vae"] == ("3", "vae_name")
    assert anima["positive"] == ("4", "text")
    assert anima["negative"] == ("5", "text")
    assert anima["seed"] == ("7", "seed")
    assert anima["width"] == ("6", "width")
    wf = load_workflow(repo_root / "workflows" / "anima_txt2img.api.json")
    assert wf["2"]["inputs"]["type"] == "stable_diffusion"
    assert wf["2"]["inputs"]["clip_name"] == "qwen_3_06b_base.safetensors"
    assert wf["3"]["inputs"]["vae_name"] == "qwen_image_vae.safetensors"


def test_h3_size_preserves_aspect_on_32_grid():
    assert h3_size(1024, 1280) == (640, 800)
    assert h3_size(1024, 1280, megapixels=0.74) == (768, 960)
    assert h3_size(1920, 1080) == (960, 544)
    assert h3_size(1080, 1920) == (544, 960)
    assert h3_size(1024, 1024) == (736, 736)
    assert h3_size(833, 481, fit=False) == (832, 480)


def test_target_dimensions(tmp_path: Path):
    from PIL import Image

    portrait = tmp_path / "portrait.png"
    Image.new("RGB", (1024, 1280), "gray").save(portrait)
    assert target_dimensions(portrait) == (640, 800)
    wide = tmp_path / "wide.png"
    Image.new("RGB", (1920, 1080), "gray").save(wide)
    assert target_dimensions(wide) == (960, 544)
    tall = tmp_path / "tall.png"
    Image.new("RGB", (1080, 1920), "gray").save(tall)
    assert target_dimensions(tall) == (544, 960)


def test_h3_apply_settings_syncs_imagescale_and_latent(repo_root):
    wf = load_workflow(repo_root / "workflows" / "minimax_h3_i2v_loop.api.json")
    wf = apply_settings(wf, {"width": 640, "height": 800})
    assert wf["5"]["class_type"] == "ImageScale"
    assert wf["5"]["inputs"]["width"] == 640
    assert wf["5"]["inputs"]["height"] == 800
    assert wf["6"]["class_type"] == "MiniMaxH3ImageToVideo"
    assert wf["6"]["inputs"]["width"] == 640
    assert wf["6"]["inputs"]["height"] == 800


def test_video_config_defaults(tmp_path: Path):
    cfg = from_dict({}, root=tmp_path)
    assert cfg.video.frames == 124
    assert cfg.video.unet.startswith("minimax_h3")
    assert cfg.video.loop is True
    assert cfg.video.audio_vae.endswith("audio_vae_fp32.safetensors")
    assert cfg.video.recipe == "acc"
    assert cfg.video.steps == 8
    assert cfg.video.sampler == "euler"
    assert cfg.video.acc_file.startswith("MiniMax-H3-FL2VA-Acc")
    assert cfg.video.shift_video == 12.0
    assert cfg.video.shift_audio == 3.0
    assert cfg.video.latent_upscale is False
    assert cfg.video.rtx_vsr is False
    assert cfg.video.size_megapixels == 0.52
