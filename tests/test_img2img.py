from pathlib import Path

import pytest

from comfyui_wrapper.config import apply_overrides, from_dict
from comfyui_wrapper.generate import build_workflow
from comfyui_wrapper.workflow import (
    apply_settings,
    clear_img2img,
    detect_bindings,
    inject_hires,
    inject_img2img,
)


def test_inject_img2img_sdxl(sdxl_wf):
    wf = inject_img2img(
        sdxl_wf, image="wrapper/img2img/init.png",
        denoise=0.6, width=1024, height=1280, batch_size=1,
    )
    assert wf["img2img_loader"]["class_type"] == "LoadImage"
    assert wf["img2img_loader"]["inputs"]["image"] == "wrapper/img2img/init.png"
    assert wf["img2img_scale"]["inputs"]["width"] == 1024
    assert wf["img2img_scale"]["inputs"]["height"] == 1280
    # SDXL VAE comes from the checkpoint loader.
    assert wf["img2img_encode"]["inputs"]["vae"] == ["4", 2]
    assert wf["3"]["inputs"]["latent_image"] == ["img2img_encode", 0]
    assert wf["3"]["inputs"]["denoise"] == 0.6
    # original untouched
    assert sdxl_wf["3"]["inputs"]["latent_image"] == ["5", 0]


def test_inject_img2img_uses_vae_loader_for_anima(repo_root):
    from comfyui_wrapper.workflow import load_workflow

    wf = load_workflow(repo_root / "workflows" / "anima_txt2img.api.json")
    out = inject_img2img(wf, image="wrapper/img2img/a.png", denoise=0.55, width=1024, height=1344)
    assert out["img2img_encode"]["inputs"]["vae"] == ["3", 0]
    assert out["7"]["inputs"]["latent_image"] == ["img2img_encode", 0]
    assert out["7"]["inputs"]["denoise"] == 0.55
    assert detect_bindings(out)["init_image"] == ("img2img_loader", "image")


def test_inject_img2img_batch_repeat(sdxl_wf):
    wf = inject_img2img(sdxl_wf, image="x.png", denoise=0.5, width=512, height=512, batch_size=3)
    assert wf["img2img_repeat"]["inputs"] == {"samples": ["img2img_encode", 0], "amount": 3}
    assert wf["3"]["inputs"]["latent_image"] == ["img2img_repeat", 0]
    # idempotent downgrade removes the repeat node
    again = inject_img2img(wf, image="y.png", denoise=0.7, width=512, height=512, batch_size=1)
    assert "img2img_repeat" not in again
    assert again["img2img_loader"]["inputs"]["image"] == "y.png"
    assert again["3"]["inputs"]["denoise"] == 0.7


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, float("nan")])
def test_inject_img2img_rejects_bad_denoise(sdxl_wf, bad):
    with pytest.raises(ValueError, match="denoise"):
        inject_img2img(sdxl_wf, image="x.png", denoise=bad, width=512, height=512)


def test_generic_image_broadcast_skips_img2img(sdxl_wf):
    wf = inject_img2img(sdxl_wf, image="init.png", denoise=0.6, width=1024, height=1280)
    patched = apply_settings(wf, {"image": "video.png", "width": 640, "height": 640})
    assert patched["img2img_loader"]["inputs"]["image"] == "init.png"
    assert patched["img2img_scale"]["inputs"]["width"] == 640


def test_clear_img2img_restores_empty_latent(sdxl_wf):
    wf = inject_img2img(sdxl_wf, image="x.png", denoise=0.6, width=1024, height=1280)
    cleared = clear_img2img(wf)
    assert not any(str(k).startswith("img2img_") for k in cleared)
    assert cleared["3"]["inputs"]["latent_image"] == ["5", 0]


def test_hires_after_img2img(sdxl_wf):
    wf = inject_img2img(sdxl_wf, image="x.png", denoise=0.6, width=1024, height=1280)
    out = inject_hires(wf, scale=1.5, steps=10, denoise=0.4, width=1024, height=1280)
    assert out["hires_up"]["inputs"]["samples"] == ["3", 0]
    assert out["8"]["inputs"]["samples"] == ["hires_sampler", 0]


def test_build_workflow_with_init_image(tmp_path):
    cfg = from_dict(
        {
            "workflow": {"file": "sdxl_txt2img.api.json"},
            "generation": {
                "positive": "a cat",
                "negative": "blurry",
                "steps": 20,
                "cfg": 5.0,
                "sampler": "euler_ancestral",
                "scheduler": "normal",
                "width": 832,
                "height": 1216,
                "seed": 7,
                "batch_size": 1,
                "denoise": 0.55,
                "init_image": "wrapper/img2img/x.png",
                "filename_prefix": "t",
            },
        },
        root=tmp_path,
    )
    # point the workflows folder at the repo bundle
    repo_wf = Path(__file__).resolve().parents[1] / "workflows"
    cfg.paths.workflows = repo_wf
    wf, seed, pos, _neg = build_workflow(cfg)
    assert wf["img2img_loader"]["inputs"]["image"] == "wrapper/img2img/x.png"
    assert wf["3"]["inputs"]["denoise"] == 0.55
    assert seed == 7


def test_config_init_image_override(tmp_path):
    cfg = from_dict({"generation": {"denoise": 1.0}}, root=tmp_path)
    cfg2 = apply_overrides(cfg, init_image="a.png", denoise=0.5)
    assert cfg2.generation.init_image == "a.png"
    assert cfg2.generation.denoise == 0.5


def test_cli_init_image_flags():
    from comfyui_wrapper.cli import build_parser

    args = build_parser().parse_args(
        ["generate", "--prompt", "hi", "--init-image", "C:/in.png", "--denoise", "0.55"]
    )
    assert args.init_image == "C:/in.png"
    assert args.denoise == 0.55


def test_format_infotext_img2img():
    from comfyui_wrapper.webui import format_infotext

    text = format_infotext(
        positive="cat",
        negative="",
        steps=20,
        sampler="euler_ancestral",
        scheduler="normal",
        cfg=5.0,
        seed=1,
        width=832,
        height=1216,
        checkpoint="model.safetensors",
        img2img={"enable": True, "denoise": 0.6},
    )
    assert "img2img: denoise 0.6" in text


def test_anima_canny_inserts_builtin_node(sdxl_wf):
    from comfyui_wrapper.workflow import inject_anima_lllite

    wf = inject_anima_lllite(
        sdxl_wf,
        [{
            "image": "control/photo.png",
            "model_patch": "anima-lllite-lineart-1.safetensors",
            "strength": 0.8,
            "preprocess": "canny",
            "canny_low": 0.3,
            "canny_high": 0.7,
        }],
    )
    assert wf["anima_canny_0"]["class_type"] == "Canny"
    assert wf["anima_canny_0"]["inputs"]["image"] == ["anima_control_image_0", 0]
    assert wf["anima_canny_0"]["inputs"]["low_threshold"] == 0.3
    assert wf["anima_canny_0"]["inputs"]["high_threshold"] == 0.7
    assert wf["anima_lllite_0"]["inputs"]["image"] == ["anima_canny_0", 0]
    assert wf["3"]["inputs"]["model"] == ["anima_lllite_0", 0]


def test_anima_none_keeps_direct_image(sdxl_wf):
    from comfyui_wrapper.workflow import inject_anima_lllite

    wf = inject_anima_lllite(
        sdxl_wf,
        [{"image": "control/depth.png", "model_patch": "anima-lllite-depth-1.safetensors"}],
    )
    assert not any("canny" in str(k) for k in wf)
    assert wf["anima_lllite_0"]["inputs"]["image"] == ["anima_control_image_0", 0]


def test_anima_rejects_bad_preprocess(sdxl_wf):
    import pytest

    from comfyui_wrapper.workflow import inject_anima_lllite

    with pytest.raises(ValueError, match="Unknown Anima preprocess"):
        inject_anima_lllite(
            sdxl_wf,
            [{"image": "x.png", "model_patch": "m.safetensors", "preprocess": "depth"}],
        )
    with pytest.raises(ValueError, match="Canny thresholds"):
        inject_anima_lllite(
            sdxl_wf,
            [{"image": "x.png", "model_patch": "m.safetensors", "preprocess": "canny",
              "canny_low": 0.8, "canny_high": 0.2}],
        )


def test_anima_controls_helper_with_canny():
    from comfyui_wrapper.webui import anima_lllite_controls

    controls = anima_lllite_controls(
        True, "C:/photo.png", "anima-lllite-lineart-1.safetensors",
        0.8, 0.0, 1.0, "canny", 0.3, 0.7,
    )
    assert controls[0]["preprocess"] == "canny"
    assert controls[0]["canny_low"] == 0.3
    # defaults stay backward compatible (no preprocess key = passthrough)
    legacy = anima_lllite_controls(True, "C:/map.png", "depth.safetensors", 1.0, 0.0, 1.0)
    assert legacy[0].get("preprocess", "none") == "none"
    assert "canny_low" not in legacy[0]


def test_cli_anima_preprocess_flags():
    from comfyui_wrapper.cli import build_parser

    args = build_parser().parse_args(
        ["generate", "--prompt", "hi",
         "--anima-control-image", "C:/photo.png",
         "--anima-lllite-model", "anima-lllite-lineart-1.safetensors",
         "--anima-preprocess", "canny"]
    )
    assert args.anima_preprocess == "canny"
    assert args.anima_canny_low == 0.4
