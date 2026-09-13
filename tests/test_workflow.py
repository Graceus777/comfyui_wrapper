from comfyui_wrapper.workflow import (
    apply_settings,
    apply_vae_loader,
    detect_bindings,
    hires_target_size,
    inject_adetailer,
    inject_hires,
    inject_loras,
    is_ui_format,
    link_ref,
    merge_bindings,
    normalize_sampler,
)


def test_detect_bindings_sdxl(sdxl_wf):
    b = detect_bindings(sdxl_wf)
    assert b["positive"] == ("6", "text")
    assert b["negative"] == ("7", "text")
    assert b["seed"] == ("3", "seed")
    assert b["steps"] == ("3", "steps")
    assert b["cfg"] == ("3", "cfg")
    assert b["sampler"] == ("3", "sampler_name")
    assert b["width"] == ("5", "width")
    assert b["height"] == ("5", "height")
    assert b["checkpoint"] == ("4", "ckpt_name")
    assert b["filename_prefix"] == ("9", "filename_prefix")


def test_apply_settings_patches_nodes(sdxl_wf):
    wf = apply_settings(
        sdxl_wf,
        {
            "positive": "a cat",
            "negative": "blurry",
            "steps": 33,
            "cfg": 5.0,
            "sampler": "Euler a",
            "width": 832,
            "height": 1216,
            "seed": 42,
            "checkpoint": "wai.safetensors",
            "filename_prefix": "batch/char",
        },
    )
    assert wf["6"]["inputs"]["text"] == "a cat"
    assert wf["7"]["inputs"]["text"] == "blurry"
    assert wf["3"]["inputs"]["steps"] == 33
    assert wf["3"]["inputs"]["sampler_name"] == "euler_ancestral"
    assert wf["3"]["inputs"]["seed"] == 42
    assert wf["5"]["inputs"]["width"] == 832
    assert wf["4"]["inputs"]["ckpt_name"] == "wai.safetensors"
    assert wf["9"]["inputs"]["filename_prefix"] == "batch/char"
    # original untouched
    assert sdxl_wf["6"]["inputs"]["text"] == "old positive"


def test_explicit_map_overrides_detection(sdxl_wf):
    b = merge_bindings(sdxl_wf, {"positive": "7.text"})
    assert b["positive"] == ("7", "text")


def test_apply_vae_loader_rewires_decode(sdxl_wf):
    wf = apply_vae_loader(sdxl_wf, "sdxl_vae.safetensors")
    assert wf["vae_loader"]["class_type"] == "VAELoader"
    assert wf["vae_loader"]["inputs"]["vae_name"] == "sdxl_vae.safetensors"
    assert wf["8"]["inputs"]["vae"] == ["vae_loader", 0]
    # checkpoint is unchanged
    assert wf["4"]["inputs"]["ckpt_name"] == "model.safetensors"
    again = apply_vae_loader(wf, "sdxl_vae.safetensors")
    assert "vae_loader_1" not in again
    assert again["8"]["inputs"]["vae"] == ["vae_loader", 0]


def test_apply_vae_loader_after_hires(sdxl_wf):
    wf = inject_hires(sdxl_wf, scale=2, steps=10, denoise=0.35, upscaler="ESRGAN_4x.pth", width=1024, height=1280)
    wf = apply_vae_loader(wf, "sdxl_vae.safetensors")
    assert wf["hires_enc"]["inputs"]["vae"] == ["vae_loader", 0]
    assert wf["hires_dec"]["inputs"]["vae"] == ["vae_loader", 0]


def test_inject_loras_rewires_model_and_clip(sdxl_wf):
    wf = inject_loras(
        sdxl_wf,
        [
            {"file": "char.safetensors", "weight": 0.8},
            {"file": "style.safetensors", "weight": 0.6},
        ],
    )
    assert wf["lora_0"]["class_type"] == "LoraLoader"
    assert wf["lora_0"]["inputs"]["model"] == ["4", 0]
    assert wf["lora_0"]["inputs"]["clip"] == ["4", 1]
    assert wf["lora_1"]["inputs"]["model"] == ["lora_0", 0]
    assert wf["lora_1"]["inputs"]["clip"] == ["lora_0", 1]
    assert wf["3"]["inputs"]["model"] == ["lora_1", 0]
    assert wf["6"]["inputs"]["clip"] == ["lora_1", 1]
    assert wf["7"]["inputs"]["clip"] == ["lora_1", 1]
    # VAE stays on the checkpoint
    assert wf["8"]["inputs"]["vae"] == ["4", 2]


def test_inject_loras_model_only(sdxl_wf):
    wf = inject_loras(sdxl_wf, [{"file": "x.safetensors", "weight": 1.0}], mode="model_only")
    assert wf["lora_0"]["class_type"] == "LoraLoaderModelOnly"
    assert wf["3"]["inputs"]["model"] == ["lora_0", 0]
    assert wf["6"]["inputs"]["clip"] == ["4", 1]


def test_normalize_sampler():
    assert normalize_sampler("Euler a") == "euler_ancestral"
    assert normalize_sampler("DPM++ 2M") == "dpmpp_2m"
    assert normalize_sampler("euler") == "euler"


def test_ui_format_detection():
    assert is_ui_format({"nodes": [], "links": [], "last_node_id": 1})
    assert not is_ui_format({"3": {"class_type": "KSampler", "inputs": {}}})


def test_link_ref():
    assert link_ref(["4", 1]) == ("4", 1)
    assert link_ref("text") is None


def test_hires_target_size():
    assert hires_target_size(1024, 1280, 1.5) == (1536, 1920)
    assert hires_target_size(832, 1216, 2.0) == (1664, 2432)


def test_inject_hires_latent(sdxl_wf):
    wf = inject_hires(sdxl_wf, scale=1.5, steps=12, denoise=0.4, width=1024, height=1280)
    assert wf["hires_up"]["class_type"] == "LatentUpscaleBy"
    assert wf["hires_up"]["inputs"]["samples"] == ["3", 0]
    assert wf["hires_sampler"]["inputs"]["denoise"] == 0.4
    assert wf["hires_sampler"]["inputs"]["steps"] == 12
    assert wf["8"]["inputs"]["samples"] == ["hires_sampler", 0]
    assert wf["9"]["inputs"]["images"] == ["8", 0]


def test_inject_hires_model_upscaler(sdxl_wf):
    wf = inject_hires(
        sdxl_wf, scale=2, steps=10, denoise=0.35, upscaler="ESRGAN_4x.pth",
        width=1024, height=1280,
    )
    assert wf["hires_loader"]["inputs"]["model_name"] == "ESRGAN_4x.pth"
    assert wf["hires_scale"]["inputs"]["width"] == 2048
    assert wf["9"]["inputs"]["images"] == ["hires_dec", 0]


def test_inject_adetailer(sdxl_wf):
    wf = inject_adetailer(sdxl_wf, model="bbox/face_yolov8n.pt", denoise=0.4, steps=16, cfg=5)
    assert wf["ad_detector"]["inputs"]["model_name"] == "bbox/face_yolov8n.pt"
    assert wf["ad_face"]["class_type"] == "FaceDetailer"
    assert wf["ad_face"]["inputs"]["image"] == ["8", 0]
    assert wf["9"]["inputs"]["images"] == ["ad_face", 0]


def test_hires_then_adetailer(sdxl_wf):
    wf = inject_hires(sdxl_wf, scale=1.5, steps=8, denoise=0.45, width=1024, height=1280)
    wf = inject_adetailer(wf, model="bbox/face_yolov8s.pt")
    assert wf["8"]["inputs"]["samples"] == ["hires_sampler", 0]
    assert wf["ad_face"]["inputs"]["image"] == ["8", 0]
    assert wf["9"]["inputs"]["images"] == ["ad_face", 0]
