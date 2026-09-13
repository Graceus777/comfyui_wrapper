from pathlib import Path

from comfyui_wrapper.config import from_dict
from comfyui_wrapper.models import (
    _covers_webui,
    _hardlink_or_symlink,
    ensure_available,
    ensure_extra_model_paths,
    extra_model_paths_payload,
    find_local,
    infer_webui_root,
    link_anima_assets,
    link_unseen_files,
    list_kind,
    list_pdd_acc,
    list_text_encoders,
    merge_names,
    scan_files,
    share_webui_models,
    reset_share_state,
)


def _cfg(tmp_path: Path, **paths):
    payload = {"paths": paths}
    return from_dict(payload, root=tmp_path)


def test_infer_webui_root_from_combinator(tmp_path: Path):
    webui = tmp_path / "sd.webui" / "webui"
    (webui / "models").mkdir(parents=True)
    combo = webui / "extensions" / "sd-combinator-ext"
    combo.mkdir(parents=True)
    cfg = _cfg(tmp_path, combinator_root=str(combo))
    assert infer_webui_root(cfg) == webui


def test_infer_webui_root_explicit(tmp_path: Path):
    webui = tmp_path / "webui"
    (webui / "models").mkdir(parents=True)
    cfg = _cfg(tmp_path, webui_root=str(webui))
    assert infer_webui_root(cfg) == webui


def test_scan_and_list_kind_from_disk(tmp_path: Path):
    webui = tmp_path / "webui"
    ckpt = webui / "models" / "Stable-diffusion"
    lora = webui / "models" / "Lora" / "Characters"
    ckpt.mkdir(parents=True)
    lora.mkdir(parents=True)
    (ckpt / "wai.safetensors").write_bytes(b"x")
    (lora / "AsheIL-21.safetensors").write_bytes(b"x")
    cfg = _cfg(tmp_path, webui_root=str(webui))
    names = list_kind("checkpoints", cfg, client=None)
    assert names == ["wai.safetensors"]
    loras = list_kind("loras", cfg, client=None)
    assert loras == ["Characters/AsheIL-21.safetensors"]
    assert find_local("loras", "AsheIL-21", cfg).name == "AsheIL-21.safetensors"


def test_list_pdd_acc_from_disk(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    folder = comfy / "models" / "pdd_acc"
    folder.mkdir(parents=True)
    (folder / "MiniMax-H3-FL2VA-Acc-8Step.safetensors").write_bytes(b"x")
    (comfy / "main.py").write_text("#", encoding="utf-8")
    cfg = _cfg(tmp_path, comfyui_root=str(comfy))
    names = list_pdd_acc(cfg, client=None)
    assert names == ["MiniMax-H3-FL2VA-Acc-8Step.safetensors"]


def test_merge_names_dedupes_case():
    assert merge_names(["Foo.safetensors"], ["foo.safetensors", "bar.safetensors"]) == [
        "Foo.safetensors",
        "bar.safetensors",
    ]


def test_ensure_available_hardlinks_into_comfy(tmp_path: Path):
    webui = tmp_path / "webui"
    comfy = tmp_path / "ComfyUI"
    src_dir = webui / "models" / "Stable-diffusion"
    dest_dir = comfy / "models" / "checkpoints"
    src_dir.mkdir(parents=True)
    dest_dir.mkdir(parents=True)
    src = src_dir / "wai.safetensors"
    src.write_bytes(b"model-bytes")
    (comfy / "main.py").write_text("#", encoding="utf-8")
    cfg = _cfg(tmp_path, webui_root=str(webui), comfyui_root=str(comfy))
    name = ensure_available("checkpoints", "wai.safetensors", cfg, client=None)
    assert name == "wai.safetensors"
    dest = dest_dir / "wai.safetensors"
    assert dest.is_file()
    assert dest.read_bytes() == b"model-bytes"
    # second call is a no-op
    assert ensure_available("checkpoints", "wai", cfg, client=None) == "wai.safetensors"


def test_hardlink_does_not_copy_bytes(tmp_path: Path):
    src = tmp_path / "a.safetensors"
    dest = tmp_path / "linked.safetensors"
    src.write_bytes(b"abc")
    _hardlink_or_symlink(src, dest)
    assert dest.read_bytes() == b"abc"
    src.write_bytes(b"xyz")
    # hardlink shares inode when possible; symlink also sees the new bytes
    assert dest.read_bytes() == b"xyz"


def test_extra_model_paths_upsert(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    webui = tmp_path / "webui"
    comfy.mkdir()
    webui.mkdir()
    assert ensure_extra_model_paths(comfy, webui) is True
    path = comfy / "extra_model_paths.yaml"
    assert path.is_file()
    text = path.read_text(encoding="utf-8")
    assert "a1111:" in text
    assert "\\" not in text or "models/Stable-diffusion" in text
    # existing a1111 block is left alone
    assert ensure_extra_model_paths(comfy, webui) is False
    payload = extra_model_paths_payload(webui)
    assert _covers_webui(payload, webui)
    assert _covers_webui({"base_path": str(webui / "models")}, webui)
    assert "text_encoders" in payload
    assert "models/Lora" in path.read_text(encoding="utf-8")


def test_link_unseen_embeddings(tmp_path: Path):
    webui = tmp_path / "webui"
    comfy = tmp_path / "ComfyUI"
    src_dir = webui / "embeddings"
    dest_dir = comfy / "models" / "embeddings"
    src_dir.mkdir(parents=True)
    dest_dir.mkdir(parents=True)
    (src_dir / "lazypos.safetensors").write_bytes(b"emb")
    (src_dir / "FastNegativeV2.pt").write_bytes(b"emb")
    (comfy / "main.py").write_text("#", encoding="utf-8")
    cfg = _cfg(tmp_path, webui_root=str(webui), comfyui_root=str(comfy))
    names = list_kind("embeddings", cfg, client=None)
    assert "lazypos.safetensors" in names
    assert "FastNegativeV2.pt" in names
    n = link_unseen_files("embeddings", cfg, client=None)
    assert n == 2
    assert (dest_dir / "lazypos.safetensors").is_file()
    assert (dest_dir / "FastNegativeV2.pt").read_bytes() == b"emb"
    assert link_unseen_files("embeddings", cfg, client=None) == 0


def test_extra_model_paths_merges_text_encoders(tmp_path: Path):
    comfy = tmp_path / "ComfyUI"
    webui = tmp_path / "webui"
    comfy.mkdir()
    webui.mkdir()
    path = comfy / "extra_model_paths.yaml"
    path.write_text(
        "a1111:\n  base_path: {}\n  checkpoints: models/Stable-diffusion\n".format(
            str(webui).replace("\\", "/")
        ),
        encoding="utf-8",
    )
    assert ensure_extra_model_paths(comfy, webui) is True
    text = path.read_text(encoding="utf-8")
    assert "text_encoders:" in text
    assert "checkpoints: models/Stable-diffusion" in text
    assert ensure_extra_model_paths(comfy, webui) is False


def test_link_anima_assets(tmp_path: Path):
    webui = tmp_path / "webui"
    comfy = tmp_path / "ComfyUI"
    ckpt = webui / "models" / "Stable-diffusion"
    vae = comfy / "models" / "vae"
    te = comfy / "models" / "text_encoders"
    unet = comfy / "models" / "diffusion_models"
    ckpt.mkdir(parents=True)
    vae.mkdir(parents=True)
    te.mkdir(parents=True)
    unet.mkdir(parents=True)
    (ckpt / "waiANIMA_v10Base10.safetensors").write_bytes(b"anima-unet")
    (ckpt / "waiIllustriousSDXL_v150.safetensors").write_bytes(b"sdxl")
    (vae / "qwen_image_vae.safetensors").write_bytes(b"anima-vae")
    (te / "qwen_3_06b_base.safetensors").write_bytes(b"anima-te")
    (comfy / "main.py").write_text("#", encoding="utf-8")
    cfg = _cfg(tmp_path, webui_root=str(webui), comfyui_root=str(comfy))
    notes = link_anima_assets(cfg)
    dest = unet / "waiANIMA_v10Base10.safetensors"
    assert dest.is_file()
    assert dest.read_bytes() == b"anima-unet"
    assert not (unet / "waiIllustriousSDXL_v150.safetensors").exists()
    assert any("Anima UNET" in n for n in notes)
    assert (webui / "models" / "VAE" / "qwen_image_vae.safetensors").read_bytes() == b"anima-vae"
    assert (webui / "models" / "text_encoders" / "qwen_3_06b_base.safetensors").read_bytes() == b"anima-te"
    notes2 = link_anima_assets(cfg)
    assert notes2 == []
    tes = list_text_encoders(cfg, client=None)
    assert "qwen_3_06b_base.safetensors" in tes
    assert ensure_available("diffusion_models", "waiANIMA_v10Base10.safetensors", cfg) == "waiANIMA_v10Base10.safetensors"


def test_share_notes_without_comfy_root(tmp_path: Path):
    reset_share_state()
    webui = tmp_path / "webui"
    (webui / "models").mkdir(parents=True)
    cfg = _cfg(tmp_path, webui_root=str(webui))
    notes = share_webui_models(cfg, client=None, force=True)
    assert any("A1111 models" in n for n in notes)
