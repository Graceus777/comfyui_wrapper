import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from comfyui_wrapper.config import from_dict
from comfyui_wrapper.h3 import CHAR_LORA
from comfyui_wrapper.video import generate_video, prepare_swap_video, video_frame_count


@pytest.fixture
def source_60fps(tmp_path):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe are needed for source-timing regression checks")
    source = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=60:duration=6",
        "-f", "lavfi", "-i", "sine=frequency=480:duration=6", "-c:v", "libx264", "-crf", "10",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(source),
    ], check=True, capture_output=True)
    return source


def probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path),
    ], capture_output=True, text=True, check=True)
    return json.loads(result.stdout)["streams"]


def first_frame(path, start=0):
    return subprocess.run([
        "ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path), "-vf", "fps=24",
        "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
    ], capture_output=True, check=True).stdout


def test_60fps_trim_keeps_speed_start_audio_and_h3_padding(source_60fps, tmp_path):
    prepared = tmp_path / "prepared.mp4"
    delivered, sampled = prepare_swap_video(
        source_60fps, prepared, width=96, height=64, start_seconds=1, end_seconds=4,
    )
    assert (delivered, sampled) == (72, 73)
    assert video_frame_count(prepared) == 73
    video, audio = probe(prepared)
    assert video["avg_frame_rate"] == "24/1"
    assert float(video["duration"]) == pytest.approx(73 / 24, abs=0.001)
    assert float(audio["duration"]) == pytest.approx(3, abs=0.05)
    expected, actual = first_frame(source_60fps, 1), first_frame(prepared)
    assert len(expected) == len(actual) == 96 * 64 * 3
    assert sum(abs(a - b) for a, b in zip(expected, actual)) / len(actual) < 8


def test_whole_silent_clip_and_short_final_frame_padding(source_60fps, tmp_path):
    silent = tmp_path / "silent.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-v", "error", "-i", str(source_60fps), "-t", "0.1", "-an", "-c:v", "copy", str(silent),
    ], check=True, capture_output=True)
    prepared = tmp_path / "short.mp4"
    delivered, sampled = prepare_swap_video(silent, prepared, width=96, height=64)
    assert 1 <= delivered < 5
    assert sampled == video_frame_count(prepared) == 5
    assert len(probe(prepared)) == 1


@pytest.mark.parametrize("mode", ["charswap", "headswap"])
def test_swap_submission_uses_prepared_frames_source_audio_and_only_generated_outputs(
    source_60fps, tmp_path, monkeypatch, mode, repo_root,
):
    cfg = from_dict({
        "paths": {"history": str(tmp_path / "history.jsonl"), "output": str(tmp_path / "out")},
        "video": {
            "character_swap_workflow": str(repo_root / "workflows" / "minimax_h3_character_swap.api.json"),
            "head_swap_workflow": str(repo_root / "workflows" / "minimax_h3_head_swap.api.json"),
            "head_swap_lora": "custom_head.safetensors",
            "head_swap_strength": 0.6,
        },
    }, root=tmp_path)
    staged = []

    class Client:
        def ping(self):
            return False

        def upload_image(self, source, subfolder):
            source = Path(source)
            if source.suffix == ".mp4":
                copy = tmp_path / "staged.mp4"
                shutil.copy2(source, copy)
                staged.append(copy)
            return f"{subfolder}/{source.name}"

        def queue_prompt(self, graph):
            self.graph = graph
            return "test-swap"

        def wait(self, prompt_id, on_progress=None):
            return SimpleNamespace(images=[
                {"filename": "source.mp4", "type": "input"},
                {"filename": "generated.mp4", "type": "output"},
            ])

    saved = []

    def save_output(client, artifacts, destination, prefix):
        saved.extend(artifacts)
        return [str(tmp_path / "generated.mp4")]

    monkeypatch.setattr("comfyui_wrapper.video._save_locally", save_output)
    result = generate_video(
        "Keep the source performance", image=tmp_path / "face.png", mode=mode, ref_videos=[source_60fps],
        source_start=1, source_end=4, width=96, height=64, steps=8, sam_target="the person",
        loop=True, save_latents=False, cfg=cfg, client=Client(),
    )
    graph = result.workflow
    assert video_frame_count(staged[0]) == graph["6"]["inputs"]["length"] == 73
    assert graph["h3_body_trim"]["inputs"]["length"] == 73
    assert graph["h3_output_trim"]["inputs"]["length"] == 72
    assert graph["13"]["inputs"]["audio"] == ["h3_body_vc", 1]
    assert saved == [{"filename": "generated.mp4", "type": "output"}]
    entry = json.loads(cfg.paths.history.read_text().strip())
    assert entry["frames"] == 72
    assert entry["sampling_frames"] == 73
    assert entry["source_start"] == 1
    assert entry["source_end"] == 4
    assert entry["files"] == ["generated.mp4"]
    if mode == "charswap":
        assert graph["h3_ref_lora"]["inputs"]["lora_name"] == CHAR_LORA
        assert graph["6"]["inputs"]["ref_images.ref_image_0"] == ["h3_ref_char", 0]
    else:
        assert graph["1"]["inputs"]["unet_name"] == cfg.video.head_swap_unet
        assert graph["h3_head_lora"]["inputs"]["lora_name"] == cfg.video.head_swap_lora
        assert graph["h3_head_lora"]["inputs"]["strength_model"] == cfg.video.head_swap_strength
        assert graph["16"]["inputs"]["model"] == ["h3_head_lora", 0]


def test_invalid_source_trim_is_rejected_before_ffmpeg(source_60fps, tmp_path):
    with pytest.raises(ValueError, match="after source start"):
        prepare_swap_video(source_60fps, tmp_path / "bad.mp4", width=96, height=64, start_seconds=3, end_seconds=1)
