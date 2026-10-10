import json
import sys
import threading
import time
from pathlib import Path

import pytest

from comfyui_wrapper.cli import build_parser
from comfyui_wrapper.config import from_dict
from comfyui_wrapper.tts import (
    REQUIRED_MODEL_FILES, VoiceCloneError, clone_voice, prepare_gpu,
)


@pytest.fixture
def voice_runtime(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    for name in REQUIRED_MODEL_FILES:
        (model / name).write_text("fixture")
    reference = tmp_path / "a reference.wav"
    reference.write_bytes(b"reference fixture")
    worker = tmp_path / "voice worker.py"
    worker.write_text(
        "import json, os, sys, wave\n"
        "from pathlib import Path\n"
        "request = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))\n"
        "assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "with wave.open(request['output'], 'wb') as stream:\n"
        "    stream.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))\n"
        "    stream.writeframes(b'\\x01\\x00' * 48000)\n"
        "print('VOICE_CLONE_JSON ' + json.dumps({'event': 'done', 'peak_vram_gb': 2.5}))\n",
        encoding="utf-8",
    )
    cfg = from_dict({
        "paths": {"output": str(tmp_path / "outputs")},
        "voice_clone": {"python": sys.executable, "model_dir": str(model), "worker": str(worker)},
    }, root=tmp_path)
    monkeypatch.setattr("comfyui_wrapper.tts.prepare_gpu", lambda cfg, progress: None)
    return cfg, reference, worker


def test_subprocess_round_trip_preserves_reference_unicode_and_settings(voice_runtime):
    cfg, reference, _ = voice_runtime
    result = clone_voice("A café in the rain.", reference, cfg=cfg,
                         reference_text="This is my reference.", seed=814, steps=16,
                         guidance_scale=1.45)
    record = json.loads(result.metadata.read_text(encoding="utf-8"))
    assert result.audio.is_file()
    assert result.duration == 1
    assert result.sample_rate == 48000
    assert record["text"] == "A café in the rain."
    assert record["steps"] == 16 and record["seed"] == 814
    assert record["guidance_scale"] == 1.45
    request = json.loads((result.metadata.parent / "request.json").read_text(encoding="utf-8"))
    assert request["guidance_scale"] == 1.45
    assert Path(record["reference_audio"]).read_bytes() == reference.read_bytes()
    assert "VOICE_CLONE_JSON " in result.log.read_text()


def test_continuation_needs_transcript_before_starting_worker(voice_runtime):
    cfg, reference, _ = voice_runtime
    with pytest.raises(VoiceCloneError, match="exact transcript"):
        clone_voice("Hello.", reference, cfg=cfg)
    assert not cfg.paths.output.exists()


def test_voice_only_drops_stale_transcript(voice_runtime):
    cfg, reference, _ = voice_runtime
    result = clone_voice("Hello.", reference, cfg=cfg, mode="voice_only",
                         reference_text="This transcript belongs to a different recording.")
    record = json.loads(result.metadata.read_text())
    assert record["reference_text"] == ""
    assert record["mode"] == "voice_only"
    assert record["guidance_scale"] == 1.2
    assert 0 <= result.seed < 2**31


@pytest.mark.parametrize("guidance_scale", [-0.05, 3.05, float("nan"), float("inf")])
def test_invalid_cfg_is_rejected_before_gpu_preparation(voice_runtime, monkeypatch, guidance_scale):
    cfg, reference, _ = voice_runtime
    monkeypatch.setattr("comfyui_wrapper.tts.prepare_gpu", lambda *args: pytest.fail("Invalid CFG reached the GPU"))
    with pytest.raises(VoiceCloneError, match="CFG must be between"):
        clone_voice("Hello.", reference, cfg=cfg, mode="voice_only", guidance_scale=guidance_scale)
    assert not cfg.paths.output.exists()


@pytest.mark.parametrize("guidance_scale", [1.35, 1.45])
def test_voice_clone_controls_reach_the_worker_request(voice_runtime, monkeypatch, guidance_scale):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    gr = pytest.importorskip("gradio")
    from comfyui_wrapper.tts_ui import build_voice_clone

    cfg, reference, _ = voice_runtime
    ctx = {"cfg": cfg, "busy": threading.Lock()}
    with gr.Blocks() as demo:
        build_voice_clone(gr, ctx)
    handler = next(fn for fn in demo.fns.values() if fn.api_name == "voice_clone")
    values = {
        "Words to speak": "A new sentence.", "Reference voice": str(reference),
        "Exact words spoken in the reference": "A reference.", "Clone mode": "Best match",
        "Language": "JA", "Quality steps (NFE)": 12, "Seed (-1 = random)": 818,
        "Clip limit (seconds)": 25, "CFG (guidance scale)": guidance_scale,
    }
    updates = list(handler.fn(*(values[component.label] for component in handler.inputs)))
    audio, files, status = updates[-1]
    assert audio and status.startswith("Done")
    record = json.loads(Path(files[1]).read_text(encoding="utf-8"))
    assert record["steps"] == 12 and record["guidance_scale"] == guidance_scale
    assert record["seed"] == 818 and record["language"] == "JA"
    assert "voice_clone_stop" not in ctx and not ctx["busy"].locked()


def test_worker_error_is_reported_with_durable_log(voice_runtime):
    cfg, reference, worker = voice_runtime
    worker.write_text(
        "print('VOICE_CLONE_JSON {\"event\":\"error\",\"message\":\"reference too long\"}')\n"
        "raise SystemExit(1)\n"
    )
    with pytest.raises(VoiceCloneError, match="reference too long") as error:
        clone_voice("Hello.", reference, cfg=cfg, mode="voice_only")
    assert "worker.log" in str(error.value)
    assert list(cfg.paths.output.glob("voice-clone/*/worker.log"))


def test_stop_terminates_the_owned_worker(voice_runtime):
    cfg, reference, worker = voice_runtime
    worker.write_text("import time\nprint('ready', flush=True)\ntime.sleep(60)\n")
    stop = threading.Event()
    timer = threading.Timer(0.5, stop.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(VoiceCloneError, match="stopped"):
            clone_voice("Hello.", reference, cfg=cfg, mode="voice_only", stop_event=stop)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 10


def test_existing_output_is_preserved(voice_runtime, tmp_path):
    cfg, reference, _ = voice_runtime
    output = tmp_path / "existing.wav"
    output.write_bytes(b"keep this recording")
    with pytest.raises(VoiceCloneError, match="already exists"):
        clone_voice("Hello.", reference, cfg=cfg, mode="voice_only", output=output)
    assert output.read_bytes() == b"keep this recording"
    assert not cfg.paths.output.exists()


def test_busy_comfyui_is_not_unloaded(monkeypatch):
    class BusyComfy:
        def __init__(self, **kwargs):
            pass

        def get(self, *args, **kwargs):
            return {"queue_running": [[1, "owned-by-someone-else"]], "queue_pending": []}

        def post(self, *args, **kwargs):
            pytest.fail("Active ComfyUI must not be unloaded or interrupted")

    monkeypatch.setattr("comfyui_wrapper.tts.ComfyClient", BusyComfy)
    with pytest.raises(VoiceCloneError, match="active or queued"):
        prepare_gpu(from_dict({}), lambda message: None)


def test_cli_voice_clone_arguments():
    args = build_parser().parse_args([
        "tts", "--reference", "voice.wav", "--reference-text", "A reference.",
        "--text", "A new sentence.", "--steps", "16", "--seed", "818",
        "--guidance-scale", "1.35",
    ])
    assert args.cmd == "tts"
    assert args.reference_text == "A reference."
    assert args.steps == 16 and args.seed == 818
    assert args.guidance_scale == 1.35
    assert build_parser().parse_args(["tts", "--check"]).guidance_scale == 1.2
