"""Short-reference voice cloning in the wrapper's existing Gradio interface."""
from __future__ import annotations

import queue
import threading

from comfyui_wrapper.tts import clone_voice, installation_status


def build_voice_clone(gr, ctx):
    def generate(text, reference, transcript, mode, language, steps, seed, limit, guidance_scale=1.2):
        if not ctx["busy"].acquire(blocking=False):
            yield None, [], "Another wrapper job is running. Wait for it to finish."
            return
        stop = threading.Event()
        ctx["voice_clone_stop"] = stop
        mailbox = queue.Queue()

        def worker():
            try:
                result = clone_voice(
                    text, reference, cfg=ctx["cfg"], reference_text=transcript,
                    mode="continuation" if mode == "Best match" else "voice_only",
                    language=language, steps=int(steps), seed=int(seed), max_seconds=float(limit),
                    guidance_scale=float(guidance_scale),
                    stop_event=stop, on_progress=lambda message: mailbox.put(("status", message)),
                )
                mailbox.put(("done", result))
            except Exception as error:
                mailbox.put(("error", str(error)))
            finally:
                if ctx.get("voice_clone_stop") is stop:
                    ctx.pop("voice_clone_stop", None)
                ctx["busy"].release()

        threading.Thread(target=worker, daemon=True).start()
        yield None, [], "Checking the reference and preparing the voice model..."
        while True:
            event, value = mailbox.get()
            if event == "status":
                yield None, [], value
            elif event == "error":
                yield None, [], f"Error: {value}"
                return
            else:
                yield str(value.audio), [str(value.audio), str(value.metadata)], (
                    f"Done — {value.duration:.2f}s, {value.sample_rate:,} Hz. Seed: {value.seed}.\n"
                    f"Peak model memory: {value.peak_vram_gb:.2f} GB. GPU memory released."
                )
                return

    def stop_clone():
        stop = ctx.get("voice_clone_stop")
        if stop is None:
            return "No voice clone is running."
        stop.set()
        return "Stopping the voice clone..."

    with gr.Tab("Voice Clone") as tab:
        gr.Markdown(
            "Clone a voice with **dots.tts-SOAR**. Upload a clean **3–10 second** recording "
            "with one speaker, then enter the new words to speak."
        )
        with gr.Row():
            with gr.Column():
                reference = gr.Audio(label="Reference voice", type="filepath", sources=["upload", "microphone"])
                mode = gr.Radio(["Best match", "Voice only"], value="Best match", label="Clone mode")
                transcript = gr.Textbox(
                    label="Exact words spoken in the reference", lines=3,
                    info="Best match uses the recording and its transcript to preserve voice and speaking style.",
                )
                text = gr.Textbox(label="Words to speak", lines=5)
                with gr.Row():
                    language = gr.Dropdown(
                        choices=[("English", "EN"), ("Chinese", "ZH"), ("Spanish", "ES"),
                                 ("French", "FR"), ("German", "DE"), ("Japanese", "JA"),
                                 ("Korean", "KO"), ("Automatic", "auto_detect")],
                        value="EN", label="Language",
                    )
                    steps = gr.Slider(10, 32, value=10, step=1, label="Quality steps (NFE)")
                guidance_scale = gr.Slider(
                    0, 3, value=1.2, step=0.05, label="CFG (guidance scale)",
                    info="Default 1.2. Compare small changes using the same seed.",
                )
                with gr.Row():
                    seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)
                    limit = gr.Slider(3, 30, value=25, step=1, label="Clip limit (seconds)")
                run = gr.Button("Generate cloned voice", variant="primary")
            with gr.Column():
                audio = gr.Audio(label="Cloned voice", type="filepath", interactive=False)
                files = gr.File(label="Saved audio and generation details", file_count="multiple", interactive=False)
                status = gr.Textbox(label="Status", value=installation_status(ctx["cfg"]), lines=4, interactive=False)
                stop = gr.Button("Stop voice clone")
        mode.change(lambda value: gr.update(visible=value == "Best match"), inputs=mode, outputs=transcript, queue=False)
        run.click(generate, inputs=[text, reference, transcript, mode, language, steps, seed, limit, guidance_scale],
                  outputs=[audio, files, status], concurrency_limit=1, api_name="voice_clone")
        stop.click(stop_clone, inputs=[], outputs=status, queue=False)
    return tab
