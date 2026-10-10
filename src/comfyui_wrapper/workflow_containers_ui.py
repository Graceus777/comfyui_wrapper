"""Dedicated input panels, independent of the main model/workflow selectors."""
from __future__ import annotations

import queue
import threading
from pathlib import Path

from comfyui_wrapper.client import ComfyClient
from comfyui_wrapper.container_runner import cancel_owned, run_container


def _progress(message):
    if message.get("type") == "container_status":
        return message["text"]
    data = message.get("data") or {}
    if message.get("type") == "progress":
        return f"Sampling {data.get('value', 0)}/{data.get('max', '?')}"
    if message.get("type") == "executing" and data.get("node"):
        return f"Running {data['node']}..."
    if message.get("type") == "execution_cached":
        return "Using cached inputs..."
    return None


def _callback(ctx, kind, payload_builder, *, video=False):
    def run(*values):
        empty = None if video else []
        if not ctx["busy"].acquire(blocking=False):
            yield empty, [], "Another wrapper job is running. Wait for it to finish."
            return
        ctx["stop"].clear()
        mailbox = queue.Queue()
        client = ComfyClient(cfg=ctx["cfg"])
        job = {"client": client, "prompt_id": None, "kind": kind}
        ctx["container_job"] = job

        def worker():
            try:
                def on_progress(message):
                    status = _progress(message)
                    if status:
                        mailbox.put(("status", status))

                def on_queued(pid):
                    job["prompt_id"] = pid

                result = run_container(kind, payload_builder(*values), cfg=ctx["cfg"],
                                       client=client, stop_event=ctx["stop"],
                                       on_progress=on_progress, on_queued=on_queued)
                mailbox.put(("done", result))
            except Exception as error:
                mailbox.put(("error", str(error)))
            finally:
                if ctx.get("container_job") is job:
                    ctx.pop("container_job", None)
                ctx["busy"].release()

        threading.Thread(target=worker, daemon=True).start()
        yield empty, [], "Checking inputs and preparing the workflow..."
        while True:
            event, value = mailbox.get()
            if event == "status":
                yield empty, [], value
            elif event == "error":
                yield empty, [], f"Error: {value}"
                return
            else:
                files = value["files"]
                media = next((p for p in files if Path(p).suffix.lower() in (".mp4", ".webm", ".mov")), None) if video else [
                    p for p in files if Path(p).suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")]
                status = f"Done. Seed: {value['seed']}." + ("\n" + "\n".join(value["notes"]) if value["notes"] else "")
                yield media, files, status
                return
    return run


def build_container_tools(gr, ctx):
    def stop_tool():
        job = ctx.get("container_job")
        if not job:
            return "No workflow tool is running."
        ctx["stop"].set()
        if job["prompt_id"]:
            try:
                cancel_owned(job["client"], job["prompt_id"])
            except Exception as error:
                return f"Stop requested; could not reach ComfyUI: {error}"
        return "Stop requested."

    def result_panel(*, video=False):
        media = gr.Video(label="Result") if video else gr.Gallery(label="Result", columns=1, height=520)
        files = gr.File(label="Saved files", file_count="multiple", interactive=False)
        status = gr.Textbox(label="Status", value="Ready", lines=3, interactive=False)
        stop = gr.Button("Stop this tool")
        stop.click(stop_tool, inputs=[], outputs=status, queue=False)
        return [media, files, status]

    def seed_control():
        return gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

    with gr.Tab("Workflow tools") as tools_tab:
        gr.Markdown("Choose a tool, add its inputs, and generate. Each tool has its own preset.")
        with gr.Tabs():
            with gr.Tab("MiniMax H3"):
                with gr.Tabs():
                    with gr.Tab("Memory saver video"):
                        gr.Markdown("Animate a first frame, with an optional ending frame. Start with a five-second clip.")
                        with gr.Row():
                            with gr.Column():
                                first = gr.Image(label="First frame", type="filepath", image_mode="RGB")
                                last = gr.Image(label="Last frame (optional)", type="filepath", image_mode="RGB")
                                prompt = gr.Textbox(label="Scene and action", lines=4)
                                with gr.Row():
                                    seconds = gr.Slider(label="Seconds", minimum=5, maximum=12, step=1, value=5)
                                    area = gr.Slider(label="Image size (MP)", minimum=0.3, maximum=0.85, step=0.05, value=0.5)
                                seed = seed_control()
                                run = gr.Button("Generate memory saver video", variant="primary")
                            with gr.Column():
                                outputs = result_panel(video=True)
                        run.click(_callback(ctx, "h3_memory_i2v", lambda a,b,c,d,e,f: {
                            "image":a,"last_image":b,"prompt":c,"seconds":d,"megapixels":e,"seed":f}, video=True),
                            inputs=[first,last,prompt,seconds,area,seed], outputs=outputs, concurrency_limit=1)

                    with gr.Tab("Character swap LoRA"):
                        gr.Markdown("Replace a person using a character picture. Start with a continuous four- or five-second shot; expressions and movement may vary.")
                        with gr.Row():
                            with gr.Column():
                                image = gr.Image(label="Replacement character", type="filepath", image_mode="RGB")
                                clip = gr.Video(label="Source clip")
                                target = gr.Textbox(label="Person to replace", value="the main person")
                                prompt = gr.Textbox(label="Additional instructions (optional)", lines=2)
                                with gr.Row():
                                    start = gr.Number(label="Source start (seconds)", value=0, minimum=0)
                                    end = gr.Number(label="Source end (seconds)", value=5, minimum=0.1)
                                with gr.Row():
                                    area = gr.Slider(label="Image size (MP)", minimum=0.3, maximum=0.85, step=0.05, value=0.5)
                                    strength = gr.Slider(label="Swap strength", minimum=0, maximum=1.5, step=0.05, value=1)
                                turbo = gr.Checkbox(label="Use Turbo 8-step LoRA", value=True,
                                                    info="8 steps with Turbo; 20 steps without it.")
                                override_steps = gr.Checkbox(label="Override steps", value=False,
                                                             info="When off, the tool ignores the slider and uses the preset step count.")
                                steps = gr.Slider(label="Sampling steps", minimum=1, maximum=40, step=1,
                                                  value=8, visible=False)
                                def update_swap_steps(use_turbo, override, current):
                                    return gr.update(visible=override,
                                                     value=current if override else (8 if use_turbo else 20))
                                for control in (turbo, override_steps):
                                    control.change(update_swap_steps, inputs=[turbo, override_steps, steps],
                                                   outputs=steps, queue=False)
                                seed = seed_control()
                                run = gr.Button("Generate LoRA character swap", variant="primary")
                            with gr.Column():
                                outputs = result_panel(video=True)
                        run.click(_callback(ctx, "h3_lora_swap", lambda a,b,c,d,e,f,g,h,i,j,k,l: {
                            "image":a,"video":b,"target":c,"prompt":d,"start":e,"end":f,
                            "megapixels":g,"strength":h,"seed":i,"turbo":j,
                            "override_steps":k,"steps":l}, video=True),
                            inputs=[image,clip,target,prompt,start,end,area,strength,seed,turbo,override_steps,steps],
                            outputs=outputs, concurrency_limit=1)

                    with gr.Tab("OmniChar video"):
                        gr.Markdown("Load an OmniChar character and describe a scene. Available character references, voice and compatible character adapters are picked automatically.")
                        with gr.Row():
                            with gr.Column():
                                character = gr.File(label="Character (.char)", file_types=[".char"], type="filepath")
                                prompt = gr.Textbox(label="Scene, action and dialogue", lines=4)
                                with gr.Row():
                                    voice = gr.Checkbox(label="Use character voice when available", value=True)
                                    adapter = gr.Checkbox(label="Use compatible H3 character adapter", value=True)
                                seconds = gr.Slider(label="Seconds", minimum=5, maximum=12, step=1, value=5)
                                seed = seed_control()
                                run = gr.Button("Generate OmniChar video", variant="primary")
                            with gr.Column():
                                outputs = result_panel(video=True)
                        run.click(_callback(ctx, "h3_omnichar", lambda a,b,c,d,e,f: {
                            "character":a,"prompt":b,"use_voice":c,"use_adapter":d,"seconds":e,"seed":f}, video=True),
                            inputs=[character,prompt,voice,adapter,seconds,seed], outputs=outputs, concurrency_limit=1)

                    with gr.Tab("Guided clip upscale"):
                        gr.Markdown("Restore fine detail in a finished clip. This is an upscale with a light redraw; start with a short source clip.")
                        with gr.Row():
                            with gr.Column():
                                clip = gr.Video(label="Finished clip")
                                with gr.Row():
                                    start = gr.Number(label="Source start (seconds)", value=0, minimum=0)
                                    end = gr.Number(label="Source end (seconds)", value=5, minimum=0.1)
                                area = gr.Slider(label="Output size (MP)", minimum=0.5, maximum=2, step=0.1, value=2)
                                run = gr.Button("Upscale finished clip", variant="primary")
                            with gr.Column():
                                outputs = result_panel(video=True)
                        run.click(_callback(ctx, "h3_guided_upscale", lambda a,b,c,d: {
                            "video":a,"start":b,"end":c,"output_mp":d}, video=True),
                            inputs=[clip,start,end,area], outputs=outputs, concurrency_limit=1)

            with gr.Tab("Qwen Image 2.1"):
                with gr.Tabs():
                    with gr.Tab("Reference control"):
                        gr.Markdown("Edit using up to three references. Strength 1 is neutral. Emphasize a phrase with `(dark green:1.2)`.")
                        with gr.Row():
                            with gr.Column():
                                refs, strengths = [], []
                                for index in range(3):
                                    refs.append(gr.Image(label=f"Image {index+1}" + (" (source)" if index == 0 else " (optional)"), type="filepath", image_mode="RGB"))
                                    strengths.append(gr.Slider(label=f"Image {index+1} influence", minimum=0, maximum=4, step=0.05, value=1))
                                prompt = gr.Textbox(label="Edit instruction", lines=3)
                                with gr.Accordion("Reference mask (optional)", open=False):
                                    gr.Markdown("White selects reference details. The mask controls what the reference contributes; it does not lock output pixels.")
                                    mask = gr.Image(label="Black and white mask", type="filepath", image_mode="RGB")
                                    index = gr.Radio(label="Mask belongs to", choices=[1,2,3], value=1)
                                    mask_strength = gr.Slider(label="Selected detail influence", minimum=0, maximum=4, step=0.05, value=1.2)
                                    isolate = gr.Checkbox(label="Use only white areas of that reference", value=False)
                                seed = seed_control()
                                run = gr.Button("Generate controlled edit", variant="primary")
                            with gr.Column():
                                outputs = result_panel()
                        run.click(_callback(ctx, "qwen_controlled", lambda a,b,c,d,e,f,g,h,i,j,k,l: {
                            "images":[a,b,c],"strengths":[d,e,f],"prompt":g,"mask":h,
                            "mask_index":i,"mask_strength":j,"isolate_mask":k,"seed":l}),
                            inputs=refs+strengths+[prompt,mask,index,mask_strength,isolate,seed], outputs=outputs, concurrency_limit=1)

                    with gr.Tab("Consistent edit"):
                        gr.Markdown("Make a small edit while keeping the surrounding picture consistent. Use Reference control for changes to poses or framing.")
                        with gr.Row():
                            with gr.Column():
                                image = gr.Image(label="Source image", type="filepath", image_mode="RGB")
                                prompt = gr.Textbox(label="Edit instruction", lines=4)
                                strength = gr.Slider(label="Consistency strength", minimum=0, maximum=1.5, step=0.05, value=1)
                                seed = seed_control()
                                run = gr.Button("Generate consistent edit", variant="primary")
                            with gr.Column():
                                outputs = result_panel()
                        run.click(_callback(ctx, "qwen_consistent", lambda a,b,c,d: {
                            "image":a,"prompt":b,"strength":c,"seed":d}), inputs=[image,prompt,strength,seed], outputs=outputs, concurrency_limit=1)

                    with gr.Tab("Outfit swap"):
                        gr.Markdown("Use the person in the first picture and the outfit in the second.")
                        with gr.Row():
                            with gr.Column():
                                image = gr.Image(label="Person", type="filepath", image_mode="RGB")
                                outfit = gr.Image(label="Outfit reference", type="filepath", image_mode="RGB")
                                prompt = gr.Textbox(label="Additional outfit instruction (optional)", lines=3,
                                    value="Change the clothing of the person in image 1 to match the outfit in image 2. Keep the person, pose, background and framing of image 1.")
                                strength = gr.Slider(label="Outfit strength", minimum=0, maximum=1.5, step=0.05, value=1)
                                seed = seed_control()
                                run = gr.Button("Generate outfit swap", variant="primary")
                            with gr.Column():
                                outputs = result_panel()
                        run.click(_callback(ctx, "qwen_outfit", lambda a,b,c,d,e: {
                            "image":a,"outfit":b,"prompt":c,"strength":d,"seed":e}), inputs=[image,outfit,prompt,strength,seed], outputs=outputs, concurrency_limit=1)

                    with gr.Tab("Outpaint"):
                        gr.Markdown("Extend the canvas on any side. The original picture is pasted back at its exact size; choose extensions that keep the total canvas under 2 MP.")
                        with gr.Row():
                            with gr.Column():
                                image = gr.Image(label="Source image", type="filepath", image_mode="RGB")
                                with gr.Row():
                                    left = gr.Number(label="Left (pixels)", value=128, minimum=0, maximum=1024, precision=0)
                                    top = gr.Number(label="Top (pixels)", value=128, minimum=0, maximum=1024, precision=0)
                                with gr.Row():
                                    right = gr.Number(label="Right (pixels)", value=128, minimum=0, maximum=1024, precision=0)
                                    bottom = gr.Number(label="Bottom (pixels)", value=128, minimum=0, maximum=1024, precision=0)
                                prompt = gr.Textbox(label="Scene description (optional)", lines=3)
                                strength = gr.Slider(label="Outpaint strength", minimum=0, maximum=1.5, step=0.05, value=1)
                                seed = seed_control()
                                run = gr.Button("Generate outpaint", variant="primary")
                            with gr.Column():
                                outputs = result_panel()
                        run.click(_callback(ctx, "qwen_outpaint", lambda a,b,c,d,e,f,g,h: {
                            "image":a,"margins":[b,c,d,e],"prompt":f,"strength":g,"seed":h}), inputs=[image,left,top,right,bottom,prompt,strength,seed], outputs=outputs, concurrency_limit=1)

            with gr.Tab("Keyframe animation"):
                gr.Markdown("Add two to five pictures in order. Describe the movement between each pair. Start with two pictures and short transitions.")
                with gr.Row():
                    with gr.Column(scale=3):
                        images = [gr.Image(label=f"Keyframe {i+1}" + (" (optional)" if i > 1 else ""),
                                           type="filepath", image_mode="RGB") for i in range(5)]
                    with gr.Column(scale=3):
                        prompts, durations = [], []
                        for index in range(4):
                            prompts.append(gr.Textbox(label=f"Movement {index+1} → {index+2}", lines=2,
                                value="Smooth continuous movement from the first keyframe to the next."))
                            durations.append(gr.Slider(label=f"Transition {index+1} (seconds)", minimum=0.5, maximum=4, step=0.1, value=1.5))
                        seed = seed_control()
                        run = gr.Button("Generate keyframe animation", variant="primary")
                    with gr.Column(scale=4):
                        outputs = result_panel(video=True)
                run.click(_callback(ctx, "wan_keyframes", lambda *v: {
                    "images":list(v[:5]),"prompts":list(v[5:9]),"durations":list(v[9:13]),"seed":v[13]}, video=True),
                    inputs=images+prompts+durations+[seed], outputs=outputs, concurrency_limit=1)
    return tools_tab
