"""Config-driven ComfyUI API wrapper with combinator-style batching."""

from comfyui_wrapper.config import WrapperConfig, load_config
from comfyui_wrapper.client import ComfyClient, ComfyError
from comfyui_wrapper.generate import GenerateResult, generate
from comfyui_wrapper.video import generate_video
from comfyui_wrapper.combinator import Job, prep_jobs, run_jobs, preview_jobs

__all__ = [
    "WrapperConfig",
    "load_config",
    "ComfyClient",
    "ComfyError",
    "GenerateResult",
    "generate",
    "generate_video",
    "Job",
    "prep_jobs",
    "run_jobs",
    "preview_jobs",
]

__version__ = "0.1.0"
