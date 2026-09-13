"""Launch the A1111-style Gradio WebUI.

    python webui.py
    python webui.py --port 7861
    python webui.py --listen
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from comfyui_wrapper.webui import main

if __name__ == "__main__":
    raise SystemExit(main())
