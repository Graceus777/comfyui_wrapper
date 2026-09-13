"""HTTP + WebSocket client for a running ComfyUI server."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from comfyui_wrapper.config import ComfyUISettings, WrapperConfig


class ComfyError(RuntimeError):
    """Raised when ComfyUI rejects a request or a job fails."""


@dataclass
class JobResult:
    prompt_id: str
    outputs: dict[str, Any] = field(default_factory=dict)
    images: list[dict[str, Any]] = field(default_factory=list)
    status: dict[str, Any] = field(default_factory=dict)


def _http_json(
    url: str,
    *,
    data: bytes | None = None,
    method: str | None = None,
    timeout: float = 30,
) -> Any:
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            body = resp.read()
            if not body:
                return None
            return json.loads(body.decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ComfyError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except URLError as exc:
        raise ComfyError(f"Cannot reach ComfyUI at {url}: {exc.reason}") from exc


class ComfyClient:
    def __init__(self, settings: ComfyUISettings | None = None, cfg: WrapperConfig | None = None):
        if settings is None:
            settings = cfg.comfyui if cfg is not None else ComfyUISettings()
        self.settings = settings
        self.client_id = str(uuid.uuid4())

    @property
    def base(self) -> str:
        return self.settings.base_url

    def get(self, path: str, timeout: float | None = None) -> Any:
        return _http_json(f"{self.base}{path}", timeout=timeout or 30)

    def post(self, path: str, payload: dict, timeout: float | None = None) -> Any:
        data = json.dumps(payload).encode("utf-8")
        return _http_json(f"{self.base}{path}", data=data, timeout=timeout or 30)

    def ping(self) -> bool:
        try:
            self.get("/system_stats", timeout=5)
            return True
        except ComfyError:
            return False

    def system_stats(self) -> dict:
        return self.get("/system_stats")

    def queue(self) -> dict:
        return self.get("/queue")

    def list_models(self, folder: str) -> list[str]:
        return list(self.get(f"/models/{folder}") or [])

    def list_checkpoints(self) -> list[str]:
        return self.list_models("checkpoints")

    def list_loras(self) -> list[str]:
        return self.list_models("loras")

    def interrupt(self) -> None:
        try:
            self.post("/interrupt", {})
        except ComfyError:
            # Some builds return an empty body; treat connectivity errors only.
            raise

    def queue_prompt(self, workflow: dict, prompt_id: str | None = None) -> str:
        payload: dict[str, Any] = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        if prompt_id:
            payload["prompt_id"] = prompt_id
        result = self.post("/prompt", payload, timeout=60)
        if not isinstance(result, dict):
            raise ComfyError("ComfyUI /prompt returned an empty response")
        if result.get("error"):
            raise ComfyError(f"ComfyUI rejected the workflow: {result['error']}")
        node_errors = result.get("node_errors") or {}
        if node_errors:
            raise ComfyError(f"ComfyUI node errors: {json.dumps(node_errors)}")
        pid = result.get("prompt_id")
        if not pid:
            raise ComfyError(f"ComfyUI /prompt missing prompt_id: {result}")
        return str(pid)

    def get_history(self, prompt_id: str) -> dict:
        data = self.get(f"/history/{prompt_id}") or {}
        return data.get(prompt_id) or data.get(str(prompt_id)) or {}

    def upload_image(self, source: str | Path, subfolder: str = "wrapper") -> str:
        """Upload an image to ComfyUI input and return the LoadImage filename key."""
        source = Path(source)
        if not source.is_file():
            raise ComfyError(f"Image not found: {source}")
        boundary = f"----comfywrap{uuid.uuid4().hex}"
        body = bytearray()

        def field(name: str, value: str) -> None:
            body.extend(f"--{boundary}\r\n".encode())
            body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
            body.extend(value.encode("utf-8"))
            body.extend(b"\r\n")

        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="image"; filename="{source.name}"\r\n'.encode()
        )
        body.extend(b"Content-Type: application/octet-stream\r\n\r\n")
        body.extend(source.read_bytes())
        body.extend(b"\r\n")
        field("type", "input")
        field("subfolder", subfolder)
        field("overwrite", "true")
        body.extend(f"--{boundary}--\r\n".encode())
        url = f"{self.base}/upload/image"
        req = Request(
            url,
            data=bytes(body),
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ComfyError(f"HTTP {exc.code} uploading {source.name}: {detail}") from exc
        except URLError as exc:
            raise ComfyError(f"Cannot upload image {source.name}: {exc.reason}") from exc
        name = str((result or {}).get("name") or "").strip()
        if not name:
            raise ComfyError(f"ComfyUI upload returned no filename: {result!r}")
        sub = str((result or {}).get("subfolder") or "").replace("\\", "/").strip("/")
        return f"{sub}/{name}" if sub else name

    def get_image_bytes(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        qs = urlencode({"filename": filename, "subfolder": subfolder, "type": folder_type})
        url = f"{self.base}/view?{qs}"
        req = Request(url)
        try:
            with urlopen(req, timeout=60) as resp:
                return resp.read()
        except HTTPError as exc:
            raise ComfyError(f"HTTP {exc.code} fetching image {filename}") from exc
        except URLError as exc:
            raise ComfyError(f"Cannot fetch image {filename}: {exc.reason}") from exc

    def wait(
        self,
        prompt_id: str,
        *,
        timeout_s: float | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> JobResult:
        """Wait until the prompt finishes. Prefers WebSocket; falls back to HTTP poll."""
        timeout_s = self.settings.timeout_s if timeout_s is None else timeout_s
        try:
            return self._wait_ws(prompt_id, timeout_s=timeout_s, on_progress=on_progress)
        except Exception as exc:
            if on_progress:
                on_progress({"type": "ws_fallback", "error": str(exc)})
            return self._wait_poll(prompt_id, timeout_s=timeout_s, on_progress=on_progress)

    def _wait_ws(
        self,
        prompt_id: str,
        *,
        timeout_s: float,
        on_progress: Callable[[dict], None] | None,
    ) -> JobResult:
        try:
            import websocket  # type: ignore
        except ImportError as exc:
            raise ComfyError("websocket-client is not installed") from exc

        url = f"{self.settings.ws_url}/ws?clientId={self.client_id}"
        ws = websocket.WebSocket()
        ws.connect(url, timeout=30)
        ws.settimeout(1.0)
        deadline = time.time() + timeout_s
        try:
            while time.time() < deadline:
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    history = self.get_history(prompt_id)
                    if history and _history_done(history):
                        return _result_from_history(prompt_id, history)
                    continue
                if not isinstance(raw, str):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                msg_type = message.get("type")
                data = message.get("data") or {}
                if data.get("prompt_id") not in (None, prompt_id):
                    continue
                if on_progress:
                    on_progress(message)
                if msg_type == "execution_error" and data.get("prompt_id") == prompt_id:
                    raise ComfyError(
                        f"ComfyUI execution_error: {data.get('exception_message') or data}"
                    )
                if msg_type == "execution_interrupted" and data.get("prompt_id") == prompt_id:
                    raise ComfyError("ComfyUI job interrupted")
                done = (
                    (msg_type == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id)
                    or (msg_type == "execution_success" and data.get("prompt_id") == prompt_id)
                )
                if done:
                    history = self.get_history(prompt_id)
                    return _result_from_history(prompt_id, history)
            raise ComfyError(f"Timed out waiting for prompt {prompt_id} after {timeout_s}s")
        finally:
            try:
                ws.close()
            except Exception:
                pass

    def _wait_poll(
        self,
        prompt_id: str,
        *,
        timeout_s: float,
        on_progress: Callable[[dict], None] | None,
    ) -> JobResult:
        deadline = time.time() + timeout_s
        interval = max(0.2, float(self.settings.poll_interval_s))
        while time.time() < deadline:
            history = self.get_history(prompt_id)
            if history:
                status = (history.get("status") or {})
                messages = status.get("messages") or []
                if any(m and m[0] == "execution_error" for m in messages):
                    raise ComfyError(f"ComfyUI execution_error in history for {prompt_id}")
                if _history_done(history):
                    return _result_from_history(prompt_id, history)
            if on_progress:
                on_progress({"type": "poll", "prompt_id": prompt_id})
            time.sleep(interval)
        raise ComfyError(f"Timed out polling prompt {prompt_id} after {timeout_s}s")


def _history_done(history: dict) -> bool:
    if history.get("outputs"):
        return True
    status = history.get("status") or {}
    if status.get("completed"):
        return True
    messages = status.get("messages") or []
    for item in messages:
        if item and item[0] in {"execution_success", "execution_interrupted", "execution_error"}:
            return True
    return False


def _result_from_history(prompt_id: str, history: dict) -> JobResult:
    outputs = history.get("outputs") or {}
    images: list[dict[str, Any]] = []
    for node_id, node_out in outputs.items():
        for image in node_out.get("images") or []:
            entry = dict(image)
            entry["node_id"] = str(node_id)
            images.append(entry)
        for key in ("gifs", "videos", "video"):
            items = node_out.get(key) or []
            if isinstance(items, dict):
                items = [items]
            for video in items:
                if not isinstance(video, dict):
                    continue
                entry = dict(video)
                entry["node_id"] = str(node_id)
                images.append(entry)
    return JobResult(
        prompt_id=prompt_id,
        outputs=outputs,
        images=images,
        status=history.get("status") or {},
    )
