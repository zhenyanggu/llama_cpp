import base64
import json
import mimetypes
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"

    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def _should_retry_loading(body: str) -> bool:
    return "Loading model" in body or '"type":"unavailable_error"' in body


def _json_request(
    url: str,
    payload: dict[str, Any] | None,
    timeout: float,
    *,
    loading_wait_timeout: float = 0.0,
    loading_retry_delay: float = 5.0,
) -> Any:
    headers = {}
    data = None
    deadline = None

    if loading_wait_timeout > 0:
        deadline = time.monotonic() + loading_wait_timeout

    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")

    while True:
        request = Request(
            url,
            data=data,
            headers=headers,
            method="POST" if payload is not None else "GET",
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as exc:
            body = exc.read().decode("utf-8", "replace").strip()
            if (
                deadline is not None
                and exc.code == 503
                and _should_retry_loading(body)
                and time.monotonic() < deadline
            ):
                time.sleep(loading_retry_delay)
                continue
            reason = body or exc.reason
            raise RuntimeError(f"HTTP {exc.code} for {url}: {reason}") from exc
        except URLError as exc:
            reason = str(exc.reason)
            if (
                deadline is not None
                and time.monotonic() < deadline
                and any(token in reason for token in ("Broken pipe", "Connection refused", "Connection reset"))
            ):
                time.sleep(loading_retry_delay)
                continue
            raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc


def list_models(base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> dict[str, Any]:
    return _json_request(f"{normalize_base_url(base_url)}/models", None, timeout)


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    temperature: float = 0.0,
    stream: bool = False,
    timeout: float = 300.0,
    loading_wait_timeout: float = 600.0,
    loading_retry_delay: float = 5.0,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    return _json_request(
        f"{normalize_base_url(base_url)}/chat/completions",
        payload,
        timeout,
        loading_wait_timeout=loading_wait_timeout,
        loading_retry_delay=loading_retry_delay,
    )


def extract_text_content(response_payload: dict[str, Any]) -> str:
    choices = response_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return ""

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()

    return str(content).strip()
