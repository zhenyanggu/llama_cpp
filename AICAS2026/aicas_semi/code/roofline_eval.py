#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from llama_server_client import (
    DEFAULT_BASE_URL,
    chat_completion,
    extract_text_content,
    image_to_data_url,
    normalize_base_url,
)


DEFAULT_IMAGE_PROMPT = (
    "Describe the image in one concise sentence. Focus only on visible content."
)
DEFAULT_TEXT_PROMPT_SEED = (
    "This roofline calibration prompt is intentionally plain and repetitive. "
    "It is used only to measure text prefill and decode performance."
)


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                data = json.load(response)
            break
        except HTTPError as exc:
            body = exc.read().decode("utf-8", "replace").strip()
            if exc.code == 503 and time.monotonic() < deadline:
                time.sleep(5.0)
                continue
            raise RuntimeError(f"HTTP {exc.code} for {url}: {body or exc.reason}") from exc
        except URLError as exc:
            if time.monotonic() < deadline:
                time.sleep(5.0)
                continue
            raise RuntimeError(f"Request failed for {url}: {exc.reason}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(data).__name__}")
    return data


def server_root_url(base_url: str) -> str:
    base = normalize_base_url(base_url)
    return base[:-3] if base.endswith("/v1") else base


def apply_template(base_url: str, messages: list[dict[str, Any]], timeout: float) -> str:
    data = post_json(
        f"{server_root_url(base_url)}/apply-template",
        {"messages": messages},
        timeout,
    )
    prompt = data.get("prompt")
    if not isinstance(prompt, str):
        raise RuntimeError("/apply-template did not return a prompt string")
    return prompt


def tokenize_count(base_url: str, content: str, timeout: float) -> int:
    data = post_json(
        f"{server_root_url(base_url)}/tokenize",
        {"content": content, "parse_special": True},
        timeout,
    )
    tokens = data.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("/tokenize did not return a token list")
    return len(tokens)


def templated_token_count(base_url: str, text: str, timeout: float) -> int:
    prompt = apply_template(
        base_url,
        [{"role": "user", "content": [{"type": "text", "text": text}]}],
        timeout,
    )
    return tokenize_count(base_url, prompt, timeout)


def build_exact_text_prompt(base_url: str, target_tokens: int, timeout: float) -> str:
    if target_tokens <= 0:
        raise ValueError("target token count must be positive")

    filler = " data"
    low = 0
    high = max(1, target_tokens * 3)

    def make_text(n: int) -> str:
        return DEFAULT_TEXT_PROMPT_SEED + filler * n

    while templated_token_count(base_url, make_text(high), timeout) < target_tokens:
        high *= 2

    best_text = DEFAULT_TEXT_PROMPT_SEED
    while low <= high:
        mid = (low + high) // 2
        text = make_text(mid)
        count = templated_token_count(base_url, text, timeout)
        if count <= target_tokens:
            best_text = text
            low = mid + 1
        else:
            high = mid - 1

    text = best_text
    count = templated_token_count(base_url, text, timeout)
    suffixes = [" data", " test", " x", " 0", ".", ",", "\n"]
    guard = 0
    while count < target_tokens and guard < 512:
        progressed = False
        for suffix in suffixes:
            candidate = text + suffix
            candidate_count = templated_token_count(base_url, candidate, timeout)
            if count < candidate_count <= target_tokens:
                text = candidate
                count = candidate_count
                progressed = True
                break
        if not progressed:
            break
        guard += 1

    if count != target_tokens:
        raise RuntimeError(
            f"Could not construct exactly {target_tokens} templated tokens; got {count}"
        )
    return text


def build_messages(args: argparse.Namespace) -> tuple[list[dict[str, Any]], int | None]:
    if args.workload == "image_mmproj":
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(str(image_path))},
                    },
                    {"type": "text", "text": args.prompt or DEFAULT_IMAGE_PROMPT},
                ],
            }
        ], None

    if args.prompt:
        text = args.prompt
    else:
        text = build_exact_text_prompt(
            args.base_url,
            args.target_prompt_tokens,
            args.request_timeout,
        )
    prompt_tokens = templated_token_count(args.base_url, text, args.request_timeout)
    return [{"role": "user", "content": [{"type": "text", "text": text}]}], prompt_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect one roofline workload timing sample.")
    parser.add_argument(
        "--workload",
        choices=("image_mmproj", "text_prefill_512", "text_decode_128avg"),
        required=True,
    )
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--image", default="test2.jpg")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--target-prompt-tokens", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=1200.0)
    parser.add_argument("--cache-prompt", action="store_true")
    parser.add_argument("--no-cache-prompt", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true", default=True)
    parser.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    messages, planned_prompt_tokens = build_messages(args)
    max_tokens = args.max_tokens
    if max_tokens is None:
        max_tokens = 128 if args.workload == "text_decode_128avg" else 1

    cache_prompt = None
    if args.cache_prompt and args.no_cache_prompt:
        raise SystemExit("--cache-prompt and --no-cache-prompt are mutually exclusive")
    if args.cache_prompt:
        cache_prompt = True
    if args.no_cache_prompt:
        cache_prompt = False

    started = time.time()
    response = chat_completion(
        base_url=args.base_url,
        model=args.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
        stream=False,
        cache_prompt=cache_prompt,
        ignore_eos=args.ignore_eos,
        timeout=args.request_timeout,
    )
    completed = time.time()

    timings = response.get("timings", {})
    usage = response.get("usage", {})
    payload = {
        "schema": "aicas_roofline_eval.v1",
        "workload": args.workload,
        "base_url": args.base_url,
        "model": args.model,
        "target_prompt_tokens": args.target_prompt_tokens
        if args.workload != "image_mmproj"
        else None,
        "planned_prompt_tokens": planned_prompt_tokens,
        "max_tokens": max_tokens,
        "ignore_eos": bool(args.ignore_eos),
        "cache_prompt": cache_prompt,
        "wall_ms": (completed - started) * 1000.0,
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
        "prompt_ms": float(timings.get("prompt_ms", 0.0) or 0.0),
        "decode_ms": float(timings.get("predicted_ms", 0.0) or 0.0),
        "prefill_speed_tps": float(timings.get("prompt_per_second", 0.0) or 0.0),
        "decode_speed_tps": float(timings.get("predicted_per_second", 0.0) or 0.0),
        "response_text": extract_text_content(response),
    }
    payload["prompt_token_match"] = (
        args.workload == "image_mmproj"
        or payload["prompt_tokens"] == args.target_prompt_tokens
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "workload",
        "prompt_tokens",
        "completion_tokens",
        "prompt_ms",
        "decode_ms",
        "wall_ms",
    )}, indent=2))


if __name__ == "__main__":
    main()
