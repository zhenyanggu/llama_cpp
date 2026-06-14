#!/usr/bin/env python3
"""Collect real Fig. 5(a) E2E latency data through the KV260 semi wrapper."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


CASES = [
    ("P256_D256", "P256\nD256", 256, 256),
    ("P256_D512", "P256\nD512", 256, 512),
    ("P512_D256", "P512\nD256", 512, 256),
    ("P512_D512", "P512\nD512", 512, 512),
]

SYSTEMS = ("cpu", "versavlm")

CPU_NPU_DISABLE_ENV = {
    "AICAS_TEXT_PREFILL_LOG8PV_NPU": "0",
    "AICAS_MMPROJ_LOG8PV_NPU": "0",
    "AICAS_TEXT_DECODE_AWQ_NPU": "0",
    "AICAS_TEXT_DECODE_AWQ_NPU_REQUIRE_ACTIVE": "0",
    "AICAS_TEXT_DECODE_ATTN_NPU": "0",
    "AICAS_TEXT_LM_HEAD_W8A16_NPU": "0",
    "GGML_NPU_TEXT_PREFILL_DYNAMIC": "0",
    "GGML_NPU_PRELOAD_WEIGHTS_ON_LOAD": "0",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_capture(cmd: list[str], cwd: Path, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")
    return path.resolve()


def git_metadata(root: Path) -> dict[str, str]:
    commit = run_capture(["git", "rev-parse", "HEAD"], root).stdout.strip()
    status = run_capture(["git", "status", "--short"], root).stdout.rstrip()
    return {"git_commit": commit, "git_status_short": status}


def load_best_config(path: Path, root: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    def resolve_model_path(value: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        repo_candidate = root / value
        model_quant_candidate = root.parent / "model-quant" / value
        if model_quant_candidate.exists() and not repo_candidate.exists():
            return model_quant_candidate
        return repo_candidate

    official = cfg.get("official_fp16_baseline", {})
    return {
        "raw": cfg,
        "versavlm_model": resolve_model_path(str(cfg["text_model"])).resolve(),
        "versavlm_mmproj": resolve_model_path(str(cfg["mmproj"])).resolve(),
        "cpu_model": resolve_model_path(str(official["text_model"])).resolve(),
        "cpu_mmproj": resolve_model_path(str(official["mmproj"])).resolve(),
    }


def tokenize_count(tokenizer: Path, model: Path, prompt: str, root: Path) -> int:
    cmd = [str(tokenizer), "-m", str(model), "--prompt", prompt, "--no-parse-special", "--show-count", "--log-disable"]
    proc = run_capture(cmd, root, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"tokenizer failed: {proc.stderr.strip() or proc.stdout.strip()}")
    for line in reversed((proc.stdout + "\n" + proc.stderr).splitlines()):
        words = [w.strip(" :") for w in line.replace("=", " ").split()]
        for word in reversed(words):
            if word.isdigit():
                return int(word)
    raise RuntimeError(f"could not parse token count from tokenizer output: {proc.stdout!r} {proc.stderr!r}")


def make_prompt(tokenizer: Path, model: Path, target_tokens: int, root: Path) -> tuple[str, int]:
    seed_sentences = [
        "Describe the image with precise visual evidence and no unsupported assumptions.",
        "Mention the layout, objects, colors, lighting, and any readable text.",
        "Keep the answer factual, ordered, and concise while covering all salient details.",
        "If a detail is ambiguous, state the uncertainty instead of inventing it.",
    ]
    prefix = (
        "You are evaluating a vision-language model for latency. "
        "Analyze the provided image carefully. "
    )
    def prompt_for_repeat(repeat: int) -> str:
        suffix = " ".join(seed_sentences[i % len(seed_sentences)] for i in range(repeat))
        return (prefix + suffix).strip()

    counts: dict[int, int] = {}

    def count_for_repeat(repeat: int) -> int:
        if repeat not in counts:
            counts[repeat] = tokenize_count(tokenizer, model, prompt_for_repeat(repeat), root)
        return counts[repeat]

    lo = 0
    hi = 1
    while count_for_repeat(hi) < target_tokens:
        lo = hi
        hi *= 2
        if hi > 1024:
            break

    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if count_for_repeat(mid) < target_tokens:
            lo = mid
        else:
            hi = mid

    candidates = sorted({max(0, lo - 1), lo, hi, hi + 1})
    best_repeat = min(candidates, key=lambda repeat: abs(count_for_repeat(repeat) - target_tokens))
    return prompt_for_repeat(best_repeat), count_for_repeat(best_repeat)


def run_id(timestamp: str, system: str, case_id: str) -> str:
    return f"{timestamp}-fig5a-{system}-{case_id}"


def build_wrapper_command(
    *,
    args: argparse.Namespace,
    root: Path,
    system: str,
    case_id: str,
    prompt: str,
    decode_tokens: int,
    output_runs_dir: Path,
    port: int,
    timestamp: str,
) -> list[str]:
    wrapper = root / "AICAS2026/aicas_semi/scripts/run_semi_eval_kv260.sh"
    server_bin = args.cpu_server_bin if system == "cpu" else args.versavlm_server_bin
    model = args.cpu_model if system == "cpu" else args.versavlm_model
    mmproj = args.cpu_mmproj if system == "cpu" else args.versavlm_mmproj
    best_config_json = "" if system == "cpu" else str(args.best_config_json)
    cache_type_k = args.cpu_cache_type_k if system == "cpu" else args.versavlm_cache_type_k
    cache_type_v = args.cpu_cache_type_v if system == "cpu" else args.versavlm_cache_type_v
    flash_attn = args.cpu_flash_attn if system == "cpu" else args.versavlm_flash_attn
    cmd = [
        "bash",
        str(wrapper),
        "--skip-build",
        "--server-bin",
        str(server_bin),
        "--best-config-json",
        best_config_json,
        "--model",
        str(model),
        "--mmproj",
        str(mmproj),
        "--output-dir",
        str(output_runs_dir),
        "--run-id",
        run_id(timestamp, system, case_id),
        "--host",
        args.host,
        "--user",
        args.user,
        "--remote-root",
        args.remote_root,
        "--port",
        str(port),
        "--threads",
        str(args.threads),
        "--ubatch-size",
        str(args.ubatch_size),
        "--cache-type-k",
        cache_type_k,
        "--cache-type-v",
        cache_type_v,
        "--flash-attn",
        flash_attn,
        "--ctx-size",
        str(args.ctx_size),
        "--throughput-profile-only",
        "--throughput-profile-max-tokens",
        str(decode_tokens),
        "--throughput-profile-prompt",
        prompt,
        "--prefill-profile-mode",
        "summary",
        "--sudo-password",
        args.sudo_password,
        "--generic-fastpath",
        "--overlay-app",
        args.overlay_app,
        "--decode-overlay-app",
        args.decode_overlay_app,
    ]
    if system == "cpu":
        cmd += [
            "--mtmd-backend-device",
            "CPU",
        "--no-decode-npu",
        "--no-npu-shape-table",
        "--no-npu-preload-weights",
        "--server-env",
        "MTMD_BACKEND_DEVICE=CPU",
        ]
        for key, value in CPU_NPU_DISABLE_ENV.items():
            cmd += ["--server-env", f"{key}={value}"]
    else:
        cmd += [
            "--mtmd-backend-device",
            "NPU",
            "--decode-npu",
            "--npu-text-prefill-dynamic",
            "--npu-preload-weights",
        ]
    return cmd


def parse_profile_metrics(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results_dir = run_dir / "results"
    metrics_path = require_file(results_dir / "throughput_metrics_profile.json", "profile metrics JSON")
    mtmd_path = require_file(results_dir / "mtmd_prefill_summary.json", "MTMD prefill summary JSON")
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    with mtmd_path.open("r", encoding="utf-8") as handle:
        mtmd = json.load(handle)

    mmproj = mtmd.get("mmproj", {}) if isinstance(mtmd, dict) else {}
    image_ms = float(mmproj.get("encode_us", 0.0) or 0.0) / 1000.0
    text_prefill_ms = float(mmproj.get("merged_decode_us", 0.0) or 0.0) / 1000.0
    prefill_ms = float(metrics.get("prompt_ms", 0.0) or 0.0)
    split_source = "mtmd.mmproj.encode_us + mtmd.mmproj.merged_decode_us"
    if text_prefill_ms <= 0.0 and image_ms > 0.0 and prefill_ms >= image_ms:
        text_prefill_ms = prefill_ms - image_ms
        split_source = "mtmd.mmproj.encode_us + prompt_ms_minus_image_ms_fallback"
    metrics.update(
        {
            "metrics_path": str(metrics_path),
            "mtmd_prefill_summary_path": str(mtmd_path),
            "server_log_path": str(run_dir / "server_profile.log"),
            "run_index": 1,
            "is_warmup": False,
            "image_ms": image_ms,
            "text_prefill_ms": text_prefill_ms,
            "prefill_split_residual_ms": prefill_ms - image_ms - text_prefill_ms,
            "prefill_split_source": split_source,
            "mtmd_prefill_summary": {
                "prefill_total_us": mtmd.get("prefill_total_us"),
                "prompt_tokens_processed": mtmd.get("prompt_tokens_processed"),
                "mmproj": {
                    "has_media": mmproj.get("has_media"),
                    "merged_prefill": mmproj.get("merged_prefill"),
                    "text_tokens": mmproj.get("text_tokens"),
                    "media_tokens": mmproj.get("media_tokens"),
                    "image_chunk_count": mmproj.get("image_chunk_count"),
                    "encode_us": mmproj.get("encode_us"),
                    "merged_decode_us": mmproj.get("merged_decode_us"),
                    "total_chunk_us": mmproj.get("total_chunk_us"),
                },
            },
        }
    )
    return [], [metrics]


def summarize(measured: list[dict[str, Any]]) -> dict[str, Any]:
    item = measured[0]
    return {
        "image_ms": float(item["image_ms"]),
        "text_prefill_ms": float(item["text_prefill_ms"]),
        "prefill_ms": float(item["prompt_ms"]),
        "decode_ms": float(item["decode_ms"]),
        "total_ms": float(item["total_ms"]),
        "prefill_split_residual_ms": float(item["prefill_split_residual_ms"]),
        "tokens_prompt_actual": int(item["prompt_tokens"]),
        "tokens_decode_actual": int(item["completion_tokens"]),
    }


def verify_logs(system: str, run_dir: Path, server_bin: Path) -> dict[str, Any]:
    log_path = run_dir / "server_profile.log"
    if not log_path.exists():
        log_path = run_dir / "server.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    return {
        "system": system,
        "server_bin": str(server_bin),
        "server_log_path": str(log_path),
        "server_bin_file": subprocess.run(["file", str(server_bin)], text=True, capture_output=True, check=False).stdout.strip(),
        "log_contains_npu": "NPU" in text or "npu" in text,
        "log_contains_cpu_backend": "CLIP using CPU backend" in text or "MTMD_BACKEND_DEVICE=CPU" in text,
        "log_contains_npu_backend": "CLIP using NPU backend" in text or "using AICAS" in text,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_outputs(
    *,
    root: Path,
    args: argparse.Namespace,
    timestamp_iso: str,
    git: dict[str, str],
    prompts: dict[str, dict[str, Any]],
    raw_cases: dict[str, Any],
    command_templates: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    notes = [
        "Measurements use llama-server response timings: timings.prompt_ms as prefill_ms and timings.predicted_ms as decode_ms.",
        "Image/text prefill split uses mtmd_prefill_summary.json: image_ms = mmproj.encode_us / 1000 and text_prefill_ms = mmproj.merged_decode_us / 1000.",
        "For CPU official FP16 runs where mtmd does not emit merged_decode_us, text_prefill_ms falls back to prompt_ms - image_ms and the raw JSON records prefill_split_source.",
        "prefill_split_residual_ms records prompt_ms - image_ms - text_prefill_ms; this is usually coarse timing overhead/accounting difference.",
        "Model load time is excluded because each measured request is sent after llama-server is already ready.",
        "A fixed image is included in every request; prompt_tokens_actual is taken from server usage and may include multimodal/template tokens.",
        "Each case/system is measured once through the semi wrapper throughput-profile-only path; no averaging is performed.",
        "CPU uses the official FP16 baseline model/mmproj from docs/aicas-current-best-config.json official_fp16_baseline and passes an empty best-config JSON to avoid quantized/NPU env injection.",
        "VersaVLM uses the current best quantized model/mmproj plus the semi-test default prefill/decode overlays.",
    ]
    raw = {
        "schema": "aicas.fig5a_e2e_latency.raw.v1",
        "generated_at": timestamp_iso,
        "metadata": {
            "repo": str(root),
            **git,
            "cpu_model": str(args.cpu_model),
            "cpu_mmproj": str(args.cpu_mmproj),
            "versavlm_model": str(args.versavlm_model),
            "versavlm_mmproj": str(args.versavlm_mmproj),
            "image": str(args.image),
            "best_config_json": str(args.best_config_json),
            "cpu_best_config_json": "",
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
            "command_templates": command_templates,
            "notes": notes,
        },
        "prompts": prompts,
        "cases": raw_cases,
    }

    paper_cases = []
    for case_id, label, prompt_target, decode_target in CASES:
        case = raw_cases[case_id]
        cpu = case["systems"]["cpu"]["summary"]
        versa = case["systems"]["versavlm"]["summary"]
        cpu_total = cpu["total_ms"]
        if cpu_total <= 0:
            raise RuntimeError(f"invalid CPU total for {case_id}: {cpu_total}")
        paper_cases.append(
            {
                "id": case_id,
                "label": label,
                "prompt_tokens_target": prompt_target,
                "decode_tokens_target": decode_target,
                "prompt_tokens_actual": int(cpu["tokens_prompt_actual"]),
                "decode_tokens_actual": int(cpu["tokens_decode_actual"]),
                "absolute_ms": {
                    "cpu": {
                        "image": cpu["image_ms"],
                        "text_prefill": cpu["text_prefill_ms"],
                        "prefill": cpu["prefill_ms"],
                        "decode": cpu["decode_ms"],
                        "total": cpu["total_ms"],
                    },
                    "versavlm": {
                        "image": versa["image_ms"],
                        "text_prefill": versa["text_prefill_ms"],
                        "prefill": versa["prefill_ms"],
                        "decode": versa["decode_ms"],
                        "total": versa["total_ms"],
                    },
                },
                "normalized": {
                    "cpu": {
                        "image": cpu["image_ms"] / cpu_total,
                        "text_prefill": cpu["text_prefill_ms"] / cpu_total,
                        "decode": cpu["decode_ms"] / cpu_total,
                    },
                    "versavlm": {
                        "image": versa["image_ms"] / cpu_total,
                        "text_prefill": versa["text_prefill_ms"] / cpu_total,
                        "decode": versa["decode_ms"] / cpu_total,
                    },
                },
                "speedup": cpu["total_ms"] / versa["total_ms"],
            }
        )

    paper = {
        "schema": "aicas.fig5a_e2e_latency.v1",
        "generated_at": timestamp_iso,
        "metadata": {
            "repo": str(root),
            **git,
            "model": str(args.cpu_model),
            "cpu_model": str(args.cpu_model),
            "cpu_mmproj": str(args.cpu_mmproj),
            "versavlm_model": str(args.versavlm_model),
            "versavlm_mmproj": str(args.versavlm_mmproj),
            "image": str(args.image),
            "cpu_command_template": command_templates["cpu"],
            "versavlm_command_template": command_templates["versavlm"],
            "notes": notes,
        },
        "phases": [
            {"id": "image", "label": "Image"},
            {"id": "text_prefill", "label": "Text Prefill"},
            {"id": "decode", "label": "Decode"},
        ],
        "cases": paper_cases,
    }
    return raw, paper


def parse_args() -> argparse.Namespace:
    root = repo_root()
    best_config_json = root / "docs/aicas-current-best-config.json"
    cfg = load_best_config(best_config_json, root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/fig5_e2e_latency/latest")
    parser.add_argument("--best-config-json", type=Path, default=best_config_json)
    parser.add_argument("--cpu-model", type=Path, default=cfg["cpu_model"])
    parser.add_argument("--cpu-mmproj", type=Path, default=cfg["cpu_mmproj"])
    parser.add_argument("--versavlm-model", type=Path, default=cfg["versavlm_model"])
    parser.add_argument("--versavlm-mmproj", type=Path, default=cfg["versavlm_mmproj"])
    parser.add_argument("--image", type=Path, default=root / "AICAS2026/aicas_semi/code/test2.jpg")
    parser.add_argument("--tokenizer", type=Path, default=root / "build-host/bin/llama-tokenize")
    parser.add_argument("--cpu-server-bin", type=Path, default=root / "build-kv260-cpu/bin/llama-server")
    parser.add_argument("--versavlm-server-bin", type=Path, default=root / "build-kv260-semi/bin/llama-server")
    parser.add_argument("--host", default="192.168.0.10")
    parser.add_argument("--user", default="ubuntu")
    parser.add_argument("--remote-root", default="/home/ubuntu/aicas-semi")
    parser.add_argument("--sudo-password", default=os.environ.get("BOARD_SUDO_PASSWORD", "123456"))
    parser.add_argument("--overlay-app", default="prefill_190m_qkpipe_20260608_0206_app")
    parser.add_argument("--decode-overlay-app", default="dec_200m_latest_0609a_app")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--measure-runs", type=int, default=1)
    parser.add_argument("--systems", default="cpu,versavlm")
    parser.add_argument("--cases", default=",".join(case[0] for case in CASES))
    parser.add_argument("--port-base", type=int, default=19080)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ubatch-size", type=int, default=1024)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--cpu-cache-type-k", default="f16")
    parser.add_argument("--cpu-cache-type-v", default="f16")
    parser.add_argument("--versavlm-cache-type-k", default="q8_0")
    parser.add_argument("--versavlm-cache-type-v", default="q8_0")
    parser.add_argument("--cpu-flash-attn", choices=("on", "off", "auto"), default="auto")
    parser.add_argument("--versavlm-flash-attn", choices=("on", "off", "auto"), default="on")
    parser.add_argument("--command-timeout", type=float, default=7200.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    root = repo_root()
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.best_config_json = require_file(args.best_config_json.resolve(), "best config JSON")
    args.cpu_model = require_file(args.cpu_model.resolve(), "CPU official FP16 model")
    args.cpu_mmproj = require_file(args.cpu_mmproj.resolve(), "CPU official FP16 mmproj")
    args.versavlm_model = require_file(args.versavlm_model.resolve(), "VersaVLM model")
    args.versavlm_mmproj = require_file(args.versavlm_mmproj.resolve(), "VersaVLM mmproj")
    args.image = require_file(args.image.resolve(), "image")
    args.tokenizer = require_file(args.tokenizer.resolve(), "tokenizer")
    args.cpu_server_bin = require_file(args.cpu_server_bin.resolve(), "CPU server")
    args.versavlm_server_bin = require_file(args.versavlm_server_bin.resolve(), "VersaVLM server")

    selected_systems = [item.strip() for item in args.systems.split(",") if item.strip()]
    selected_cases = [item.strip() for item in args.cases.split(",") if item.strip()]
    unknown_systems = sorted(set(selected_systems) - set(SYSTEMS))
    known_case_ids = {case[0] for case in CASES}
    unknown_cases = sorted(set(selected_cases) - known_case_ids)
    if unknown_systems:
        raise SystemExit(f"unknown systems: {unknown_systems}")
    if unknown_cases:
        raise SystemExit(f"unknown cases: {unknown_cases}")
    if set(selected_systems) != set(SYSTEMS):
        raise SystemExit("both cpu and versavlm are required to produce the paper JSON")
    if args.warmup_runs != 0 or args.measure_runs != 1:
        raise SystemExit("current Fig. 5(a) protocol requires --warmup-runs 0 --measure-runs 1")

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    timestamp_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    runs_dir = args.output_dir / "runs"
    logs_dir = args.output_dir / "host_logs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    git = git_metadata(root)
    prompts: dict[str, dict[str, Any]] = {}
    raw_cases: dict[str, Any] = {}

    print(f"[fig5a] repo={root}")
    print(f"[fig5a] output={args.output_dir}")
    print(f"[fig5a] cpu_model={args.cpu_model}")
    print(f"[fig5a] versavlm_model={args.versavlm_model}")
    print(f"[fig5a] image={args.image}")

    for case_id, _label, prompt_target, decode_target in CASES:
        if case_id not in selected_cases:
            continue
        prompt, tokenized_count = make_prompt(args.tokenizer, args.cpu_model, prompt_target, root)
        prompts[case_id] = {
            "prompt": prompt,
            "tokenizer_count": tokenized_count,
            "target": prompt_target,
        }
        raw_cases[case_id] = {
            "prompt_tokens_target": prompt_target,
            "decode_tokens_target": decode_target,
            "prompt": prompt,
            "systems": {},
        }
        for system_index, system in enumerate(selected_systems):
            port = args.port_base + len(raw_cases) * 10 + system_index
            cmd = build_wrapper_command(
                args=args,
                root=root,
                system=system,
                case_id=case_id,
                prompt=prompt,
                decode_tokens=decode_target,
                output_runs_dir=runs_dir,
                port=port,
                timestamp=timestamp,
            )
            host_log = logs_dir / f"{timestamp}-fig5a-{system}-{case_id}.log"
            command_str = shlex.join(cmd)
            run_path = runs_dir / run_id(timestamp, system, case_id)
            print(f"[fig5a] running {system} {case_id} on port {port}")
            print(f"[fig5a] log {host_log}")
            if args.dry_run:
                host_log.write_text(command_str + "\n", encoding="utf-8")
                continue
            proc = subprocess.run(
                cmd,
                cwd=str(root),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=args.command_timeout,
                check=False,
            )
            host_log.write_text(command_str + "\n\n" + proc.stdout, encoding="utf-8")
            if proc.returncode != 0:
                raise RuntimeError(f"{system} {case_id} failed with rc={proc.returncode}; see {host_log}")

            warmup, measured = parse_profile_metrics(run_path)
            server_bin = args.cpu_server_bin if system == "cpu" else args.versavlm_server_bin
            raw_cases[case_id]["systems"][system] = {
                "command": command_str,
                "host_log_path": str(host_log),
                "run_dir": str(run_path),
                "warmup": warmup,
                "measured": measured,
                "summary": summarize(measured),
                "path_verification": verify_logs(system, run_path, server_bin),
            }

    if args.dry_run:
        print("[fig5a] dry-run only; no JSON generated")
        return 0

    command_templates = {
        system: raw_cases[next(iter(raw_cases))]["systems"][system]["command"] if raw_cases else ""
        for system in SYSTEMS
    }
    raw, paper = build_outputs(
        root=root,
        args=args,
        timestamp_iso=timestamp_iso,
        git=git,
        prompts=prompts,
        raw_cases=raw_cases,
        command_templates=command_templates,
    )
    raw_path = args.output_dir / "fig5a_e2e_latency_raw.json"
    paper_path = args.output_dir / "fig5a_e2e_latency_for_paper.json"
    write_json(raw_path, raw)
    write_json(paper_path, paper)
    print(f"[fig5a] wrote {raw_path}")
    print(f"[fig5a] wrote {paper_path}")
    for case in paper["cases"]:
        cpu_total = case["absolute_ms"]["cpu"]["total"]
        versa_total = case["absolute_ms"]["versavlm"]["total"]
        print(f"[fig5a] {case['id']}: cpu={cpu_total:.2f} ms versavlm={versa_total:.2f} ms speedup={case['speedup']:.3f}x")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
