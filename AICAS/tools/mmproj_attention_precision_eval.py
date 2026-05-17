#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[2]
AICAS_DIR = ROOT_DIR / "AICAS"
SIBLING_AICAS = ROOT_DIR.parent / "llama.cpp-kv260-20260407" / "AICAS"

DEFAULT_TEXT_MODEL = (
    SIBLING_AICAS
    / "output"
    / "text-decode-awq-repro"
    / "calib16-a0p125-g32"
    / "text_sq_prefill_decode_awq_calib16_a0p125_g32_kvq8_scale_f16.gguf"
)
DEFAULT_MMPROJ = SIBLING_AICAS / "output" / "smoothquant" / "full-alpha-0_5-minmax" / "mmproj.gguf"
DEFAULT_IMAGE_ROOT = SIBLING_AICAS / "data"
DEFAULT_EVAL_JSON = AICAS_DIR / "sampled.json"
DEFAULT_OUTPUT_ROOT = AICAS_DIR / "output" / "mmproj-attn-precision"

SEMI_30_TARGETS = {
    "Regular Text Recognition": 3,
    "Irregular Text Recognition": 3,
    "Artistic Text Recognition": 3,
    "Handwriting Recognition": 3,
    "Digit String Recognition": 3,
    "Non-Semantic Text Recognition": 3,
    "Scene Text-centric VQA": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local CPU mmproj attention precision accuracy experiments.")
    parser.add_argument("--model", default=str(DEFAULT_TEXT_MODEL), help="Text GGUF model path")
    parser.add_argument("--mmproj", default=str(DEFAULT_MMPROJ), help="Baseline mmproj GGUF path")
    parser.add_argument("--image-root", default=str(DEFAULT_IMAGE_ROOT), help="OCRBench image root")
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL_JSON), help="100-sample OCRBench JSON")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Experiment output root")
    parser.add_argument("--run-id", default="", help="Run id; defaults to UTC timestamp")
    parser.add_argument("--variants", default="bf16,f16", help="Comma-separated candidate precisions")
    parser.add_argument("--bfp16m-k-block", type=int, default=64, help="K block size for BFP16-M matmul simulation")
    parser.add_argument(
        "--bfp16m-exp-mode",
        choices=["kblock", "per_channel"],
        default="kblock",
        help="BFP16-M exponent metadata mode",
    )
    parser.add_argument(
        "--scope",
        choices=["core", "block"],
        default="core",
        help="Precision scope: core casts QK/PV inputs; block also rounds attention-branch activations",
    )
    parser.add_argument("--max-drop-30", type=int, default=2, help="Allowed score drop on 30-sample gate")
    parser.add_argument("--skip-100", action="store_true", help="Only run the 30-sample gate")
    parser.add_argument("--server-bin", default="", help="Optional llama-server binary")
    parser.add_argument("--threads", type=int, default=0, help="llama-server threads; default lets wrapper choose")
    parser.add_argument("--port-base", type=int, default=18080, help="First local port to use")
    parser.add_argument("--python", default=sys.executable, help="Python executable for run_local_acc_eval.sh")
    parser.add_argument("--skip-venv", action="store_true", help="Use current Python in run_local_acc_eval.sh")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def make_sample_30(eval_json: Path, image_root: Path, output: Path) -> list[dict]:
    items = load_json(eval_json)
    by_type: dict[str, list[dict]] = {key: [] for key in SEMI_30_TARGETS}

    for item in items:
        type_name = item.get("type")
        image_path = item.get("image_path")
        if type_name in by_type and image_path and (image_root / image_path).exists():
            by_type[type_name].append(item)

    selected: list[dict] = []
    for type_name, count in SEMI_30_TARGETS.items():
        available = by_type[type_name]
        if len(available) < count:
            raise SystemExit(f"Not enough available samples for {type_name}: need {count}, have {len(available)}")
        selected.extend(available[:count])

    write_json(output, selected)
    return selected


def summarize_result(result_json: Path) -> dict[str, Any]:
    items = load_json(result_json)
    if not isinstance(items, list):
        raise SystemExit(f"Result JSON is not a list: {result_json}")

    type_total = Counter()
    type_score = Counter()
    errors = 0
    for item in items:
        type_name = item.get("type", "")
        if type_name:
            type_total[type_name] += 1
        if item.get("result") == 1 and type_name:
            type_score[type_name] += 1
        predict = item.get("predict")
        if isinstance(predict, str) and predict.startswith(("API_ERROR:", "ERROR:")):
            errors += 1

    score = sum(1 for item in items if item.get("result") == 1)
    return {
        "result_json": str(result_json),
        "samples": len(items),
        "score": score,
        "errors": errors,
        "by_type": {
            key: {"score": type_score[key], "total": type_total[key]}
            for key in sorted(type_total)
        },
    }


def result_diffs(baseline_json: Path, candidate_json: Path) -> list[dict[str, Any]]:
    baseline = load_json(baseline_json)
    candidate = load_json(candidate_json)
    out = []
    for idx, (base_item, cand_item) in enumerate(zip(baseline, candidate), start=1):
        if base_item.get("result") == cand_item.get("result") and base_item.get("predict") == cand_item.get("predict"):
            continue
        out.append(
            {
                "index": idx,
                "image_path": base_item.get("image_path"),
                "question": base_item.get("question"),
                "answers": base_item.get("answers"),
                "baseline_result": base_item.get("result"),
                "candidate_result": cand_item.get("result"),
                "baseline_predict": base_item.get("predict"),
                "candidate_predict": cand_item.get("predict"),
            }
        )
    return out


def run_eval(
    *,
    args: argparse.Namespace,
    precision: str,
    sample_json: Path,
    output_dir: Path,
    port: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        str(AICAS_DIR / "scripts" / "run_local_acc_eval.sh"),
        "--model",
        str(Path(args.model).resolve()),
        "--mmproj",
        str(Path(args.mmproj).resolve()),
        "--image-folder",
        str(Path(args.image_root).resolve()),
        "--ocrbench-file",
        str(sample_json.resolve()),
        "--output-folder",
        str(output_dir.resolve()),
        "--save-name",
        "SmolVLM2",
        "--port",
        str(port),
        "--python",
        args.python,
        "--request-timeout",
        str(args.request_timeout),
        "--request-retries",
        str(args.request_retries),
        "--retry-delay",
        str(args.retry_delay),
    ]
    if args.server_bin:
        cmd.extend(["--server-bin", str(Path(args.server_bin).resolve())])
    if args.threads > 0:
        cmd.extend(["--threads", str(args.threads)])
    if args.skip_venv:
        cmd.append("--skip-venv")

    env = os.environ.copy()
    env["AICAS_MMPROJ_ATTN_PRECISION"] = precision
    env["AICAS_MMPROJ_ATTN_PRECISION_SCOPE"] = args.scope
    env["AICAS_MMPROJ_BFP16M_K_BLOCK"] = str(args.bfp16m_k_block)
    env["AICAS_MMPROJ_BFP16M_EXP_MODE"] = args.bfp16m_exp_mode
    log_path = output_dir / "eval.log"
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.write(f"AICAS_MMPROJ_ATTN_PRECISION={precision}\n\n")
        log.write(f"AICAS_MMPROJ_ATTN_PRECISION_SCOPE={args.scope}\n\n")
        log.write(f"AICAS_MMPROJ_BFP16M_K_BLOCK={args.bfp16m_k_block}\n\n")
        log.write(f"AICAS_MMPROJ_BFP16M_EXP_MODE={args.bfp16m_exp_mode}\n\n")
        log.flush()
        subprocess.run(cmd, cwd=ROOT_DIR, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    result_json = output_dir / "SmolVLM2.json"
    require_path(result_json, f"{precision} result JSON")
    summary = summarize_result(result_json)
    summary["precision"] = precision
    summary["scope"] = args.scope
    summary["output_dir"] = str(output_dir)
    summary["eval_log"] = str(log_path)
    return summary


def write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# mmproj Attention Precision Eval",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- model: `{payload['model']}`",
        f"- mmproj: `{payload['mmproj']}`",
        f"- image_root: `{payload['image_root']}`",
        f"- scope: `{payload.get('scope', 'core')}`",
        f"- bfp16m_k_block: `{payload.get('bfp16m_k_block', 64)}`",
        f"- bfp16m_exp_mode: `{payload.get('bfp16m_exp_mode', 'kblock')}`",
        f"- max_drop_30: `{payload['max_drop_30']}`",
        "",
        "## Results",
        "",
        "| phase | scope | precision | score | total | delta | errors | passed_30 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for phase in ("gate_30", "eval_100"):
        for row in payload.get(phase, []):
            delta = row.get("delta")
            lines.append(
                f"| {phase} | {row.get('scope', payload.get('scope', 'core'))} | {row['precision']} | {row['score']} | {row['samples']} | "
                f"{'' if delta is None else delta} | {row['errors']} | {row.get('passed_30', '')} |"
            )

    lines.extend(["", "## Artifacts", ""])
    for phase in ("gate_30", "eval_100"):
        for row in payload.get(phase, []):
            lines.append(f"- {phase} `{row['precision']}`: `{row['output_dir']}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    model = Path(args.model).resolve()
    mmproj = Path(args.mmproj).resolve()
    image_root = Path(args.image_root).resolve()
    eval_json = Path(args.eval_json).resolve()
    output_root = Path(args.output_root).resolve()

    require_path(model, "text model")
    require_path(mmproj, "mmproj")
    require_path(image_root, "image root")
    require_path(eval_json, "eval JSON")
    require_path(AICAS_DIR / "scripts" / "run_local_acc_eval.sh", "run_local_acc_eval.sh")

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    invalid = [item for item in variants if item not in {"bf16", "f16", "bfp16m"}]
    if invalid:
        raise SystemExit(f"Invalid candidate variants: {invalid}; expected bf16, f16, and/or bfp16m")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-attn-precision")
    run_dir = output_root / run_id
    sample_30 = run_dir / "sample_30.json"
    selected = make_sample_30(eval_json, image_root, sample_30)

    payload: dict[str, Any] = {
        "schema": "aicas.mmproj_attention_precision_eval.v1",
        "run_id": run_id,
        "model": str(model),
        "mmproj": str(mmproj),
        "image_root": str(image_root),
        "eval_json_100": str(eval_json),
        "sample_30": str(sample_30),
        "sample_30_type_counts": dict(Counter(item["type"] for item in selected)),
        "variants": variants,
        "scope": args.scope,
        "bfp16m_k_block": args.bfp16m_k_block,
        "bfp16m_exp_mode": args.bfp16m_exp_mode,
        "max_drop_30": args.max_drop_30,
        "gate_30": [],
        "eval_100": [],
    }
    write_json(run_dir / "run_meta.json", payload)

    print(f"Run directory: {run_dir}")
    print(f"30-sample JSON: {sample_30}")

    next_port = args.port_base
    baseline_30 = run_eval(args=args, precision="f32", sample_json=sample_30, output_dir=run_dir / "30-f32", port=next_port)
    next_port += 1
    baseline_30["delta"] = 0
    baseline_30["passed_30"] = True
    payload["gate_30"].append(baseline_30)
    write_json(run_dir / "summary.json", payload)

    passing_variants = []
    for variant in variants:
        row = run_eval(args=args, precision=variant, sample_json=sample_30, output_dir=run_dir / f"30-{variant}", port=next_port)
        next_port += 1
        row["delta"] = row["score"] - baseline_30["score"]
        row["passed_30"] = row["errors"] == 0 and row["score"] >= baseline_30["score"] - args.max_drop_30
        row["diffs_vs_f32"] = result_diffs(Path(baseline_30["result_json"]), Path(row["result_json"]))
        payload["gate_30"].append(row)
        if row["passed_30"]:
            passing_variants.append(variant)
        write_json(run_dir / "summary.json", payload)
        write_markdown_summary(run_dir / "summary.md", payload)

    if not args.skip_100 and passing_variants:
        baseline_100 = run_eval(args=args, precision="f32", sample_json=eval_json, output_dir=run_dir / "100-f32", port=next_port)
        next_port += 1
        baseline_100["delta"] = 0
        payload["eval_100"].append(baseline_100)
        write_json(run_dir / "summary.json", payload)

        for variant in passing_variants:
            row = run_eval(args=args, precision=variant, sample_json=eval_json, output_dir=run_dir / f"100-{variant}", port=next_port)
            next_port += 1
            row["delta"] = row["score"] - baseline_100["score"]
            row["diffs_vs_f32"] = result_diffs(Path(baseline_100["result_json"]), Path(row["result_json"]))
            payload["eval_100"].append(row)
            write_json(run_dir / "summary.json", payload)
            write_markdown_summary(run_dir / "summary.md", payload)

    write_json(run_dir / "summary.json", payload)
    write_markdown_summary(run_dir / "summary.md", payload)
    print(f"Summary JSON: {run_dir / 'summary.json'}")
    print(f"Summary MD:   {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
