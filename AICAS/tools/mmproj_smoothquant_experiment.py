#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AICAS = ROOT / "AICAS"
DEFAULT_LAYER_MANIFEST = AICAS / "artifacts" / "layer_manifest.json"
DEFAULT_CALIB_MANIFEST = AICAS / "calib" / "calib_manifest.json"
DEFAULT_RAW_ACT_STATS = AICAS / "artifacts" / "smoothquant_raw_act_stats.json"
DEFAULT_CANDIDATES = AICAS / "artifacts" / "smoothquant_candidates.json"
DEFAULT_OUTPUT_ROOT = AICAS / "output" / "smoothquant"
DEFAULT_FP16_TEXT_MODEL = AICAS / "gguf" / "SmolVLM2-500M-Video-Instruct-f16.gguf"
DEFAULT_FP16_MMPROJ = AICAS / "gguf" / "mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf"
DEFAULT_EVAL_JSON = AICAS / "sampled.json"
DEFAULT_IMAGE_DIR = AICAS / "data"
DEFAULT_PILOT_JSON = AICAS / "artifacts" / "eval_subsets" / "pilot24_smoothquant.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SmoothQuant PTQ experiments for mmproj.")
    parser.add_argument("--layer-manifest", default=str(DEFAULT_LAYER_MANIFEST))
    parser.add_argument("--profiler-bin", default=str(ROOT / "build-host/bin/llama-mtmd-profiler"))
    parser.add_argument("--text-model", default=str(DEFAULT_FP16_TEXT_MODEL))
    parser.add_argument("--mmproj-gguf", default=str(DEFAULT_FP16_MMPROJ))
    parser.add_argument("--image-folder", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--eval-json", default=str(DEFAULT_EVAL_JSON))
    parser.add_argument("--calib-manifest", default=str(DEFAULT_CALIB_MANIFEST))
    parser.add_argument("--raw-act-stats", default=str(DEFAULT_RAW_ACT_STATS))
    parser.add_argument("--candidates", default=str(DEFAULT_CANDIDATES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--baseline-score", type=int, default=53)
    parser.add_argument("--threshold", type=int, default=-1)
    parser.add_argument("--calib-images", type=int, default=256)
    parser.add_argument("--samples-per-tensor", type=int, default=4096)
    parser.add_argument("--alpha-grid", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--pilot-size", type=int, default=24)
    parser.add_argument("--weight-granularity", choices=["per_channel", "per_tensor"], default="per_channel")
    parser.add_argument("--server-bin", default="")
    parser.add_argument("--port-base", type=int, default=18100)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--python", default="python3")
    parser.add_argument("--skip-venv", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--request-retries", type=int, default=0)
    parser.add_argument("--retry-delay", type=float, default=0.5)
    parser.add_argument("--force-recollect", action="store_true")
    parser.add_argument("--force-repack", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def score_eval_result(path: Path) -> int:
    data = load_json(path)
    return sum(1 for row in data if row.get("result") == 1)


def build_pilot_subset(source_json: Path, output_json: Path, pilot_size: int) -> None:
    if output_json.exists():
        return
    data = load_json(source_json)
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in data:
        buckets[(str(item.get("type", "")), str(item.get("dataset_name", "")))].append(item)
    selected = []
    ordered_keys = sorted(buckets)
    max_bucket_len = max((len(bucket) for bucket in buckets.values()), default=0)
    for idx in range(max_bucket_len):
        for key in ordered_keys:
            bucket = buckets[key]
            if idx < len(bucket):
                selected.append(bucket[idx])
                if len(selected) >= pilot_size:
                    write_json(output_json, selected)
                    return
    write_json(output_json, selected)


def ensure_calib_manifest(args: argparse.Namespace) -> Path:
    calib_manifest = Path(args.calib_manifest).resolve()
    if calib_manifest.exists():
        return calib_manifest
    run(
        [
            sys.executable,
            str(AICAS / "tools" / "mmproj_make_calib_manifest.py"),
            "--data-root",
            str(Path(args.image_folder).resolve()),
            "--eval-json",
            str(Path(args.eval_json).resolve()),
            "--output",
            str(calib_manifest),
            "--num-images",
            str(args.calib_images),
        ]
    )
    return calib_manifest


def ensure_raw_act_stats(args: argparse.Namespace, calib_manifest: Path) -> Path:
    raw_act_stats = Path(args.raw_act_stats).resolve()
    if raw_act_stats.exists() and not args.force_recollect:
        return raw_act_stats
    run(
        [
            sys.executable,
            str(AICAS / "tools" / "mmproj_collect_act_stats.py"),
            "--profiler-bin",
            str(Path(args.profiler_bin).resolve()),
            "--model-gguf",
            str(Path(args.text_model).resolve()),
            "--mmproj-gguf",
            str(Path(args.mmproj_gguf).resolve()),
            "--calib-manifest",
            str(calib_manifest),
            "--output",
            str(raw_act_stats),
            "--limit",
            str(args.calib_images),
            "--samples-per-tensor",
            str(args.samples_per_tensor),
        ]
    )
    return raw_act_stats


def ensure_candidates(args: argparse.Namespace, raw_act_stats: Path) -> Path:
    candidates = Path(args.candidates).resolve()
    if candidates.exists() and not args.force_recollect:
        return candidates
    run(
        [
            sys.executable,
            str(AICAS / "tools" / "mmproj_smoothquant_prepare.py"),
            "--layer-manifest",
            str(Path(args.layer_manifest).resolve()),
            "--act-stats",
            str(raw_act_stats),
            "--mmproj-gguf",
            str(Path(args.mmproj_gguf).resolve()),
            "--output",
            str(candidates),
            "--alpha-grid",
            args.alpha_grid,
            "--eps",
            str(args.eps),
            "--percentile",
            str(args.percentile),
        ]
    )
    return candidates


def build_inventory_from_policy(policy_doc: dict) -> list[dict]:
    items = []
    for layer in policy_doc.get("layers", []):
        items.append(
            {
                "tensor_name": layer["tensor_name"],
                "kind": layer["kind"],
                "layer_index": layer["layer_index"],
                "is_prefill_mulmat": True,
                "is_decode_gemv": False,
                "bucket": "A",
                "quantized": bool(layer.get("enabled") and layer.get("policy") == "W8A8"),
                "reason": layer.get("reason", "ok"),
            }
        )
    return items


def run_eval(
    args: argparse.Namespace,
    candidates: Path,
    output_root: Path,
    alpha: float,
    clip_mode: str,
    skip_tensors: list[str],
    experiment_id: str,
    port: int,
    eval_json: Path,
) -> dict:
    experiment_dir = output_root / experiment_id
    policy_path = experiment_dir / "policy.json"
    packed_path = experiment_dir / "mmproj.gguf"
    pack_summary_path = experiment_dir / "pack_summary.json"
    result_json = experiment_dir / "result.json"

    if result_json.exists() and not args.force_repack:
        score = score_eval_result(result_json)
        policy_doc = load_json(policy_path)
        return {
            "experiment_id": experiment_id,
            "alpha": alpha,
            "clip_mode": clip_mode,
            "skip_tensors": skip_tensors,
            "score": score,
            "policy_path": str(policy_path),
            "result_json": str(result_json),
            "pack_summary_json": str(pack_summary_path),
            "quantized_layers": sum(1 for layer in policy_doc["layers"] if layer.get("enabled")),
            "weight_granularity": args.weight_granularity,
        }

    experiment_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(AICAS / "tools" / "mmproj_smoothquant_make_policy.py"),
        "--candidates",
        str(candidates),
        "--alpha",
        str(alpha),
        "--clip-mode",
        clip_mode,
        "--output",
        str(policy_path),
    ]
    for name in skip_tensors:
        cmd.extend(["--skip-tensor", name])
    run(cmd)

    run(
        [
            sys.executable,
            str(AICAS / "tools" / "mmproj_pack_gguf.py"),
            "--input-gguf",
            str(Path(args.mmproj_gguf).resolve()),
            "--layer-manifest",
            str(Path(args.layer_manifest).resolve()),
            "--quant-params",
            str(policy_path),
            "--output-gguf",
            str(packed_path),
            "--output-quant-params",
            str(pack_summary_path),
            "--mode",
            "quantize",
            "--weight-granularity",
            args.weight_granularity,
        ]
    )

    eval_cmd = [
        "bash",
        str(AICAS / "scripts" / "run_local_acc_eval.sh"),
        "--model",
        str(Path(args.text_model).resolve()),
        "--mmproj",
        str(packed_path),
        "--image-folder",
        str(Path(args.image_folder).resolve()),
        "--ocrbench-file",
        str(eval_json),
        "--output-folder",
        str(experiment_dir),
        "--save-name",
        "result",
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
        eval_cmd.extend(["--server-bin", args.server_bin])
    if args.threads:
        eval_cmd.extend(["--threads", str(args.threads)])
    if args.skip_venv:
        eval_cmd.append("--skip-venv")
    run(eval_cmd)

    policy_doc = load_json(policy_path)
    score = score_eval_result(result_json)
    return {
        "experiment_id": experiment_id,
        "alpha": alpha,
        "clip_mode": clip_mode,
        "skip_tensors": skip_tensors,
        "score": score,
        "policy_path": str(policy_path),
        "result_json": str(result_json),
        "pack_summary_json": str(pack_summary_path),
        "quantized_layers": sum(1 for layer in policy_doc["layers"] if layer.get("enabled")),
        "weight_granularity": args.weight_granularity,
    }


def main() -> int:
    args = parse_args()
    threshold = args.threshold if args.threshold >= 0 else args.baseline_score - 5
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    calib_manifest = ensure_calib_manifest(args)
    raw_act_stats = ensure_raw_act_stats(args, calib_manifest)
    candidates = ensure_candidates(args, raw_act_stats)
    build_pilot_subset(Path(args.eval_json).resolve(), DEFAULT_PILOT_JSON, args.pilot_size)

    candidate_doc = load_json(candidates)
    alphas = [float(entry["alpha"]) for entry in candidate_doc.get("alpha_results", [])]
    layer_template = candidate_doc["alpha_results"][0]["layers"]
    layer_by_tensor = {layer["tensor_name"]: layer for layer in layer_template}
    kinds = sorted({layer["kind"] for layer in layer_template})
    block_kind_map: dict[str, list[str]] = defaultdict(list)
    for layer in layer_template:
        block_kind_map[f"blk_{layer['layer_index']:02d}.{layer['kind']}"].append(layer["tensor_name"])

    records = []
    port = args.port_base

    pilot_rows = []
    for alpha in alphas:
        row = run_eval(
            args,
            candidates,
            output_root,
            alpha,
            "minmax",
            [],
            f"pilot-alpha-{str(alpha).replace('.', '_')}-minmax",
            port,
            DEFAULT_PILOT_JSON.resolve(),
        )
        pilot_rows.append(row)
        records.append(row)
        port += 1
    pilot_rows.sort(key=lambda row: (-row["score"], -row["quantized_layers"], row["alpha"]))
    shortlist = [row["alpha"] for row in pilot_rows[: min(2, len(pilot_rows))]]

    full_rows = []
    for alpha in shortlist:
        row = run_eval(
            args,
            candidates,
            output_root,
            alpha,
            "minmax",
            [],
            f"full-alpha-{str(alpha).replace('.', '_')}-minmax",
            port,
            Path(args.eval_json).resolve(),
        )
        full_rows.append(row)
        records.append(row)
        port += 1

    best_row = sorted(full_rows, key=lambda row: (-row["score"], -row["quantized_layers"], row["alpha"]))[0]
    if best_row["score"] < threshold:
        percentile_rows = []
        for alpha in shortlist:
            row = run_eval(
                args,
                candidates,
                output_root,
                alpha,
                "percentile",
                [],
                f"full-alpha-{str(alpha).replace('.', '_')}-percentile",
                port,
                Path(args.eval_json).resolve(),
            )
            percentile_rows.append(row)
            records.append(row)
            port += 1
        if percentile_rows:
            best_percentile = sorted(percentile_rows, key=lambda row: (-row["score"], -row["quantized_layers"], row["alpha"]))[0]
            if best_percentile["score"] >= best_row["score"]:
                best_row = best_percentile

    chosen_skip_tensors: list[str] = list(best_row["skip_tensors"])
    search_trace = []
    current_score = best_row["score"]
    current_alpha = best_row["alpha"]
    current_clip_mode = best_row["clip_mode"]

    if current_score < threshold:
        for kind in kinds:
            candidate_skip = sorted(chosen_skip_tensors + [layer["tensor_name"] for layer in layer_template if layer["kind"] == kind])
            row = run_eval(
                args,
                candidates,
                output_root,
                current_alpha,
                current_clip_mode,
                candidate_skip,
                f"disable-kind-{kind}-alpha-{str(current_alpha).replace('.', '_')}-{current_clip_mode}",
                port,
                Path(args.eval_json).resolve(),
            )
            records.append(row)
            search_trace.append({"stage": "kind", "candidate": kind, "score": row["score"]})
            port += 1
            if row["score"] > current_score or (row["score"] == current_score and row["quantized_layers"] > best_row["quantized_layers"]):
                best_row = row
                current_score = row["score"]
                chosen_skip_tensors = candidate_skip
            if current_score >= threshold:
                break

    if current_score < threshold:
        for group_name, tensor_names in sorted(block_kind_map.items()):
            if any(name in chosen_skip_tensors for name in tensor_names):
                continue
            candidate_skip = sorted(chosen_skip_tensors + tensor_names)
            row = run_eval(
                args,
                candidates,
                output_root,
                current_alpha,
                current_clip_mode,
                candidate_skip,
                f"disable-{group_name}-alpha-{str(current_alpha).replace('.', '_')}-{current_clip_mode}",
                port,
                Path(args.eval_json).resolve(),
            )
            records.append(row)
            search_trace.append({"stage": "block_kind", "candidate": group_name, "score": row["score"]})
            port += 1
            if row["score"] > current_score or (row["score"] == current_score and row["quantized_layers"] > best_row["quantized_layers"]):
                best_row = row
                current_score = row["score"]
                chosen_skip_tensors = candidate_skip
            if current_score >= threshold:
                break

    if current_score < threshold:
        remaining_tensors = [
            layer["tensor_name"]
            for layer in layer_template
            if layer["tensor_name"] not in chosen_skip_tensors
        ]
        for tensor_name in remaining_tensors:
            candidate_skip = sorted(chosen_skip_tensors + [tensor_name])
            row = run_eval(
                args,
                candidates,
                output_root,
                current_alpha,
                current_clip_mode,
                candidate_skip,
                f"disable-tensor-{tensor_name.replace('.', '_')}",
                port,
                Path(args.eval_json).resolve(),
            )
            records.append(row)
            search_trace.append({"stage": "tensor", "candidate": tensor_name, "score": row["score"]})
            port += 1
            if row["score"] > current_score or (row["score"] == current_score and row["quantized_layers"] > best_row["quantized_layers"]):
                best_row = row
                current_score = row["score"]
                chosen_skip_tensors = candidate_skip
            if current_score >= threshold:
                break

    final_policy = load_json(Path(best_row["policy_path"]))
    inventory = build_inventory_from_policy(final_policy)
    skipped = [item for item in inventory if not item["quantized"]]
    quantized = [item for item in inventory if item["quantized"]]

    summary = {
        "schema": "aicas.mmproj.smoothquant_experiment_summary.v1",
        "baseline_score": args.baseline_score,
        "threshold": threshold,
        "calib_manifest": str(calib_manifest),
        "raw_act_stats": str(raw_act_stats),
        "candidates": str(candidates),
        "alpha_grid": alphas,
        "weight_granularity": args.weight_granularity,
        "pilot_rows": pilot_rows,
        "full_rows": full_rows,
        "best_result": best_row,
        "search_trace": search_trace,
        "layer_inventory": inventory,
        "quantized_layers": quantized,
        "skipped_layers": skipped,
        "final_policy": str(best_row["policy_path"]),
        "final_result_json": str(best_row["result_json"]),
        "all_records": records,
    }
    write_json(output_root / "final_summary.json", summary)
    print(json.dumps(summary["best_result"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
