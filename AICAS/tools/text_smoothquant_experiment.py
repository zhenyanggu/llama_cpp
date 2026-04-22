#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AICAS = ROOT / "AICAS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run text-model SmoothQuant experiments.")
    parser.add_argument("--profiler-bin", default="build-host/bin/llama-mtmd-profiler")
    parser.add_argument("--eval-script", default="AICAS/scripts/run_local_acc_eval.sh")
    parser.add_argument("--manifest", default="AICAS/artifacts/text_sq_manifest.json")
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--source-weights-gguf", default="")
    parser.add_argument("--mmproj-gguf", required=True)
    parser.add_argument("--eval-json", default="AICAS/sampled.json")
    parser.add_argument("--image-root", default="AICAS/data")
    parser.add_argument("--ocrbench-file", default="AICAS/sampled.json")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--calib-sizes", default="32,64")
    parser.add_argument("--alpha-grid", default="0.4,0.5,0.6")
    parser.add_argument("--clip-modes", default="minmax")
    parser.add_argument("--weight-granularities", default="per_channel,per_tensor")
    parser.add_argument("--missing-stat-strategy", choices=["skip", "same_kind_prev_layer"], default="same_kind_prev_layer")
    parser.add_argument("--samples-per-tensor", type=int, default=4096)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--port-base", type=int, default=18080)
    parser.add_argument("--save-name-prefix", default="text_sq")
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    parser.add_argument("--skip-venv", action="store_true")
    return parser.parse_args()


def parse_csv_int(text: str) -> list[int]:
    return [int(token.strip()) for token in text.split(",") if token.strip()]


def parse_csv_str(text: str) -> list[str]:
    return [token.strip() for token in text.split(",") if token.strip()]


def parse_csv_float(text: str) -> list[float]:
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def count_correct(path: Path) -> int:
    data = load_json(path)
    return int(sum(int(item.get("result", 0)) for item in data))


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    calib_sizes = parse_csv_int(args.calib_sizes)
    alpha_grid = parse_csv_float(args.alpha_grid)
    clip_modes = parse_csv_str(args.clip_modes)
    weight_granularities = parse_csv_str(args.weight_granularities)

    summary = {
        "schema": "aicas.llama.text_sq_experiment_summary.v1",
        "model_gguf": str((ROOT / args.model_gguf).resolve() if not Path(args.model_gguf).is_absolute() else Path(args.model_gguf).resolve()),
        "source_weights_gguf": str((ROOT / args.source_weights_gguf).resolve() if args.source_weights_gguf and not Path(args.source_weights_gguf).is_absolute() else (Path(args.source_weights_gguf).resolve() if args.source_weights_gguf else "")),
        "mmproj_gguf": str((ROOT / args.mmproj_gguf).resolve() if not Path(args.mmproj_gguf).is_absolute() else Path(args.mmproj_gguf).resolve()),
        "manifest": str((ROOT / args.manifest).resolve() if not Path(args.manifest).is_absolute() else Path(args.manifest).resolve()),
        "eval_json": str((ROOT / args.eval_json).resolve() if not Path(args.eval_json).is_absolute() else Path(args.eval_json).resolve()),
        "ocrbench_file": str((ROOT / args.ocrbench_file).resolve() if not Path(args.ocrbench_file).is_absolute() else Path(args.ocrbench_file).resolve()),
        "image_root": str((ROOT / args.image_root).resolve() if not Path(args.image_root).is_absolute() else Path(args.image_root).resolve()),
        "calib_sizes": calib_sizes,
        "alpha_grid": alpha_grid,
        "clip_modes": clip_modes,
        "weight_granularities": weight_granularities,
        "missing_stat_strategy": args.missing_stat_strategy,
        "runs": [],
    }

    for calib_size in calib_sizes:
        calib_dir = output_root / f"calib-{calib_size}"
        calib_dir.mkdir(parents=True, exist_ok=True)
        raw_stats = calib_dir / "raw_act_stats.json"
        candidates = calib_dir / "candidates.json"

        run(
            [
                sys.executable,
                str(AICAS / "tools" / "text_smoothquant_collect_act_stats.py"),
                "--profiler-bin",
                args.profiler_bin,
                "--model-gguf",
                args.model_gguf,
                "--mmproj-gguf",
                args.mmproj_gguf,
                "--eval-json",
                args.eval_json,
                "--image-root",
                args.image_root,
                "--limit",
                str(calib_size),
                "--samples-per-tensor",
                str(args.samples_per_tensor),
                "--output",
                str(raw_stats),
            ],
            ROOT,
        )
        run(
            [
                sys.executable,
                str(AICAS / "tools" / "text_smoothquant_prepare.py"),
                "--manifest",
                args.manifest,
                "--act-stats",
                str(raw_stats),
                "--model-gguf",
                args.model_gguf,
                "--alpha-grid",
                args.alpha_grid,
                "--missing-stat-strategy",
                args.missing_stat_strategy,
                "--output",
                str(candidates),
            ],
            ROOT,
        )

        prepared = load_json(candidates)
        layer_resolution = prepared.get("layer_resolution", [])
        ready_layers = sum(1 for item in layer_resolution if item.get("status") == "ready")
        missing_layers = sum(1 for item in layer_resolution if item.get("status") == "missing")
        backfilled_layers = sum(1 for item in layer_resolution if str(item.get("stat_source", "")).startswith("backfill:"))

        for alpha in alpha_grid:
            alpha_tag = str(alpha).replace(".", "_")
            for clip_mode in clip_modes:
                for weight_granularity in weight_granularities:
                    tag = f"a{alpha_tag}-{clip_mode}-{weight_granularity}"
                    run_dir = calib_dir / tag
                    run_dir.mkdir(parents=True, exist_ok=True)
                    policy = run_dir / "policy.json"
                    model_out = run_dir / "model.gguf"
                    pack_summary = run_dir / "pack_summary.json"
                    eval_dir = run_dir / "eval"

                    run(
                        [
                            sys.executable,
                            str(AICAS / "tools" / "text_smoothquant_make_policy.py"),
                            "--candidates",
                            str(candidates),
                            "--alpha",
                            str(alpha),
                            "--clip-mode",
                            clip_mode,
                            "--output",
                            str(policy),
                        ],
                        ROOT,
                    )
                    pack_cmd = [
                        sys.executable,
                        str(AICAS / "tools" / "text_smoothquant_pack_gguf.py"),
                        "--input-gguf",
                        args.model_gguf,
                        "--policy",
                        str(policy),
                        "--output-gguf",
                        str(model_out),
                        "--output-summary",
                        str(pack_summary),
                        "--weight-granularity",
                        weight_granularity,
                    ]
                    if args.source_weights_gguf:
                        pack_cmd.extend(["--source-weights-gguf", args.source_weights_gguf])
                    run(pack_cmd, ROOT)

                    eval_cmd = [
                        str(ROOT / args.eval_script),
                        "--server-bin",
                        str(ROOT / "build-host" / "bin" / "llama-server"),
                        "--model",
                        str(model_out),
                        "--mmproj",
                        args.mmproj_gguf,
                        "--image-folder",
                        args.image_root,
                        "--ocrbench-file",
                        args.ocrbench_file,
                        "--output-folder",
                        str(eval_dir),
                        "--save-name",
                        f"{args.save_name_prefix}_{calib_size}_{tag}",
                        "--port",
                        str(args.port_base),
                        "--threads",
                        str(args.threads),
                        "--request-timeout",
                        str(args.request_timeout),
                        "--request-retries",
                        str(args.request_retries),
                        "--retry-delay",
                        str(args.retry_delay),
                    ]
                    if args.skip_venv:
                        eval_cmd.append("--skip-venv")
                    run(eval_cmd, ROOT)

                    result_json = eval_dir / f"{args.save_name_prefix}_{calib_size}_{tag}.json"
                    correct = count_correct(result_json)
                    summary["runs"].append(
                        {
                            "calib_size": calib_size,
                            "alpha": alpha,
                            "clip_mode": clip_mode,
                            "weight_granularity": weight_granularity,
                            "raw_act_stats": str(raw_stats.resolve()),
                            "candidates": str(candidates.resolve()),
                            "policy": str(policy.resolve()),
                            "model_gguf": str(model_out.resolve()),
                            "pack_summary": str(pack_summary.resolve()),
                            "result_json": str(result_json.resolve()),
                            "correct": correct,
                            "ready_layers": ready_layers,
                            "backfilled_layers": backfilled_layers,
                            "missing_layers": missing_layers,
                        }
                    )
                    summary_path = output_root / "summary.json"
                    with summary_path.open("w", encoding="utf-8") as f:
                        json.dump(summary, f, indent=2, ensure_ascii=False)
                    print(f"[summary] calib={calib_size} alpha={alpha} clip={clip_mode} weight={weight_granularity} correct={correct}")

    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote experiment summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
