#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect decode-stage text activation stats for AWQ via llama-mtmd-profiler.")
    parser.add_argument("--profiler-bin", default="build-host/bin/llama-mtmd-profiler")
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--mmproj-gguf", required=True)
    parser.add_argument("--eval-json", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", default="AICAS/artifacts/text_decode_awq_raw_act_stats.json")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--samples-per-tensor", type=int, default=4096)
    parser.add_argument("--predict-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--keep-temp", action="store_true")
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge(dst: dict, src: dict, max_samples: int, rng: random.Random) -> None:
    dst["count"] = int(dst.get("count", 0)) + int(src.get("count", 0))
    dst["min"] = min(float(dst.get("min", float("inf"))), float(src.get("min", float("inf"))))
    dst["max"] = max(float(dst.get("max", float("-inf"))), float(src.get("max", float("-inf"))))
    src_in_channels = int(src.get("in_channels", 0) or 0)
    if src_in_channels > 0:
        if int(dst.get("in_channels", 0) or 0) in (0, src_in_channels):
            dst["in_channels"] = src_in_channels
        else:
            raise RuntimeError(f"in_channels mismatch for {dst.get('tensor_name')}")

    for key, reducer in (
        ("per_channel_min", min),
        ("per_channel_max", max),
        ("per_channel_absmax", max),
    ):
        src_values = src.get(key, [])
        if not src_values:
            continue
        if key not in dst or not dst[key]:
            dst[key] = [float(v) for v in src_values]
            continue
        if len(dst[key]) != len(src_values):
            raise RuntimeError(f"{key} length mismatch for {dst.get('tensor_name')}")
        dst[key] = [float(reducer(a, float(b))) for a, b in zip(dst[key], src_values)]

    samples = list(dst.get("samples", []))
    sample_channels = list(dst.get("sample_channels", []))
    seen = int(dst.get("sample_seen", len(samples)))
    src_channels = list(src.get("sample_channels", []))
    for idx, value in enumerate(src.get("samples", [])):
        channel = int(src_channels[idx]) if idx < len(src_channels) else 0
        seen += 1
        if len(samples) < max_samples:
            samples.append(float(value))
            sample_channels.append(channel)
            continue
        slot = rng.randrange(seen)
        if slot < max_samples:
            samples[slot] = float(value)
            sample_channels[slot] = channel
    dst["samples"] = samples
    dst["sample_channels"] = sample_channels
    dst["sample_seen"] = seen


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    data = load_json(args.eval_json)
    samples = list(data[: args.limit]) if args.limit > 0 else list(data)
    profiler_bin = Path(args.profiler_bin)
    if not profiler_bin.is_absolute():
        profiler_bin = Path.cwd() / profiler_bin

    merged: dict[str, dict] = {}
    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="text-decode-awq-act-stats-")
    tmp_dir = Path(tmp_dir_obj.name)
    try:
        for index, item in enumerate(samples, start=1):
            stats_path = tmp_dir / f"decode_stats_{index:04d}.json"
            log_path = tmp_dir / f"decode_stats_{index:04d}.log"
            image_path = Path(args.image_root) / item["image_path"]

            env = os.environ.copy()
            env["AICAS_TEXT_ACT_STATS_FILE"] = str(stats_path)
            env["AICAS_TEXT_ACT_SAMPLES"] = str(args.samples_per_tensor)
            env["AICAS_TEXT_ACT_COLLECT_MODE"] = "decode"
            env["AICAS_TEXT_SQ_ENABLE_DECODE_GEMV"] = "0"
            env["AICAS_TEXT_DECODE_AWQ"] = "0"

            cmd = [
                str(profiler_bin),
                "-m", args.model_gguf,
                "--mmproj", args.mmproj_gguf,
                "--image", str(image_path),
                "-p", item["question"],
                "-n", str(args.predict_tokens),
            ]
            with open(log_path, "wb") as logf:
                subprocess.run(cmd, check=True, env=env, stdout=logf, stderr=subprocess.STDOUT)

            stats_doc = load_json(str(stats_path))
            for tensor in stats_doc.get("tensors", []):
                slot = merged.setdefault(
                    tensor["tensor_name"],
                    {
                        "tensor_name": tensor["tensor_name"],
                        "count": 0,
                        "in_channels": int(tensor.get("in_channels", 0) or 0),
                        "min": float("inf"),
                        "max": float("-inf"),
                        "per_channel_min": [],
                        "per_channel_max": [],
                        "per_channel_absmax": [],
                        "samples": [],
                        "sample_channels": [],
                        "sample_seen": 0,
                    },
                )
                merge(slot, tensor, args.samples_per_tensor, rng)
            print(f"[{index}/{len(samples)}] {item['image_path']}")

        tensors = []
        for name in sorted(merged):
            item = merged[name]
            item.pop("sample_seen", None)
            tensors.append(item)

        out = {
            "schema": "aicas.llama.text.decode_awq_act_stats.v1",
            "model_gguf": str(Path(args.model_gguf).resolve()),
            "mmproj_gguf": str(Path(args.mmproj_gguf).resolve()),
            "eval_json": str(Path(args.eval_json).resolve()),
            "image_root": str(Path(args.image_root).resolve()),
            "samples_per_tensor": args.samples_per_tensor,
            "predict_tokens": args.predict_tokens,
            "items_processed": len(samples),
            "tensors": tensors,
        }
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"Wrote decode AWQ text activation stats: {output}")
        print(f"Tensors: {len(tensors)}")
        return 0
    finally:
        if args.keep_temp:
            print(f"Temporary files kept in: {tmp_dir}")
            tmp_dir_obj._finalizer.detach()  # type: ignore[attr-defined]
        else:
            tmp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
