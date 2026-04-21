#!/usr/bin/env python3
import argparse
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Collect mmproj activation stats by running llama-mtmd-profiler on calibration images.")
    p.add_argument("--profiler-bin", default="build-host/bin/llama-mtmd-profiler")
    p.add_argument("--model-gguf", required=True)
    p.add_argument("--mmproj-gguf", required=True)
    p.add_argument("--calib-manifest", required=True)
    p.add_argument("--data-root", default="")
    p.add_argument("--output", default="AICAS/artifacts/mmproj_act_stats.json")
    p.add_argument("--limit", type=int, default=64)
    p.add_argument("--samples-per-tensor", type=int, default=4096)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--keep-temp", action="store_true")
    return p.parse_args()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_tensor_stats(dst: dict, src: dict, max_samples: int, rng: random.Random) -> None:
    dst["count"] = int(dst.get("count", 0)) + int(src.get("count", 0))
    dst["min"] = min(float(dst.get("min", float("inf"))), float(src.get("min", float("inf"))))
    dst["max"] = max(float(dst.get("max", float("-inf"))), float(src.get("max", float("-inf"))))
    src_in_channels = int(src.get("in_channels", 0) or 0)
    if src_in_channels > 0:
        if int(dst.get("in_channels", 0) or 0) in (0, src_in_channels):
            dst["in_channels"] = src_in_channels
        else:
            raise RuntimeError(
                f"in_channels mismatch for {dst.get('tensor_name')}: "
                f"{dst.get('in_channels')} vs {src_in_channels}"
            )

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
            raise RuntimeError(
                f"{key} length mismatch for {dst.get('tensor_name')}: "
                f"{len(dst[key])} vs {len(src_values)}"
            )
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
        idx = rng.randrange(seen)
        if idx < max_samples:
            samples[idx] = float(value)
            sample_channels[idx] = channel

    dst["samples"] = samples
    dst["sample_channels"] = sample_channels
    dst["sample_seen"] = seen


def main() -> int:
    args = parse_args()
    calib = _load_json(args.calib_manifest)
    data_root = Path(args.data_root or calib.get("data_root") or ".")
    images = list(calib.get("images", []))
    if args.limit > 0:
        images = images[:args.limit]

    profiler_bin = Path(args.profiler_bin)
    if not profiler_bin.is_absolute():
        profiler_bin = Path.cwd() / profiler_bin
    clip_cpp = Path.cwd() / "tools/mtmd/clip.cpp"
    if profiler_bin.exists() and clip_cpp.exists():
        if profiler_bin.stat().st_mtime < clip_cpp.stat().st_mtime:
            print(
                "warning: profiler binary is older than tools/mtmd/clip.cpp; "
                "rebuild llama-mtmd-profiler before collecting activation stats",
                flush=True,
            )

    merged: dict[str, dict] = {}
    rng = random.Random(args.seed)
    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="mmproj-act-stats-")
    tmp_dir = Path(tmp_dir_obj.name)

    try:
        for idx, rel_path in enumerate(images, start=1):
            image_path = data_root / rel_path
            stats_path = tmp_dir / f"stats_{idx:04d}.json"
            env = os.environ.copy()
            env["AICAS_MMPROJ_ACT_STATS_FILE"] = str(stats_path)
            env["AICAS_MMPROJ_ACT_SAMPLES"] = str(args.samples_per_tensor)
            log_path = tmp_dir / f"profiler_{idx:04d}.log"

            cmd = [
                str(profiler_bin),
                "-m", args.model_gguf,
                "--mmproj", args.mmproj_gguf,
                "--image", str(image_path),
                "--n-predict", "0",
            ]
            with open(log_path, "wb") as logf:
                subprocess.run(cmd, check=True, env=env, stdout=logf, stderr=subprocess.STDOUT)

            if not stats_path.exists():
                tail = ""
                try:
                    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    tail = "\n".join(lines[-40:])
                except Exception:
                    pass
                raise RuntimeError(
                    "profiler completed but did not write activation stats file: "
                    f"{stats_path}\n"
                    "Most likely cause: build-host/bin/llama-mtmd-profiler was not rebuilt after "
                    "the clip.cpp activation sampling changes.\n"
                    f"Profiler log tail:\n{tail}"
                )

            with open(stats_path, "r", encoding="utf-8") as f:
                stats_doc = json.load(f)

            for item in stats_doc.get("tensors", []):
                slot = merged.setdefault(
                    item["tensor_name"],
                    {
                        "tensor_name": item["tensor_name"],
                        "count": 0,
                        "min": float("inf"),
                        "max": float("-inf"),
                        "in_channels": int(item.get("in_channels", 0) or 0),
                        "per_channel_min": [],
                        "per_channel_max": [],
                        "per_channel_absmax": [],
                        "samples": [],
                        "sample_channels": [],
                        "sample_seen": 0,
                    },
                )
                _merge_tensor_stats(slot, item, args.samples_per_tensor, rng)

            print(f"[{idx}/{len(images)}] {rel_path}")

        tensors = []
        for name in sorted(merged):
            item = merged[name]
            item.pop("sample_seen", None)
            tensors.append(item)

        out = {
            "schema": "aicas.mmproj.act_stats.v1",
            "profiler_bin": str(profiler_bin),
            "model_gguf": str(Path(args.model_gguf).resolve()),
            "mmproj_gguf": str(Path(args.mmproj_gguf).resolve()),
            "calib_manifest": str(Path(args.calib_manifest).resolve()),
            "images_processed": len(images),
            "samples_per_tensor": args.samples_per_tensor,
            "tensors": tensors,
        }

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)

        print(f"Wrote activation stats: {output}")
        print(f"Tensors: {len(tensors)}")
        return 0
    finally:
        if args.keep_temp:
            print(f"Temporary stats kept in: {tmp_dir}")
            tmp_dir_obj._finalizer.detach()  # type: ignore[attr-defined]
        else:
            tmp_dir_obj.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
