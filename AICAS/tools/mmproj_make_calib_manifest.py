#!/usr/bin/env python3
import argparse
import json
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build calibration image manifest from AICAS/data.")
    p.add_argument("--data-root", default="AICAS/data", help="Image root directory")
    p.add_argument("--eval-json", default="AICAS/sampled.json", help="Eval JSON to exclude")
    p.add_argument("--output", default="AICAS/calib/calib_manifest.json", help="Output manifest path")
    p.add_argument("--num-images", type=int, default=256, help="Total calibration images")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def list_images(root: str) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for dp, _, files in os.walk(root):
        for fn in files:
            if os.path.splitext(fn.lower())[1] in exts:
                abspath = os.path.join(dp, fn)
                out.append(os.path.relpath(abspath, root))
    return out


def load_eval_set(eval_json: str) -> set[str]:
    with open(eval_json, "r", encoding="utf-8") as f:
        items = json.load(f)
    return {x.get("image_path", "") for x in items if x.get("image_path")}


def main() -> int:
    args = parse_args()
    random.seed(args.seed)

    all_images = list_images(args.data_root)
    eval_set = load_eval_set(args.eval_json)

    pool = [p for p in all_images if p not in eval_set]
    if len(pool) < args.num_images:
        raise RuntimeError(f"Not enough images for calibration: need {args.num_images}, have {len(pool)}")

    by_dataset = {}
    for rel in pool:
        top = rel.split("/", 1)[0]
        by_dataset.setdefault(top, []).append(rel)

    for v in by_dataset.values():
        random.shuffle(v)

    selected = []
    dataset_names = sorted(by_dataset.keys())
    while len(selected) < args.num_images and dataset_names:
        next_round = []
        for name in dataset_names:
            arr = by_dataset[name]
            if arr:
                selected.append(arr.pop())
                if len(selected) >= args.num_images:
                    break
            if arr:
                next_round.append(name)
        dataset_names = next_round

    if len(selected) < args.num_images:
        raise RuntimeError("Failed to collect enough stratified samples")

    selected = selected[: args.num_images]
    manifest = {
        "schema": "aicas.mmproj.calib_manifest.v1",
        "data_root": os.path.abspath(args.data_root),
        "eval_json_excluded": os.path.abspath(args.eval_json),
        "seed": args.seed,
        "num_images": len(selected),
        "images": selected,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Wrote calibration manifest: {out}")
    print(f"Images: {len(selected)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
