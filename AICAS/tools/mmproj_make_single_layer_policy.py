#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build a policy that enables exactly one W8A8 layer and falls back all others.")
    p.add_argument("--input-policy", required=True, help="Input policy/calibration JSON with layers[]")
    p.add_argument("--target-tensor", required=True, help="Tensor name to keep as W8A8")
    p.add_argument("--output", required=True, help="Output policy JSON")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.input_policy, "r", encoding="utf-8") as f:
        doc = json.load(f)

    layers = doc.get("layers", [])
    if not isinstance(layers, list) or len(layers) == 0:
        raise SystemExit("input-policy has no layers[]")

    found = False
    for layer in layers:
        name = layer.get("tensor_name", "")
        if name == args.target_tensor:
            layer["enabled"] = True
            layer["policy"] = "W8A8"
            found = True
        else:
            layer["enabled"] = False
            layer["policy"] = "F16_FALLBACK"

    if not found:
        raise SystemExit(f"target tensor not found: {args.target_tensor}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"Wrote single-layer policy: {out}")
    print(f"Enabled tensor: {args.target_tensor}")
    print(f"Total layers: {len(layers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
