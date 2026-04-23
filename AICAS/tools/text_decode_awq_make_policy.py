#!/usr/bin/env python3
import argparse
import json
import os
import re
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create decode-only AWQ policy.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--skip-kind", action="append", default=[])
    parser.add_argument("--skip-tensor", action="append", default=[])
    parser.add_argument("--skip-regex", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    doc = json.load(open(args.candidates, "r", encoding="utf-8"))
    entry = next((item for item in doc["alpha_results"] if abs(float(item["alpha"]) - args.alpha) < 1e-8), None)
    if entry is None:
        raise SystemExit(f"alpha {args.alpha} not found")

    skip_kind = set(args.skip_kind)
    skip_tensor = set(args.skip_tensor)
    skip_regex = [re.compile(pattern) for pattern in args.skip_regex]
    layers = []
    for item in entry["layers"]:
        name = item["tensor_name"]
        enabled = True
        reason = item.get("reason", "decode_gemv_only")
        policy = "Q4_AWQ"
        if item["kind"] in skip_kind:
            enabled = False
            policy = "F16_FALLBACK"
            reason = f"skip_kind:{item['kind']}"
        elif name in skip_tensor:
            enabled = False
            policy = "F16_FALLBACK"
            reason = f"skip_tensor:{name}"
        else:
            for regex in skip_regex:
                if regex.search(name):
                    enabled = False
                    policy = "F16_FALLBACK"
                    reason = f"skip_regex:{regex.pattern}"
                    break

        layers.append(
            {
                "tensor_name": name,
                "kind": item["kind"],
                "layer_index": item["layer_index"],
                "enabled": enabled,
                "policy": policy,
                "reason": reason,
                "stat_source": item.get("stat_source", "observed"),
                "alpha": item["alpha"],
                "eps": item["eps"],
                "group_size": args.group_size,
                "weight_bits": item["weight_bits"],
                "smooth_scale": item["smooth_scale"],
                "in_features": item["in_features"],
                "out_features": item["out_features"],
            }
        )

    out = {
        "schema": "aicas.llama.text_decode_awq_policy.v1",
        "candidates": str(Path(args.candidates).resolve()),
        "alpha": args.alpha,
        "group_size": args.group_size,
        "layers": layers,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(out, indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, delete=False) as f:
        f.write(encoded)
        tmp_path = Path(f.name)
    with tmp_path.open("r", encoding="utf-8") as f:
        json.load(f)
    os.replace(tmp_path, output)
    print(f"Wrote decode AWQ policy: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
