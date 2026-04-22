#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create text-model SmoothQuant policy.")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--clip-mode", choices=["minmax", "percentile"], default="minmax")
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
        reason = item.get("reason", "prefill_gemm_only")
        policy = "W8A8"
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
        quant = item[args.clip_mode]
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
                "smooth_scale": item["smooth_scale"],
                "act_quant_mode": item["act_quant_mode"],
                "act_scale": quant["act_scale"],
                "act_zero_point": quant["act_zero_point"],
                "clip_mode": args.clip_mode,
                "clip_min": quant["clip_min"],
                "clip_max": quant["clip_max"],
                "in_features": item["in_features"],
                "out_features": item["out_features"],
            }
        )

    out = {
        "schema": "aicas.llama.text_sq_policy.v1",
        "candidates": str(Path(args.candidates).resolve()),
        "alpha": args.alpha,
        "clip_mode": args.clip_mode,
        "layers": layers,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote text SmoothQuant policy: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
