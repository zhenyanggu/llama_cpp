#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a concrete SmoothQuant policy from prepared candidates.")
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
    with open(args.candidates, "r", encoding="utf-8") as f:
        doc = json.load(f)

    alpha_entry = None
    for entry in doc.get("alpha_results", []):
        if abs(float(entry["alpha"]) - float(args.alpha)) < 1e-8:
            alpha_entry = entry
            break
    if alpha_entry is None:
        raise SystemExit(f"alpha {args.alpha} not found in {args.candidates}")

    regexes = [re.compile(pattern) for pattern in args.skip_regex]
    skip_kind = set(args.skip_kind)
    skip_tensor = set(args.skip_tensor)

    layers_out = []
    for layer in alpha_entry.get("layers", []):
        selected = layer["smoothed_activation"][args.clip_mode]
        name = layer["tensor_name"]
        reason = "ok"
        enabled = True
        policy = "W8A8"
        if layer["kind"] in skip_kind:
            enabled = False
            policy = "F16_FALLBACK"
            reason = f"skip_kind:{layer['kind']}"
        elif name in skip_tensor:
            enabled = False
            policy = "F16_FALLBACK"
            reason = f"skip_tensor:{name}"
        else:
            for regex in regexes:
                if regex.search(name):
                    enabled = False
                    policy = "F16_FALLBACK"
                    reason = f"skip_regex:{regex.pattern}"
                    break

        layers_out.append(
            {
                "id": layer["id"],
                "tensor_name": name,
                "kind": layer["kind"],
                "layer_index": layer["layer_index"],
                "in_features": layer["in_features"],
                "out_features": layer["out_features"],
                "enabled": enabled,
                "policy": policy,
                "reason": reason,
                "scheme": layer["scheme"],
                "act_quant_mode": layer["act_quant_mode"],
                "alpha": layer["alpha"],
                "eps": layer["eps"],
                "smooth_enabled": layer["smooth_enabled"],
                "smooth_scale": layer["smooth_scale"],
                "act_scale": selected["act_scale"],
                "act_zero_point": selected["act_zero_point"],
                "clip_mode": args.clip_mode,
                "clip_min": selected["clip_min"],
                "clip_max": selected["clip_max"],
                "clipped_ratio": selected["clipped_ratio"],
            }
        )

    out = {
        "schema": "aicas.mmproj.smoothquant_policy.v1",
        "candidates": str(Path(args.candidates).resolve()),
        "alpha": float(args.alpha),
        "clip_mode": args.clip_mode,
        "act_quant_mode": doc.get("act_quant_mode", "asymmetric_u8"),
        "layers": layers_out,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote SmoothQuant policy: {output}")
    print(f"Layers: {len(layers_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
