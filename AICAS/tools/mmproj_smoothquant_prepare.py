#!/usr/bin/env python3
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFReader  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SmoothQuant calibration candidates for mmproj layers.")
    parser.add_argument("--layer-manifest", required=True)
    parser.add_argument("--act-stats", required=True)
    parser.add_argument("--mmproj-gguf", required=True)
    parser.add_argument("--output", default="AICAS/artifacts/smoothquant_candidates.json")
    parser.add_argument("--alpha-grid", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--act-quant-mode", choices=["asymmetric_u8"], default="asymmetric_u8")
    parser.add_argument("--min-act-scale", type=float, default=1e-8)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_alpha_grid(text: str) -> list[float]:
    out = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    if not out:
        raise SystemExit("empty --alpha-grid")
    return out


def compute_u8_params(lo: float, hi: float, min_scale: float) -> tuple[float, int, float, float]:
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("non-finite range")
    if hi <= lo:
        hi = lo + min_scale * 255.0
    scale = max((hi - lo) / 255.0, min_scale)
    zero_point = int(round(-lo / scale))
    zero_point = max(0, min(255, zero_point))
    q_lo = -zero_point * scale
    q_hi = (255 - zero_point) * scale
    return float(scale), int(zero_point), float(q_lo), float(q_hi)


def main() -> int:
    args = parse_args()
    manifest = load_json(args.layer_manifest)
    act_stats = load_json(args.act_stats)
    reader = GGUFReader(args.mmproj_gguf)
    tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    stats_map = {item["tensor_name"]: item for item in act_stats.get("tensors", [])}
    alpha_grid = parse_alpha_grid(args.alpha_grid)

    alpha_results: list[dict] = []
    for alpha in alpha_grid:
        layers_out = []
        for layer in manifest.get("layers", []):
            name = layer["tensor_name"]
            stat = stats_map.get(name)
            if stat is None:
                raise RuntimeError(f"missing activation stats for {name}")
            tensor = tensor_map.get(name)
            if tensor is None:
                raise RuntimeError(f"missing tensor in gguf: {name}")

            weight = np.asarray(tensor.data, dtype=np.float32)
            out_features = int(layer["out_features"])
            in_features = int(layer["in_features"])
            if weight.ndim != 2 or weight.shape != (out_features, in_features):
                raise RuntimeError(
                    f"{name}: unexpected weight shape {weight.shape}, expected {(out_features, in_features)}"
                )

            act_absmax = np.asarray(stat.get("per_channel_absmax", []), dtype=np.float32)
            act_min = np.asarray(stat.get("per_channel_min", []), dtype=np.float32)
            act_max = np.asarray(stat.get("per_channel_max", []), dtype=np.float32)
            if act_absmax.size != in_features or act_min.size != in_features or act_max.size != in_features:
                raise RuntimeError(
                    f"{name}: activation channel stats length mismatch, "
                    f"expected {in_features}, got abs={act_absmax.size}, min={act_min.size}, max={act_max.size}"
                )

            weight_absmax = np.max(np.abs(weight), axis=0).astype(np.float32)
            act_term = np.power(np.maximum(act_absmax, args.eps), alpha).astype(np.float32)
            weight_term = np.power(np.maximum(weight_absmax, args.eps), 1.0 - alpha).astype(np.float32)
            smooth_scale = np.maximum(act_term / (weight_term + args.eps), args.eps).astype(np.float32)

            smooth_min_per_channel = act_min / smooth_scale
            smooth_max_per_channel = act_max / smooth_scale
            smooth_min = float(np.min(smooth_min_per_channel))
            smooth_max = float(np.max(smooth_max_per_channel))
            minmax_scale, minmax_zero_point, minmax_q_lo, minmax_q_hi = compute_u8_params(
                smooth_min,
                smooth_max,
                args.min_act_scale,
            )

            sample_values = np.asarray(stat.get("samples", []), dtype=np.float32)
            sample_channels = np.asarray(stat.get("sample_channels", []), dtype=np.int64)
            if sample_values.size != sample_channels.size:
                raise RuntimeError(
                    f"{name}: sample values/channel ids mismatch: {sample_values.size} vs {sample_channels.size}"
                )
            if sample_values.size > 0:
                smoothed_samples = sample_values / smooth_scale[sample_channels]
                pct_lo = float(np.percentile(smoothed_samples, 100.0 - args.percentile))
                pct_hi = float(np.percentile(smoothed_samples, args.percentile))
                pct_scale, pct_zero_point, pct_q_lo, pct_q_hi = compute_u8_params(
                    pct_lo,
                    pct_hi,
                    args.min_act_scale,
                )
                pct_clipped_ratio = float(np.mean((smoothed_samples < pct_q_lo) | (smoothed_samples > pct_q_hi)))
            else:
                smoothed_samples = np.asarray([], dtype=np.float32)
                pct_lo = smooth_min
                pct_hi = smooth_max
                pct_scale, pct_zero_point, pct_q_lo, pct_q_hi = minmax_scale, minmax_zero_point, minmax_q_lo, minmax_q_hi
                pct_clipped_ratio = 0.0

            layers_out.append(
                {
                    "id": int(layer["id"]),
                    "tensor_name": name,
                    "kind": layer["kind"],
                    "layer_index": int(layer["layer_index"]),
                    "in_features": in_features,
                    "out_features": out_features,
                    "enabled": True,
                    "policy": "W8A8",
                    "scheme": "smoothquant_v1",
                    "act_quant_mode": args.act_quant_mode,
                    "alpha": float(alpha),
                    "eps": float(args.eps),
                    "smooth_enabled": True,
                    "smooth_scale": smooth_scale.astype(np.float32).tolist(),
                    "weight_input_absmax": weight_absmax.astype(np.float32).tolist(),
                    "activation_input_absmax": act_absmax.astype(np.float32).tolist(),
                    "raw_activation_min": float(stat["min"]),
                    "raw_activation_max": float(stat["max"]),
                    "smoothed_activation": {
                        "min": smooth_min,
                        "max": smooth_max,
                        "minmax": {
                            "act_scale": minmax_scale,
                            "act_zero_point": minmax_zero_point,
                            "clip_min": smooth_min,
                            "clip_max": smooth_max,
                            "quant_min": minmax_q_lo,
                            "quant_max": minmax_q_hi,
                            "clipped_ratio": 0.0,
                        },
                        "percentile": {
                            "percentile": float(args.percentile),
                            "act_scale": pct_scale,
                            "act_zero_point": pct_zero_point,
                            "clip_min": pct_lo,
                            "clip_max": pct_hi,
                            "quant_min": pct_q_lo,
                            "quant_max": pct_q_hi,
                            "clipped_ratio": pct_clipped_ratio,
                        },
                    },
                    "sample_count": int(sample_values.size),
                }
            )

        alpha_results.append(
            {
                "alpha": float(alpha),
                "layer_count": len(layers_out),
                "layers": layers_out,
            }
        )

    out = {
        "schema": "aicas.mmproj.smoothquant_candidates.v1",
        "layer_manifest": str(Path(args.layer_manifest).resolve()),
        "act_stats": str(Path(args.act_stats).resolve()),
        "mmproj_gguf": str(Path(args.mmproj_gguf).resolve()),
        "alpha_grid": alpha_grid,
        "eps": float(args.eps),
        "percentile": float(args.percentile),
        "act_quant_mode": args.act_quant_mode,
        "alpha_results": alpha_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote SmoothQuant candidates: {output}")
    print(f"Alphas: {len(alpha_results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
