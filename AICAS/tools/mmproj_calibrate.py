#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build static activation quantization params for mmproj layers.")
    p.add_argument("--layer-manifest", required=True, help="Path to layer_manifest.json")
    p.add_argument("--act-stats", required=True, help="Path to collected activation stats JSON")
    p.add_argument("--calib-manifest", default="", help="Optional calib_manifest.json for traceability")
    p.add_argument("--output", default="AICAS/artifacts/quant_params.json", help="Output JSON")
    p.add_argument(
        "--act-quant-mode",
        choices=["asymmetric_u8", "symmetric_u8"],
        default="asymmetric_u8",
        help="Activation quantization mode written into the output calibration JSON",
    )
    p.add_argument("--percentile-low", type=float, default=0.1, help="Lower percentile for asymmetric clipping")
    p.add_argument("--percentile-high", type=float, default=99.9, help="Upper percentile for asymmetric clipping")
    p.add_argument("--fallback-act-scale", type=float, default=0.02, help="Fallback activation scale when stats are missing")
    p.add_argument("--fallback-act-zero-point", type=int, default=128, help="Fallback activation zero-point when stats are missing")
    p.add_argument("--min-act-scale", type=float, default=1e-8, help="Minimum activation scale")
    return p.parse_args()


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_u8_params(samples: np.ndarray, p_low: float, p_high: float, min_scale: float) -> tuple[float, int, float, float, float]:
    if samples.size == 0:
        raise ValueError("empty samples")

    lo = float(np.percentile(samples, p_low))
    hi = float(np.percentile(samples, p_high))
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("non-finite percentile values")

    if hi <= lo:
        hi = lo + min_scale * 255.0

    scale = max((hi - lo) / 255.0, min_scale)
    zp = int(round(-lo / scale))
    zp = max(0, min(255, zp))

    q_low = -zp * scale
    q_high = (255 - zp) * scale
    clipped = float(np.mean((samples < q_low) | (samples > q_high))) if samples.size else 0.0
    return scale, zp, q_low, q_high, clipped


def _compute_symmetric_u8_params(samples: np.ndarray, p_low: float, p_high: float, min_scale: float) -> tuple[float, int, float, float, float]:
    if samples.size == 0:
        raise ValueError("empty samples")

    lo = float(np.percentile(samples, p_low))
    hi = float(np.percentile(samples, p_high))
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("non-finite percentile values")

    max_abs = max(abs(lo), abs(hi))
    scale = max(max_abs / 127.0, min_scale)
    zp = 128
    q_low = -128.0 * scale
    q_high = 127.0 * scale
    clipped = float(np.mean((samples < q_low) | (samples > q_high))) if samples.size else 0.0
    return scale, zp, q_low, q_high, clipped


def main() -> int:
    args = parse_args()
    manifest = _load_json(args.layer_manifest)
    stats_doc = _load_json(args.act_stats)
    fallback_zero_point = 128 if args.act_quant_mode == "symmetric_u8" else args.fallback_act_zero_point

    stats_map = {item["tensor_name"]: item for item in stats_doc.get("tensors", [])}

    layers_out = []
    for layer in manifest.get("layers", []):
        tensor_name = layer["tensor_name"]
        enabled = bool(layer.get("enabled", True))
        base_policy = str(layer.get("policy", "W8A8"))
        stats = stats_map.get(tensor_name)

        if stats is None or not stats.get("samples"):
            layers_out.append(
                {
                    "id": layer["id"],
                    "tensor_name": tensor_name,
                    "enabled": False,
                    "policy": "F16_FALLBACK",
                    "act_scale": float(layer.get("act_scale", args.fallback_act_scale)),
                    "act_zero_point": int(layer.get("act_zero_point", fallback_zero_point)),
                    "act_quant_mode": args.act_quant_mode,
                    "observed_min": None,
                    "observed_max": None,
                    "clip_min": None,
                    "clip_max": None,
                    "sample_count": 0,
                    "observed_count": int(stats.get("count", 0)) if stats else 0,
                    "clipped_ratio": None,
                    "reason": "missing_stats",
                }
            )
            continue

        samples = np.asarray(stats["samples"], dtype=np.float32)
        if args.act_quant_mode == "symmetric_u8":
            scale, zp, clip_min, clip_max, clipped_ratio = _compute_symmetric_u8_params(
                samples,
                args.percentile_low,
                args.percentile_high,
                args.min_act_scale,
            )
        else:
            scale, zp, clip_min, clip_max, clipped_ratio = _compute_u8_params(
                samples,
                args.percentile_low,
                args.percentile_high,
                args.min_act_scale,
            )

        layers_out.append(
            {
                "id": layer["id"],
                "tensor_name": tensor_name,
                "enabled": enabled,
                "policy": base_policy if enabled else "F16_FALLBACK",
                "act_scale": float(scale),
                "act_zero_point": int(zp),
                "act_quant_mode": args.act_quant_mode,
                "observed_min": float(stats.get("min", float(samples.min()))),
                "observed_max": float(stats.get("max", float(samples.max()))),
                "clip_min": float(clip_min),
                "clip_max": float(clip_max),
                "sample_count": int(samples.size),
                "observed_count": int(stats.get("count", samples.size)),
                "clipped_ratio": float(clipped_ratio),
                "reason": "ok",
            }
        )

    out = {
        "schema": "aicas.mmproj.calibration.v2",
        "layer_manifest": str(Path(args.layer_manifest).resolve()),
        "act_stats": str(Path(args.act_stats).resolve()),
        "calib_manifest": str(Path(args.calib_manifest).resolve()) if args.calib_manifest else "",
        "percentile_low": args.percentile_low,
        "percentile_high": args.percentile_high,
        "fallback_act_scale": args.fallback_act_scale,
        "fallback_act_zero_point": fallback_zero_point,
        "act_quant_mode": args.act_quant_mode,
        "layers": layers_out,
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    ok_layers = sum(1 for layer in layers_out if layer["reason"] == "ok")
    fallback_layers = len(layers_out) - ok_layers
    print(f"Wrote calibration params: {path}")
    print(f"Layers: {len(layers_out)} | calibrated: {ok_layers} | fallback: {fallback_layers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
