#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFReader  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare text-model SmoothQuant candidates.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--act-stats", required=True)
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--output", default="AICAS/artifacts/text_sq_candidates.json")
    parser.add_argument("--alpha-grid", default="0.3,0.4,0.5,0.6,0.7")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--min-act-scale", type=float, default=1e-8)
    parser.add_argument(
        "--missing-stat-strategy",
        choices=["skip", "same_kind_prev_layer"],
        default="skip",
        help="How to handle manifest layers missing activation stats.",
    )
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_alpha_grid(text: str) -> list[float]:
    return [float(token.strip()) for token in text.split(",") if token.strip()]


def compute_u8_params(lo: float, hi: float, min_scale: float) -> tuple[float, int]:
    lo = min(lo, 0.0)
    hi = max(hi, 0.0)
    if hi <= lo:
        hi = lo + min_scale * 255.0
    scale = max((hi - lo) / 255.0, min_scale)
    zp = int(round(-lo / scale))
    zp = max(0, min(255, zp))
    return float(scale), int(zp)


def resolve_stat_for_item(item: dict, stats_map: dict[str, dict], strategy: str) -> tuple[dict | None, str]:
    name = item["tensor_name"]
    if name in stats_map:
        return stats_map[name], "observed"
    if strategy == "skip":
        return None, "missing"

    if strategy == "same_kind_prev_layer":
        kind = item["kind"]
        layer_index = int(item["layer_index"])
        for prev_layer in range(layer_index - 1, -1, -1):
            candidate_name = f"blk.{prev_layer}.{kind}.weight"
            if candidate_name in stats_map:
                return stats_map[candidate_name], f"backfill:{candidate_name}"
    return None, "missing"


def main() -> int:
    args = parse_args()
    manifest = load_json(args.manifest)
    act_stats = load_json(args.act_stats)
    reader = GGUFReader(args.model_gguf)
    tensor_map = {tensor.name: tensor for tensor in reader.tensors}
    stats_map = {item["tensor_name"]: item for item in act_stats["tensors"]}
    alpha_grid = parse_alpha_grid(args.alpha_grid)

    alpha_results = []
    layer_resolution = []
    for alpha in alpha_grid:
        layers = []
        for item in manifest["layers"]:
            if not item["enabled"]:
                continue
            name = item["tensor_name"]
            stat, stat_source = resolve_stat_for_item(item, stats_map, args.missing_stat_strategy)
            if stat is None:
                if alpha == alpha_grid[0]:
                    layer_resolution.append(
                        {
                            "tensor_name": name,
                            "kind": item["kind"],
                            "layer_index": item["layer_index"],
                            "status": "missing",
                            "stat_source": stat_source,
                        }
                    )
                continue
            if alpha == alpha_grid[0]:
                layer_resolution.append(
                    {
                        "tensor_name": name,
                        "kind": item["kind"],
                        "layer_index": item["layer_index"],
                        "status": "ready",
                        "stat_source": stat_source,
                    }
                )
            tensor = tensor_map[name]
            weight = np.asarray(tensor.data, dtype=np.float32)
            out_features = int(weight.shape[0])
            in_features = int(weight.shape[1])

            act_absmax = np.asarray(stat["per_channel_absmax"], dtype=np.float32)
            act_min = np.asarray(stat["per_channel_min"], dtype=np.float32)
            act_max = np.asarray(stat["per_channel_max"], dtype=np.float32)
            weight_absmax = np.max(np.abs(weight), axis=0).astype(np.float32)
            smooth_scale = np.power(np.maximum(act_absmax, args.eps), alpha).astype(np.float32)
            smooth_scale /= np.power(np.maximum(weight_absmax, args.eps), 1.0 - alpha).astype(np.float32) + args.eps
            smooth_scale = np.maximum(smooth_scale, args.eps)

            smooth_min = float(np.min(act_min / smooth_scale))
            smooth_max = float(np.max(act_max / smooth_scale))
            act_scale, act_zero_point = compute_u8_params(smooth_min, smooth_max, args.min_act_scale)

            samples = np.asarray(stat.get("samples", []), dtype=np.float32)
            sample_channels = np.asarray(stat.get("sample_channels", []), dtype=np.int64)
            if samples.size > 0 and sample_channels.size == samples.size:
                smoothed_samples = samples / smooth_scale[sample_channels]
                pct_lo = float(np.percentile(smoothed_samples, 100.0 - args.percentile))
                pct_hi = float(np.percentile(smoothed_samples, args.percentile))
                pct_scale, pct_zp = compute_u8_params(pct_lo, pct_hi, args.min_act_scale)
                clipped_ratio = float(np.mean((smoothed_samples < pct_lo) | (smoothed_samples > pct_hi)))
            else:
                pct_lo, pct_hi = smooth_min, smooth_max
                pct_scale, pct_zp = act_scale, act_zero_point
                clipped_ratio = 0.0

            layers.append(
                {
                    "tensor_name": name,
                    "kind": item["kind"],
                    "layer_index": item["layer_index"],
                    "enabled": True,
                    "policy": "W8A8",
                    "reason": "prefill_gemm_only",
                    "stat_source": stat_source,
                    "alpha": alpha,
                    "eps": args.eps,
                    "smooth_scale": smooth_scale.astype(np.float32).tolist(),
                    "act_quant_mode": "asymmetric_u8",
                    "in_features": in_features,
                    "out_features": out_features,
                    "minmax": {
                        "act_scale": act_scale,
                        "act_zero_point": act_zero_point,
                        "clip_min": smooth_min,
                        "clip_max": smooth_max,
                    },
                    "percentile": {
                        "act_scale": pct_scale,
                        "act_zero_point": pct_zp,
                        "clip_min": pct_lo,
                        "clip_max": pct_hi,
                        "clipped_ratio": clipped_ratio,
                    },
                }
            )
        alpha_results.append({"alpha": alpha, "layers": layers})

    out = {
        "schema": "aicas.llama.text_sq_candidates.v1",
        "manifest": str(Path(args.manifest).resolve()),
        "act_stats": str(Path(args.act_stats).resolve()),
        "model_gguf": str(Path(args.model_gguf).resolve()),
        "alpha_grid": alpha_grid,
        "missing_stat_strategy": args.missing_stat_strategy,
        "layer_resolution": layer_resolution,
        "alpha_results": alpha_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote text SmoothQuant candidates: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
