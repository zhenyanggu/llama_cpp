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
    parser = argparse.ArgumentParser(description="Prepare decode-only AWQ candidates on top of GGUF text weights.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--act-stats", required=True)
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--output", default="AICAS/artifacts/text_decode_awq_candidates.json")
    parser.add_argument("--alpha-grid", default="0.25,0.5")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument(
        "--missing-stat-strategy",
        choices=["skip", "same_kind_prev_layer"],
        default="same_kind_prev_layer",
    )
    return parser.parse_args()


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_alpha_grid(text: str) -> list[float]:
    return [float(token.strip()) for token in text.split(",") if token.strip()]


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


def normalize_awq_scale(scale: np.ndarray, eps: float) -> np.ndarray:
    scale = np.maximum(scale.astype(np.float32), eps)
    scale_min = float(np.min(scale))
    scale_max = float(np.max(scale))
    norm = max((scale_min * scale_max) ** 0.5, eps)
    return np.asarray(scale / norm, dtype=np.float32)


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
            act_absmax = np.asarray(stat["per_channel_absmax"], dtype=np.float32)
            weight_absmax = np.max(np.abs(weight), axis=0).astype(np.float32)
            smooth_scale = np.power(np.maximum(act_absmax, args.eps), alpha)
            smooth_scale /= np.power(np.maximum(weight_absmax, args.eps), 1.0 - alpha) + args.eps
            smooth_scale = normalize_awq_scale(smooth_scale, args.eps)

            layers.append(
                {
                    "tensor_name": name,
                    "kind": item["kind"],
                    "layer_index": item["layer_index"],
                    "enabled": True,
                    "policy": "Q4_AWQ",
                    "reason": "decode_gemv_only",
                    "stat_source": stat_source,
                    "alpha": alpha,
                    "eps": args.eps,
                    "group_size": args.group_size,
                    "weight_bits": 4,
                    "smooth_scale": smooth_scale.tolist(),
                    "in_features": int(weight.shape[1]),
                    "out_features": int(weight.shape[0]),
                }
            )
        alpha_results.append({"alpha": alpha, "layers": layers})

    out = {
        "schema": "aicas.llama.text_decode_awq_candidates.v1",
        "manifest": str(Path(args.manifest).resolve()),
        "act_stats": str(Path(args.act_stats).resolve()),
        "model_gguf": str(Path(args.model_gguf).resolve()),
        "alpha_grid": alpha_grid,
        "group_size": args.group_size,
        "missing_stat_strategy": args.missing_stat_strategy,
        "layer_resolution": layer_resolution,
        "alpha_results": alpha_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote decode AWQ candidates: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
