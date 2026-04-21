#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 mmproj mulmat 层分组目录。")
    parser.add_argument("--layer-manifest", required=True, help="layer_manifest.json 路径")
    parser.add_argument("--inventory", required=True, help="mmproj_mulmat_inventory.json 路径")
    parser.add_argument(
        "--output",
        default="AICAS/artifacts/mmproj_group_catalog.json",
        help="输出分组目录 JSON 路径",
    )
    return parser.parse_args()


def _kind_priority(kind: str) -> str:
    mapping = {
        "projector": "高风险/高收益",
        "ffn_up": "高收益但疑似敏感",
        "ffn_down": "高收益低优先级筛查",
        "attn_k": "高优先级筛查",
        "attn_out": "高收益但已知常回退",
        "attn_q": "中优先级筛查",
        "attn_v": "中优先级筛查",
    }
    return mapping.get(kind, "待确认")


def _coarse_rank(kind: str) -> int:
    order = {
        "projector": 0,
        "ffn_up": 1,
        "ffn_down": 2,
        "attn_k": 3,
        "attn_out": 4,
        "attn_q": 5,
        "attn_v": 6,
    }
    return order.get(kind, 999)


def _shape_text(in_features: int, out_features: int) -> str:
    return f"{in_features}x{out_features}"


def _build_group(
    *,
    group_id: str,
    level: str,
    module: str,
    kind: str,
    block_ids: list[int],
    layers: list[dict],
    matched_signatures: dict,
) -> dict:
    param_count = 0
    shape_set = set()
    node_event_count = 0
    src1_shapes = set()
    dst_shapes = set()
    tensor_names = []

    for layer in layers:
        param_count += int(layer["param_count"])
        shape_set.add(layer["shape"])
        tensor_names.append(layer["tensor_name"])
        sig = matched_signatures.get(layer["tensor_name"], {})
        node_event_count += int(sig.get("node_event_count", 0))
        src1_shape = sig.get("src1_shape") or []
        dst_shape = sig.get("dst_shape") or []
        if src1_shape:
            src1_shapes.add(tuple(src1_shape))
        if dst_shape:
            dst_shapes.add(tuple(dst_shape))

    return {
        "group_id": group_id,
        "level": level,
        "module": module,
        "kind": kind,
        "block_ids": sorted(block_ids),
        "tensor_names": tensor_names,
        "layer_count": len(layers),
        "param_count": param_count,
        "shape_set": sorted(shape_set),
        "matched_node_event_count": node_event_count,
        "src1_shapes": [list(item) for item in sorted(src1_shapes)],
        "dst_shapes": [list(item) for item in sorted(dst_shapes)],
        "is_mulmat": True,
        "decode_gemv": False,
        "likely_prefill_main_path": True,
        "priority_hint": _kind_priority(kind),
    }


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.layer_manifest).resolve()
    inventory_path = Path(args.inventory).resolve()
    output_path = Path(args.output).resolve()

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    with inventory_path.open("r", encoding="utf-8") as f:
        inventory = json.load(f)

    matched_signatures = inventory.get("profile", {}).get("matched_signatures", {})
    raw_layers = manifest.get("layers", [])
    if not raw_layers:
        raise SystemExit("layer_manifest.json 中没有 layers[]")

    layers = []
    for layer in raw_layers:
        module = "projector" if layer["kind"] == "projector" else "vision_blocks"
        block_id = int(layer["layer_index"])
        in_features = int(layer["in_features"])
        out_features = int(layer["out_features"])
        param_count = in_features * out_features
        layers.append(
            {
                "id": int(layer["id"]),
                "tensor_name": layer["tensor_name"],
                "module": module,
                "kind": layer["kind"],
                "block_id": block_id,
                "in_features": in_features,
                "out_features": out_features,
                "shape": _shape_text(in_features, out_features),
                "param_count": param_count,
                "is_mulmat": True,
                "decode_gemv": False,
                "likely_prefill_main_path": True,
                "priority_hint": _kind_priority(layer["kind"]),
                "matched_node_event_count": int(
                    matched_signatures.get(layer["tensor_name"], {}).get("node_event_count", 0)
                ),
            }
        )

    module_map: dict[str, list[dict]] = defaultdict(list)
    kind_map: dict[str, list[dict]] = defaultdict(list)
    block_map: dict[int, list[dict]] = defaultdict(list)
    block_kind_map: dict[tuple[int, str], list[dict]] = defaultdict(list)

    for layer in layers:
        module_map[layer["module"]].append(layer)
        kind_map[layer["kind"]].append(layer)
        block_map[layer["block_id"]].append(layer)
        block_kind_map[(layer["block_id"], layer["kind"])].append(layer)

    groups = []

    for module, grouped_layers in sorted(module_map.items()):
        group_id = f"module:{module}"
        groups.append(
            _build_group(
                group_id=group_id,
                level="module",
                module=module,
                kind=module if module == "projector" else "mixed",
                block_ids=[layer["block_id"] for layer in grouped_layers if layer["block_id"] >= 0],
                layers=grouped_layers,
                matched_signatures=matched_signatures,
            )
        )

    for kind, grouped_layers in sorted(kind_map.items(), key=lambda item: (_coarse_rank(item[0]), item[0])):
        group_id = f"kind:{kind}_all"
        groups.append(
            _build_group(
                group_id=group_id,
                level="kind",
                module="projector" if kind == "projector" else "vision_blocks",
                kind=kind,
                block_ids=[layer["block_id"] for layer in grouped_layers if layer["block_id"] >= 0],
                layers=grouped_layers,
                matched_signatures=matched_signatures,
            )
        )

    for block_id, grouped_layers in sorted(block_map.items()):
        if block_id < 0:
            continue
        groups.append(
            _build_group(
                group_id=f"block:blk_{block_id:02d}_all",
                level="block",
                module="vision_blocks",
                kind="mixed",
                block_ids=[block_id],
                layers=sorted(grouped_layers, key=lambda item: (_coarse_rank(item["kind"]), item["tensor_name"])),
                matched_signatures=matched_signatures,
            )
        )

    for (block_id, kind), grouped_layers in sorted(block_kind_map.items(), key=lambda item: (item[0][0], _coarse_rank(item[0][1]))):
        if block_id < 0:
            continue
        groups.append(
            _build_group(
                group_id=f"block_kind:blk_{block_id:02d}.{kind}",
                level="block_kind",
                module="vision_blocks",
                kind=kind,
                block_ids=[block_id],
                layers=grouped_layers,
                matched_signatures=matched_signatures,
            )
        )

    catalog = {
        "schema": "aicas.mmproj.group_catalog.v1",
        "layer_manifest": str(manifest_path),
        "inventory": str(inventory_path),
        "model": manifest.get("model", {}),
        "summary": {
            "layer_count": len(layers),
            "module_group_count": 2,
            "kind_group_count": len(kind_map),
            "block_group_count": len([key for key in block_map if key >= 0]),
            "block_kind_group_count": len([key for key in block_kind_map if key[0] >= 0]),
            "total_param_count": sum(layer["param_count"] for layer in layers),
        },
        "layers": layers,
        "groups": groups,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print(f"Wrote group catalog: {output_path}")
    print(f"Layers: {len(layers)} | Groups: {len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
