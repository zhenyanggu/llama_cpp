#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "gguf-py")))
from gguf import GGUFReader  # type: ignore

ROLE_ORDER = {
    "attn_q": 0,
    "attn_k": 1,
    "attn_v": 2,
    "attn_out": 3,
    "ffn_up": 4,
    "ffn_down": 5,
    "projector": 6,
}

V_BLOCK_RE = re.compile(r"^v\.blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_out|ffn_up|ffn_down)\.weight$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mmproj mul_mat manifest for SmolVLM2 IDEFICS3.")
    p.add_argument("--mmproj-gguf", required=True, help="Input mmproj GGUF")
    p.add_argument("--profile-json", default="", help="Optional mmproj profiler JSON")
    p.add_argument("--output-manifest", default="AICAS/artifacts/layer_manifest.json")
    p.add_argument("--output-inventory", default="AICAS/artifacts/mmproj_mulmat_inventory.json")
    return p.parse_args()


def gguf_field_contents(reader: GGUFReader, key: str):
    field = reader.get_field(key)
    return field.contents() if field is not None else None


def build_manifest(reader: GGUFReader) -> list[dict]:
    tensors = list(reader.tensors)
    tensor_map = {t.name: t for t in tensors}
    entries = []

    for t in tensors:
        if t.name == "mm.model.fc.weight":
            bias_name = "mm.model.fc.bias"
            entries.append(
                {
                    "tensor_name": t.name,
                    "kind": "projector",
                    "layer_index": -1,
                    "in_features": int(t.shape[0]),
                    "out_features": int(t.shape[1]),
                    "has_bias": bias_name in tensor_map,
                    "bias_name": bias_name if bias_name in tensor_map else "",
                    "enabled": True,
                    "policy": "W8A8",
                }
            )
            continue

        m = V_BLOCK_RE.match(t.name)
        if not m:
            continue

        block = int(m.group(1))
        kind = m.group(2)
        bias_name = t.name.replace(".weight", ".bias")
        entries.append(
            {
                "tensor_name": t.name,
                "kind": kind,
                "layer_index": block,
                "in_features": int(t.shape[0]),
                "out_features": int(t.shape[1]),
                "has_bias": bias_name in tensor_map,
                "bias_name": bias_name if bias_name in tensor_map else "",
                "enabled": True,
                "policy": "W8A8",
            }
        )

    entries.sort(key=lambda x: (x["layer_index"], ROLE_ORDER.get(x["kind"], 99), x["tensor_name"]))
    for i, e in enumerate(entries):
        e["id"] = i
    return entries


def build_inventory(entries: list[dict], profile_json_path: str) -> dict:
    inventory = OrderedDict()
    inventory["target_tensor_count"] = len(entries)
    inventory["target_tensors"] = [e["tensor_name"] for e in entries]

    if not profile_json_path:
        inventory["profile"] = {"available": False}
        return inventory

    with open(profile_json_path, "r", encoding="utf-8") as f:
        prof = json.load(f)

    signatures = prof.get("mul_mat_signatures", [])
    per_weight = {}
    ignored = []
    target_set = {e["tensor_name"] for e in entries}

    for sig in signatures:
        src0 = (sig.get("src0") or {}).get("name", "")
        row = {
            "src0_name": src0,
            "node_event_count": int(sig.get("node_event_count", 0)),
            "src0_shape": (sig.get("src0") or {}).get("shape", []),
            "src1_shape": (sig.get("src1") or {}).get("shape", []),
            "dst_shape": (sig.get("dst") or {}).get("shape", []),
            "example_node_names": sig.get("example_node_names", []),
        }
        if src0 in target_set:
            per_weight[src0] = row
        else:
            ignored.append(row)

    inventory["profile"] = {
        "available": True,
        "mul_mat_total_us": prof.get("mul_mat_total_us", 0.0),
        "mul_mat_share_of_total_time_pct": prof.get("mul_mat_share_of_total_time_pct", 0.0),
        "matched_signatures": per_weight,
        "ignored_signatures": ignored,
    }
    return inventory


def main() -> int:
    args = parse_args()

    reader = GGUFReader(args.mmproj_gguf)
    architecture = gguf_field_contents(reader, "general.architecture")
    model_type = gguf_field_contents(reader, "general.type")
    projector_type = gguf_field_contents(reader, "clip.projector_type")

    entries = build_manifest(reader)
    manifest = OrderedDict()
    manifest["schema"] = "aicas.mmproj.mul_mat_manifest.v1"
    manifest["source_mmproj_gguf"] = os.path.abspath(args.mmproj_gguf)
    manifest["model"] = {
        "architecture": architecture,
        "type": model_type,
        "projector_type": projector_type,
    }
    manifest["layer_count"] = len(entries)
    manifest["layers"] = entries

    inventory = build_inventory(entries, args.profile_json)

    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_inventory).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(args.output_inventory, "w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)

    print(f"Wrote manifest: {args.output_manifest}")
    print(f"Wrote inventory: {args.output_inventory}")
    print(f"Target tensors: {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
