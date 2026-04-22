#!/usr/bin/env python3
import argparse
import json
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
from gguf import GGUFReader  # type: ignore


ATTN_RE = re.compile(r"^blk\.(\d+)\.(attn_q|attn_k|attn_v|attn_output)\.weight$")
FFN_RE = re.compile(r"^blk\.(\d+)\.(ffn_gate|ffn_up|ffn_down)\.weight$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build text-model SmoothQuant candidate manifest from GGUF.")
    parser.add_argument("--model-gguf", required=True)
    parser.add_argument("--output-manifest", default="AICAS/artifacts/text_sq_manifest.json")
    parser.add_argument("--output-summary", default="AICAS/artifacts/text_sq_manifest_summary.json")
    return parser.parse_args()


def classify_tensor(name: str) -> tuple[str, int, str]:
    if name == "output.weight":
        return "output", -1, "B"

    m = ATTN_RE.match(name)
    if m:
        return m.group(2), int(m.group(1)), "A"

    m = FFN_RE.match(name)
    if m:
        return m.group(2), int(m.group(1)), "A"

    return "skip", -1, "C"


def main() -> int:
    args = parse_args()
    reader = GGUFReader(args.model_gguf)

    entries = []
    summary = Counter()
    by_kind = Counter()
    for tensor in reader.tensors:
        name = tensor.name
        if len(tensor.shape) != 2:
            continue

        kind, layer_index, bucket = classify_tensor(name)
        if kind == "skip":
            continue

        np_shape = tensor.data.shape
        out_features = int(np_shape[0])
        in_features = int(np_shape[1])

        enabled = bucket == "A"
        reason = "prefill_gemm_only"
        if kind == "output":
            enabled = False
            reason = "skip_output_head_for_now"

        entries.append(
            {
                "tensor_name": name,
                "kind": kind,
                "layer_index": layer_index,
                "bucket": bucket,
                "enabled": enabled,
                "reason": reason,
                "in_features": in_features,
                "out_features": out_features,
                "param_count": in_features * out_features,
                "prefill_gemm_candidate": True,
                "decode_gemv_skip": True,
                "flash_attn_related": False,
            }
        )
        summary[bucket] += 1
        by_kind[kind] += 1

    entries.sort(key=lambda item: (item["layer_index"], item["kind"], item["tensor_name"]))
    for idx, item in enumerate(entries):
        item["id"] = idx

    manifest = OrderedDict()
    manifest["schema"] = "aicas.llama.text_smoothquant_manifest.v1"
    manifest["source_model_gguf"] = str(Path(args.model_gguf).resolve())
    manifest["layer_count"] = len(entries)
    manifest["layers"] = entries

    summary_doc = OrderedDict()
    summary_doc["schema"] = "aicas.llama.text_smoothquant_manifest_summary.v1"
    summary_doc["source_model_gguf"] = str(Path(args.model_gguf).resolve())
    summary_doc["layer_count"] = len(entries)
    summary_doc["bucket_counts"] = dict(summary)
    summary_doc["kind_counts"] = dict(by_kind)
    summary_doc["enabled_layers"] = sum(1 for item in entries if item["enabled"])
    summary_doc["disabled_layers"] = sum(1 for item in entries if not item["enabled"])

    output_manifest = Path(args.output_manifest)
    output_summary = Path(args.output_summary)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with output_summary.open("w", encoding="utf-8") as f:
        json.dump(summary_doc, f, indent=2, ensure_ascii=False)

    print(f"Wrote manifest: {output_manifest}")
    print(f"Wrote summary: {output_summary}")
    print(f"Layers: {len(entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
