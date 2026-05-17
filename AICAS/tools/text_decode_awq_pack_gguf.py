#!/usr/bin/env python3
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "gguf-py"))
import gguf  # type: ignore


@dataclass
class LayerPolicy:
    tensor_name: str
    enabled: bool
    policy: str
    reason: str
    alpha: float
    eps: float
    group_size: int
    weight_bits: int
    smooth_scale: np.ndarray
    in_features: int
    out_features: int
    stat_source: str
    quant_tensor_name: str
    scale_tensor_name: str
    zero_tensor_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack decode-only AWQ GGUF on top of an existing runtime GGUF.")
    parser.add_argument("--input-gguf", required=True)
    parser.add_argument("--source-weights-gguf", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-gguf", required=True)
    parser.add_argument("--output-summary", default="AICAS/artifacts/text_decode_awq_pack_summary.json")
    parser.add_argument("--min-scale", type=float, default=1e-8)
    parser.add_argument("--scale-dtype", choices=["f32", "f16"], default="f32")
    parser.add_argument(
        "--override-group-size",
        type=int,
        default=None,
        help="Override the AWQ group size stored in the policy before packing.",
    )
    return parser.parse_args()


def load_policy(path: str) -> list[LayerPolicy]:
    doc = json.load(open(path, "r", encoding="utf-8"))
    layers = []
    for item in doc["layers"]:
        tensor_name = str(item["tensor_name"])
        layers.append(
            LayerPolicy(
                tensor_name=tensor_name,
                enabled=bool(item["enabled"]),
                policy=str(item["policy"]),
                reason=str(item.get("reason", "")),
                alpha=float(item["alpha"]),
                eps=float(item["eps"]),
                group_size=int(item["group_size"]),
                weight_bits=int(item["weight_bits"]),
                smooth_scale=np.asarray(item["smooth_scale"], dtype=np.float32),
                in_features=int(item["in_features"]),
                out_features=int(item["out_features"]),
                stat_source=str(item.get("stat_source", "observed")),
                quant_tensor_name=str(item.get("quant_tensor_name", tensor_name + ".aicas_awq_q4")),
                scale_tensor_name=str(item.get("scale_tensor_name", tensor_name + ".aicas_awq_scale")),
                zero_tensor_name=str(item.get("zero_tensor_name", tensor_name + ".aicas_awq_zero")),
            )
        )
    return layers


def copy_metadata(reader: gguf.GGUFReader, writer: gguf.GGUFWriter) -> None:
    for field in reader.fields.values():
        if field.name.startswith("GGUF."):
            continue
        if field.name == gguf.Keys.General.ARCHITECTURE:
            continue
        if field.name.startswith("aicas.text_decode_awq."):
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub_type)


def quantize_groupwise_q4(
    weight: np.ndarray,
    group_size: int,
    min_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    out_features, in_features = weight.shape
    groups = (in_features + group_size - 1) // group_size
    packed = np.zeros((out_features, (in_features + 1) // 2), dtype=np.uint8)
    scales = np.zeros((out_features, groups), dtype=np.float32)
    zeros = np.zeros((out_features, groups), dtype=np.float32)

    for row in range(out_features):
        for group in range(groups):
            start = group * group_size
            end = min(in_features, start + group_size)
            segment = weight[row, start:end]
            w_min = float(np.min(segment))
            w_max = float(np.max(segment))
            scale = max((w_max - w_min) / 15.0, min_scale)
            zero = float(np.clip(np.round(-w_min / scale), 0, 15))
            q = np.clip(np.round(segment / scale + zero), 0, 15).astype(np.uint8)
            scales[row, group] = scale
            zeros[row, group] = zero
            for local_idx, qv in enumerate(q):
                idx = start + local_idx
                dst = idx // 2
                if (idx & 1) == 0:
                    packed[row, dst] = (packed[row, dst] & 0xF0) | int(qv)
                else:
                    packed[row, dst] = (packed[row, dst] & 0x0F) | (int(qv) << 4)

    return packed.view(np.int8), scales, zeros


def main() -> int:
    args = parse_args()
    layers = load_policy(args.policy)
    if args.override_group_size is not None:
        if args.override_group_size <= 0:
            raise RuntimeError("--override-group-size must be positive")
        for layer in layers:
            layer.group_size = args.override_group_size
    reader = gguf.GGUFReader(args.input_gguf, "r")
    source_reader = gguf.GGUFReader(args.source_weights_gguf, "r")
    source_tensor_map = {tensor.name: tensor for tensor in source_reader.tensors}
    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    writer = gguf.GGUFWriter(args.output_gguf, arch=arch, endianess=reader.endianess)

    alignment = reader.get_field(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    copy_metadata(reader, writer)
    writer.add_string("aicas.text_decode_awq.schema", "llama_text_decode_awq_v1")
    writer.add_int32("aicas.text_decode_awq.tensor_count", len(layers))

    staged: list[np.ndarray] = []
    summary = {
        "schema": "aicas.llama.text_decode_awq_pack_summary.v1",
        "input_gguf": os.path.abspath(args.input_gguf),
        "source_weights_gguf": os.path.abspath(args.source_weights_gguf),
        "policy": os.path.abspath(args.policy),
        "scale_dtype": args.scale_dtype,
        "override_group_size": args.override_group_size,
        "layers": [],
    }

    for tensor in reader.tensors:
        arr = np.ascontiguousarray(np.asarray(tensor.data))
        writer.add_tensor_info(tensor.name, arr.shape, arr.dtype, arr.nbytes, raw_dtype=tensor.tensor_type)
        staged.append(arr)

    for index, layer in enumerate(layers):
        prefix = f"aicas.text_decode_awq.tensor.{index}."
        writer.add_string(prefix + "name", layer.tensor_name)
        writer.add_bool(prefix + "enabled", layer.enabled)
        writer.add_string(prefix + "policy", layer.policy)
        writer.add_int32(prefix + "weight_bits", layer.weight_bits)
        writer.add_int32(prefix + "group_size", layer.group_size)
        writer.add_int32(prefix + "in_features", layer.in_features)
        writer.add_float32(prefix + "smooth_alpha", layer.alpha)
        writer.add_float32(prefix + "smooth_eps", layer.eps)
        writer.add_array(prefix + "smooth_scale", layer.smooth_scale.astype(np.float32).tolist())
        writer.add_string(prefix + "quant_tensor_name", layer.quant_tensor_name)
        writer.add_string(prefix + "scale_tensor_name", layer.scale_tensor_name)
        writer.add_string(prefix + "zero_tensor_name", layer.zero_tensor_name)

        entry = {
            "tensor_name": layer.tensor_name,
            "enabled": layer.enabled,
            "policy": layer.policy,
            "reason": layer.reason,
            "group_size": layer.group_size,
            "weight_bits": layer.weight_bits,
            "stat_source": layer.stat_source,
        }

        if layer.enabled and layer.policy == "Q4_AWQ":
            source = np.asarray(source_tensor_map[layer.tensor_name].data, dtype=np.float32)
            if source.shape != (layer.out_features, layer.in_features):
                raise RuntimeError(f"shape mismatch for {layer.tensor_name}: {source.shape} vs expected {(layer.out_features, layer.in_features)}")
            scaled = np.asarray(source * layer.smooth_scale[np.newaxis, :], dtype=np.float32)
            packed, scales, zeros = quantize_groupwise_q4(scaled, layer.group_size, args.min_scale)
            scales_out = scales.astype(np.float16) if args.scale_dtype == "f16" else scales

            writer.add_tensor_info(layer.quant_tensor_name, packed.shape, packed.dtype, packed.nbytes, raw_dtype=None)
            writer.add_tensor_info(layer.scale_tensor_name, scales_out.shape, scales_out.dtype, scales_out.nbytes, raw_dtype=None)
            writer.add_tensor_info(layer.zero_tensor_name, zeros.shape, zeros.dtype, zeros.nbytes, raw_dtype=None)
            staged.append(np.ascontiguousarray(packed))
            staged.append(np.ascontiguousarray(scales_out))
            staged.append(np.ascontiguousarray(zeros))
            entry["groups"] = int(scales.shape[1])
            entry["scale_dtype"] = args.scale_dtype

        summary["layers"].append(entry)

    Path(args.output_gguf).parent.mkdir(parents=True, exist_ok=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for data in staged:
        writer.write_tensor_data(data)
    writer.close()

    output_summary = Path(args.output_summary)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(output_summary, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"Wrote decode AWQ GGUF: {args.output_gguf}")
    print(f"Wrote pack summary: {args.output_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
